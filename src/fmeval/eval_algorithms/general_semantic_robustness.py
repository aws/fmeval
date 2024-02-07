import functools
import itertools
import logging

import evaluate as hf_evaluate
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np

from fmeval import util
from fmeval.constants import (
    DatasetColumns,
    MEAN,
    BUTTER_FINGER,
    RANDOM_UPPER_CASE,
    WHITESPACE_ADD_REMOVE,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.data_loaders.util import get_dataset
from fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalScore,
    EvalOutput,
    DATASET_CONFIGS,
    EVAL_DATASETS,
    DEFAULT_PROMPT_TEMPLATE,
    get_default_prompt_template,
)
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmConfig, EvalAlgorithmInterface
from fmeval.eval_algorithms.semantic_perturbation_utils import (
    ButterFinger,
    RandomUpperCase,
    WhitespaceAddRemove,
    ButterFingerConfig,
    RandomUpperCaseConfig,
    WhitespaceAddRemoveConfig,
)
from fmeval.eval_algorithms.util import (
    validate_dataset,
    save_dataset,
    aggregate_evaluation_scores,
    generate_output_dataset_path,
    generate_prompt_column_for_dataset,
    generate_model_predict_response_for_dataset,
    verify_model_determinism,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.perf_util import timed_block
from fmeval.eval_algorithms.util import get_bert_score
from fmeval.constants import BertscoreModels, BERTSCORE_DEFAULT_MODEL
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModel

logger = logging.getLogger(__name__)

# All the perturbation types supported by this eval algo
PERTURBATION_TYPE_TO_HELPER_CLASS = {
    BUTTER_FINGER: ButterFinger,
    RANDOM_UPPER_CASE: RandomUpperCase,
    WHITESPACE_ADD_REMOVE: WhitespaceAddRemove,
}

WER_SCORE = "word_error_rate"
BERT_SCORE = "bertscore"


@dataclass(frozen=True)
class GeneralSemanticRobustnessConfig(EvalAlgorithmConfig):
    """
    Configuration for the general semantic robustness eval algorithm.

    :param perturbation_type: perturbation type for generating perturbed inputs
    :param num_perturbations: Number of perturbed inputs to be generated for robustness evaluation
    :param butter_finger_perturbation_prob: The probability that a given character will be perturbed. Used for
        butter_finger perturbation_type
    :param random_uppercase_corrupt_proportion: Fraction of characters to be changed to uppercase. Used for
        random_upper_case perturbation_type
    :param whitespace_remove_prob: Given a whitespace, remove it with this much probability. Used for
        whitespace_add_remove perturbation_type
    :param whitespace_add_prob: Given a non-whitespace, add a whitespace before it with this probability. Used for
        whitespace_add_remove perturbation_type
    :param model_type_for_bertscore: model to use for bert score
    """

    perturbation_type: str = BUTTER_FINGER
    num_perturbations: int = 5
    butter_finger_perturbation_prob: float = 0.1
    random_uppercase_corrupt_proportion: float = 0.1
    whitespace_remove_prob: float = 0.1
    whitespace_add_prob: float = 0.05
    model_type_for_bertscore: str = BERTSCORE_DEFAULT_MODEL

    def __post_init__(self):
        if self.perturbation_type not in PERTURBATION_TYPE_TO_HELPER_CLASS.keys():
            raise EvalAlgorithmClientError(
                f"Invalid perturbation type '{self.perturbation_type} requested, please "
                f"choose from acceptable values: {PERTURBATION_TYPE_TO_HELPER_CLASS.keys()}"
            )
        if not BertscoreModels.model_is_allowed(self.model_type_for_bertscore):
            raise EvalAlgorithmClientError(
                f"Invalid model_type_for_bertscore: {self.model_type_for_bertscore} requested in "
                f"GeneralSemanticRobustnessConfig, please choose from acceptable values: {BertscoreModels.model_list()}."
            )


class GeneralSemanticRobustness(EvalAlgorithmInterface):
    """
    Semantic Robustness Eval algorithm for General task LLMs.

    This evaluation measures how much the model output changes as a result of semantic preserving
    perturbations. Given the input, e.g., "A quick brown fox jumps over the lazy dog", the
    evaluation creates a perturbation that preserves the semantic meaning of the input e.g.,
    whitespace perturbation that changes the input text to "A q uick bro wn fox ju mps overthe lazy
    dog". The evaluation then measures how much the model output changes when prompted with the
    original vs. perturbed input.

    The output difference is measured using two metrics: the Word Error Rate
    (https://huggingface.co/spaces/evaluate-metric/wer) and the BERTScore (https://huggingface.co/spaces/evaluate-metric/bertscore)
    between the original and the perturbed outputs. WER measures syntactic differences, that is,
    changes in the words, while BERTScore measures semantic differences.

    Note: We only compute WER when the model output is deterministic. For non-deterministic models,
    even with the same prompt, the phrasing of the model output might change even if the meaning is
    the same, e.g., consider the outputs "highly recommend this movie" vs. "you should really watch
    this film".
    """

    def __init__(self, eval_algorithm_config: GeneralSemanticRobustnessConfig = GeneralSemanticRobustnessConfig()):
        """Default constructor

        :param eval_algorithm_config: General Semantic Robustness eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.eval_name = EvalAlgorithm.GENERAL_SEMANTIC_ROBUSTNESS.value
        self._eval_algorithm_config = eval_algorithm_config
        self._is_model_deterministic: Optional[bool] = None

        if self._eval_algorithm_config.perturbation_type == BUTTER_FINGER:
            self._perturbation_config = ButterFingerConfig(self._eval_algorithm_config.butter_finger_perturbation_prob)
        elif self._eval_algorithm_config.perturbation_type == RANDOM_UPPER_CASE:
            self._perturbation_config = RandomUpperCaseConfig(
                self._eval_algorithm_config.random_uppercase_corrupt_proportion
            )
        else:
            self._perturbation_config = WhitespaceAddRemoveConfig(
                self._eval_algorithm_config.whitespace_remove_prob, self._eval_algorithm_config.whitespace_add_prob
            )

        self._bertscore_helper_model = BertscoreHelperModel.remote(
            model_type=self._eval_algorithm_config.model_type_for_bertscore
        )

    def evaluate_sample(
        self,
        model_input: str,
        model: ModelRunner,
        model_output: Optional[str] = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> List[EvalScore]:  # type: ignore[override]
        """
        Semantic Robustness evaluate sample.

        :param model_input: text input for model
        :param model: An instance of ModelRunner which is the model under evaluation
        :param model_output: The output of a model that we want to evaluate.
        :param prompt_template: A template which can be used to compose prompt using model_input
        :return: list of EvalScore object
        """
        util.require(model_input, "Missing required input: model_input, for GeneralSemanticRobustness evaluate_sample")
        util.require(
            model, "Missing required input: model i.e. ModelRunner, for GeneralSemanticRobustness " "evaluate_sample"
        )

        prompt_composer = PromptComposer(prompt_template)
        original_prompt = prompt_composer.compose(model_input)
        original_model_output = model_output if model_output else model.predict(original_prompt)[0]

        is_model_deterministic = self._is_model_deterministic
        if is_model_deterministic is None:
            is_model_deterministic = model.predict(original_prompt)[0] == original_model_output

        perturbation = PERTURBATION_TYPE_TO_HELPER_CLASS[self._eval_algorithm_config.perturbation_type]()
        perturbed_inputs = perturbation.perturb(
            text=model_input,
            config=self._perturbation_config,
            num_perturbations=self._eval_algorithm_config.num_perturbations,
        )
        perturbed_input_prompts = [prompt_composer.compose(perturbed_input) for perturbed_input in perturbed_inputs]
        perturbed_input_outputs = [model.predict(prompt)[0] for prompt in perturbed_input_prompts]

        bert_score_value = np.mean(
            list(
                map(
                    functools.partial(get_bert_score, helper_model=self._bertscore_helper_model),
                    itertools.repeat(original_model_output, len(perturbed_input_outputs)),
                    perturbed_input_outputs,
                )
            )
        )
        eval_scores = [EvalScore(name=BERT_SCORE, value=bert_score_value)]

        if is_model_deterministic:
            wer = hf_evaluate.load("wer")
            wer_value = wer.compute(
                predictions=perturbed_input_outputs,
                references=list(itertools.repeat(original_model_output, self._eval_algorithm_config.num_perturbations)),
            )
            eval_scores.append(EvalScore(name=WER_SCORE, value=wer_value))

        return eval_scores

    def evaluate(
        self,
        model: ModelRunner,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        save: bool = False,
        num_records=100,
    ) -> List[EvalOutput]:
        """
        Semantic Robustness evaluate.

        :param model: An instance of ModelRunner which is the model under evaluation
        :param dataset_config: Configures the single dataset used for evaluation. If not provided,
            evaluation will use all of it's supported built-in datasets
        :param prompt_template: A template which can be used to generate prompts, optional, if not provided defaults
            will be used.
        :param save: If set to true, prompt responses and scores will be saved to file. The output is written to
                     EvalAlgorithmInterface.EVAL_RESULTS_PATH
        :param num_records: The number of records to be sampled randomly from the input dataset to perform the
                            evaluation
        :return: List of EvalOutput objects.
        """
        util.require(
            model, "Missing required input: model i.e. ModelRunner, for GeneralSemanticRobustness evaluate method"
        )
        if dataset_config:
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[self.eval_name]]

        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [DatasetColumns.MODEL_INPUT.value.name])
            dataset_prompt_template = (
                get_default_prompt_template(dataset_config.dataset_name) if not prompt_template else prompt_template
            )
            dataset = generate_prompt_column_for_dataset(
                dataset_prompt_template,
                dataset,
                DatasetColumns.MODEL_INPUT.value.name,
                DatasetColumns.PROMPT.value.name,
            )

            self._is_model_deterministic = verify_model_determinism(model, dataset, DatasetColumns.PROMPT.value.name)
            dataset = generate_model_predict_response_for_dataset(
                model=model,
                data=dataset,
                model_input_column_name=DatasetColumns.PROMPT.value.name,
                model_output_column_name=DatasetColumns.MODEL_OUTPUT.value.name,
            )
            with (timed_block(f"Computing score and aggregation on dataset {dataset_config.dataset_name}", logger)):

                def _generate_general_semantic_robustness_score(
                    row: Dict[str, Any]
                ) -> Dict[str, Any]:  # pragma: no cover
                    """
                    Map function generating the scores for every input record in input dataset
                    """
                    scores = self.evaluate_sample(
                        model_input=row[DatasetColumns.MODEL_INPUT.value.name],
                        model=model,
                        model_output=row[DatasetColumns.MODEL_OUTPUT.value.name],
                        prompt_template=dataset_prompt_template,
                    )
                    row[BERT_SCORE] = scores[0].value
                    if self._is_model_deterministic:
                        row[WER_SCORE] = scores[1].value

                    return row

                dataset = dataset.map(_generate_general_semantic_robustness_score).materialize()
                dataset_scores, category_scores = aggregate_evaluation_scores(
                    dataset, [BERT_SCORE, WER_SCORE] if self._is_model_deterministic else [BERT_SCORE], agg_method=MEAN
                )
                eval_outputs.append(
                    EvalOutput(
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                        prompt_template=dataset_prompt_template,
                        dataset_scores=dataset_scores,
                        category_scores=category_scores,
                        output_path=generate_output_dataset_path(
                            path_to_parent_dir=self._eval_results_path,
                            eval_name=self.eval_name,
                            dataset_name=dataset_config.dataset_name,
                        ),
                    )
                )
            self._is_model_deterministic = None
            if save:
                save_dataset(
                    dataset=dataset,
                    score_names=[WER_SCORE],
                    path=generate_output_dataset_path(
                        path_to_parent_dir=self._eval_results_path,
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                    ),
                )

        return eval_outputs
