import itertools
import logging

import evaluate as hf_evaluate
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


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

logger = logging.getLogger(__name__)

# All the perturbation types supported by this eval algo
PERTURBATION_TYPE_TO_HELPER_CLASS = {
    BUTTER_FINGER: ButterFinger,
    RANDOM_UPPER_CASE: RandomUpperCase,
    WHITESPACE_ADD_REMOVE: WhitespaceAddRemove,
}

WER_SCORE = "word_error_rate"


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
    """

    perturbation_type: str = BUTTER_FINGER
    num_perturbations: int = 5
    butter_finger_perturbation_prob: float = 0.1
    random_uppercase_corrupt_proportion: float = 0.1
    whitespace_remove_prob: float = 0.1
    whitespace_add_prob: float = 0.05

    def __post_init__(self):
        if self.perturbation_type not in PERTURBATION_TYPE_TO_HELPER_CLASS.keys():
            raise EvalAlgorithmClientError(
                f"Invalid perturbation type '{self.perturbation_type} requested, please "
                f"choose from acceptable values: {PERTURBATION_TYPE_TO_HELPER_CLASS.keys()}"
            )


class GeneralSemanticRobustness(EvalAlgorithmInterface):
    """
    Semantic Robustness Eval algorithm for General task LLMs

    This evaluation measures how much the model output changes as a result of semantic preserving
    perturbations. Given the input, e.g., "A quick brown fox jumps over the lazy dog", the
    evaluation creates a perturbation that preserves the semantic meaning of the input e.g.,
    whitespace perturbation that changes the input text to "A q uick bro wn fox ju mps overthe lazy
    dog". The evaluation then measures how much the model output changes when prompted with the
    original vs. perturbed input. The output difference is measured using Word Error Rate (WER).
    https://huggingface.co/spaces/evaluate-metric/wer
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

        if self._is_model_deterministic is None:
            if model.predict(original_prompt)[0] != original_model_output:
                raise EvalAlgorithmClientError("For evaluating semantic robustness, the model should be deterministic.")

        perturbation = PERTURBATION_TYPE_TO_HELPER_CLASS[self._eval_algorithm_config.perturbation_type]()
        perturbed_inputs = perturbation.perturb(
            text=model_input,
            config=self._perturbation_config,
            num_perturbations=self._eval_algorithm_config.num_perturbations,
        )
        perturbed_input_prompts = [prompt_composer.compose(perturbed_input) for perturbed_input in perturbed_inputs]
        perturbed_input_outputs = [model.predict(prompt)[0] for prompt in perturbed_input_prompts]

        wer = hf_evaluate.load("wer")

        return [
            EvalScore(
                name=WER_SCORE,
                value=wer.compute(
                    predictions=perturbed_input_outputs,
                    references=list(
                        itertools.repeat(original_model_output, self._eval_algorithm_config.num_perturbations)
                    ),
                ),
            )
        ]

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
            if not self._is_model_deterministic:
                raise EvalAlgorithmClientError("For evaluating semantic robustness, the model should be deterministic.")
            dataset = generate_model_predict_response_for_dataset(
                model=model,
                data=dataset,
                model_input_column_name=DatasetColumns.PROMPT.value.name,
                model_output_column_name=DatasetColumns.MODEL_OUTPUT.value.name,
            )
            with timed_block(f"Computing score and aggregation on dataset {dataset_config.dataset_name}", logger):

                def _generate_general_semantic_robustness_score(
                    row: Dict[str, Any]
                ) -> Dict[str, Any]:  # pragma: no cover
                    """
                    Map function generating the scores for every input record in input dataset
                    """
                    row[WER_SCORE] = self.evaluate_sample(
                        model_input=row[DatasetColumns.MODEL_INPUT.value.name],
                        model=model,
                        model_output=row[DatasetColumns.MODEL_OUTPUT.value.name],
                        prompt_template=dataset_prompt_template,
                    )[0].value
                    return row

                dataset = dataset.map(_generate_general_semantic_robustness_score).materialize()

                dataset_scores, category_scores = aggregate_evaluation_scores(dataset, [WER_SCORE], agg_method=MEAN)
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
