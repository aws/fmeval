import logging
from collections import defaultdict

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import ray.data
from ray.data import Dataset

from amazon_fmeval import util
from amazon_fmeval.constants import (
    MODEL_INPUT_COLUMN_NAME,
    TARGET_OUTPUT_COLUMN_NAME,
    MEAN,
    BUTTER_FINGER,
    RANDOM_UPPER_CASE,
    WHITESPACE_ADD_REMOVE,
    PREFIX_FOR_DELTA_SCORES,
)
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.data_loaders.util import get_dataset
from amazon_fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalScore,
    EvalOutput,
    DATASET_CONFIGS,
    EVAL_DATASETS,
    EVAL_PROMPT_TEMPLATES,
)
from amazon_fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmConfig, EvalAlgorithmInterface
from amazon_fmeval.eval_algorithms.semantic_perturbation_utils import (
    ButterFinger,
    RandomUpperCase,
    WhitespaceAddRemove,
    ButterFingerConfig,
    RandomUpperCaseConfig,
    WhitespaceAddRemoveConfig,
)
from amazon_fmeval.eval_algorithms.summarization_accuracy import (
    ROUGE_2,
    DEFAULT_MODEL_TYPE,
    SummarizationAccuracyConfig,
    ROUGE_TYPES,
    MODEL_TYPES_SUPPORTED,
    SummarizationAccuracy,
    ROUGE_SCORE,
    METEOR_SCORE,
    BERT_SCORE,
)
from amazon_fmeval.eval_algorithms.util import (
    validate_dataset,
    save_dataset,
    aggregate_evaluation_scores,
    generate_output_dataset_path,
    generate_prompt_column_for_dataset,
    get_num_actors,
    generate_mean_delta_score,
)
from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.composers.composers import PromptComposer
from amazon_fmeval.model_runners.model_runner import ModelRunner
from amazon_fmeval.perf_util import timed_block

logger = logging.getLogger(__name__)

# All the perturbation types supported by this eval algo
PERTURBATION_TYPE_TO_HELPER_CLASS = {
    BUTTER_FINGER: ButterFinger,
    RANDOM_UPPER_CASE: RandomUpperCase,
    WHITESPACE_ADD_REMOVE: WhitespaceAddRemove,
}

PROMPT_COLUMN_NAME = "prompt"

PREFIX_FOR_DELTA_SCORES = "delta_"
DELTA_ROUGE_SCORE = PREFIX_FOR_DELTA_SCORES + ROUGE_SCORE
DELTA_METEOR_SCORE = PREFIX_FOR_DELTA_SCORES + METEOR_SCORE
DELTA_BERT_SCORE = PREFIX_FOR_DELTA_SCORES + BERT_SCORE


@dataclass(frozen=True)
class SummarizationAccuracySemanticRobustnessConfig(EvalAlgorithmConfig):
    """
    Configuration for the summarization accuracy semantic robustness eval algorithm.

    :param perturbation_type: perturbation type for generating perturbed inputs
    :param num_perturbations: Number of perturbed inputs to be generated for robustness evaluation
    :param seed: Seed to be configured for generating perturbations
    :param butter_finger_perturbation_prob: The probability that a given character will be perturbed. Used for
        butter_finger perturbation_type
    :param random_uppercase_corrupt_proportion: Fraction of characters to be changed to uppercase. Used for
        random_upper_case perturbation_type
    :param whitespace_remove_prob: Given a whitespace, remove it with this much probability. Used for
        whitespace_add_remove perturbation_type
    :param whitespace_add_prob: Given a non-whitespace, add a whitespace before it with this probability. Used for
        whitespace_add_remove perturbation_type
    :param rouge_type: Type of rouge metric in eval results
    :param use_stemmer_for_rouge: bool value to set using stemmer for rouge metric
    :param model_type_for_bertscore: model to use for bert score
    """

    perturbation_type: str = BUTTER_FINGER
    num_perturbations: int = 5
    seed: int = 5
    butter_finger_perturbation_prob: Optional[float] = 0.1
    random_uppercase_corrupt_proportion: Optional[float] = 0.1
    whitespace_remove_prob: Optional[float] = 0.1
    whitespace_add_prob: Optional[float] = 0.05
    rouge_type: Optional[str] = ROUGE_2
    use_stemmer_for_rouge: bool = True
    model_type_for_bertscore: Optional[str] = DEFAULT_MODEL_TYPE

    def __post_init__(self):
        if self.perturbation_type not in PERTURBATION_TYPE_TO_HELPER_CLASS.keys():
            raise EvalAlgorithmClientError(
                f"Invalid perturbation type '{self.perturbation_type} requested, please "
                f"choose from acceptable values: {PERTURBATION_TYPE_TO_HELPER_CLASS.keys()}"
            )

        if not self.rouge_type in ROUGE_TYPES:
            raise EvalAlgorithmClientError(
                f"Invalid rouge_type: {self.rouge_type} requested in SummarizationAccuracyConfig, "
                f"please choose from acceptable values: {ROUGE_TYPES}"
            )

        if not self.model_type_for_bertscore in MODEL_TYPES_SUPPORTED:
            raise EvalAlgorithmClientError(
                f"Invalid model_type_for_bertscore: {self.model_type_for_bertscore} requested in "
                f"SummarizationAccuracyConfig, please choose from acceptable values: {MODEL_TYPES_SUPPORTED}"
            )


class SummarizationAccuracySemanticRobustness(EvalAlgorithmInterface):
    """
    Semantic Robustness Eval algorithm for Summarization Accuracy task LLMs

    This evaluation measures how much the model output changes as a result of semantic preserving
    perturbations. Given the input, e.g., "A quick brown fox jumps over the lazy dog", the
    evaluation creates a perturbation that preserves the semantic meaning of the input e.g.,
    whitespace perturbation that changes the input text to "A q uick bro wn fox ju mps overthe lazy
    dog". The evaluation then measures how much the model output changes when prompted with the
    original vs. perturbed input. The algo compares summarization accuracy of model output for original model output
    and model output for perturbed inputs, returns delta between rouge, meteor and bert
    scores.
    """

    def __init__(self, eval_algorithm_config: SummarizationAccuracySemanticRobustnessConfig):
        """Default constructor

        :param eval_algorithm_config: Summarization Accuracy Semantic Robustness eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.eval_name = EvalAlgorithm.SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS.value
        self._eval_algorithm_config = eval_algorithm_config

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

        self._summarization_accuracy_eval_algo = SummarizationAccuracy(
            SummarizationAccuracyConfig(
                rouge_type=eval_algorithm_config.rouge_type,
                use_stemmer_for_rouge=eval_algorithm_config.use_stemmer_for_rouge,
                model_type_for_bertscore=eval_algorithm_config.model_type_for_bertscore,
            )
        )

    def __reduce__(self):  # pragma: no cover
        """
        Custom serializer method used by Ray when it serializes instances of this
        class during dataset.map() operations.
        """
        serialized_data = (self._eval_algorithm_config,)
        return SummarizationAccuracySemanticRobustness, serialized_data

    def evaluate_sample(
        self, model_input: str, target_output: str, model: ModelRunner, prompt_template: str = "$feature"
    ) -> List[EvalScore]:  # type: ignore[override]
        """
        Summarization Accuracy Semantic Robustness evaluate sample.

        :param model_input: text input for model
        :param target_output: The expected responses from the model
        :param model: An instance of ModelRunner which is the model under evaluation
        :param prompt_template: A template which can be used to compose prompt using model_input
        :return: list of EvalScore object
        """
        util.require(
            model_input,
            "Missing required input: model_input, for SummarizationAccuracySemanticRobustness evaluate_sample",
        )
        util.require(
            model,
            "Missing required input: model i.e. ModelRunner, for "
            "SummarizationAccuracySemanticRobustness evaluate_sample",
        )
        util.require(
            target_output,
            "Missing required input: target_output, for " "SummarizationAccuracySemanticRobustness evaluate_sample",
        )

        prompt_composer = PromptComposer(prompt_template)
        original_prompt = prompt_composer.compose(model_input)
        original_model_output = model.predict(original_prompt)[0]

        # Check if predictor is deterministic
        if model.predict(original_prompt)[0] != original_model_output:
            raise EvalAlgorithmClientError("For evaluating semantic robustness, the model should be deterministic.")

        perturbation = PERTURBATION_TYPE_TO_HELPER_CLASS[self._eval_algorithm_config.perturbation_type](
            seed=self._eval_algorithm_config.seed
        )
        perturbed_inputs = perturbation.perturb(
            text=model_input,
            config=self._perturbation_config,
            num_perturbations=self._eval_algorithm_config.num_perturbations,
        )
        perturbed_input_prompts = [prompt_composer.compose(perturbed_input) for perturbed_input in perturbed_inputs]
        perturbed_input_outputs = [model.predict(prompt)[0] for prompt in perturbed_input_prompts]

        original_summarization_accuracy_scores = self._summarization_accuracy_eval_algo.evaluate_sample(
            target_output=target_output, model_output=original_model_output
        )

        perturbed_outputs_summarization_accuracy_scores = defaultdict(list)
        for perturbed_input_output in perturbed_input_outputs:
            accuracy_scores = self._summarization_accuracy_eval_algo.evaluate_sample(
                target_output=target_output, model_output=perturbed_input_output
            )
            for accuracy_score in accuracy_scores:
                perturbed_outputs_summarization_accuracy_scores[accuracy_score.name].append(accuracy_score)

        return [
            EvalScore(
                name=PREFIX_FOR_DELTA_SCORES + original_score.name,
                value=generate_mean_delta_score(
                    original_score, perturbed_outputs_summarization_accuracy_scores[original_score.name]
                ),
            )
            for original_score in original_summarization_accuracy_scores
        ]

    def evaluate(  # type: ignore[override]
        self,
        model: ModelRunner,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        save: bool = False,
        num_records: int = 100,
    ) -> List[EvalOutput]:
        """
        Semantic Robustness evaluate.

        :param model: An instance of ModelRunner which is the model under evaluation
        :param dataset_config: Configures the single dataset used for evaluation. If not provided,
            evaluation will use all of it's supported built-in datasets
        :param prompt_template: A template which can be used to generate prompts, optional for the built-in datasets.
        :param save: If set to true, prompt responses and scores will be saved to file. The output is written to
                     EvalAlgorithmInterface.EVAL_RESULTS_PATH
        :return: List of EvalOutput objects.
        """
        util.require(
            model,
            "Missing required input: model i.e. ModelRunner, for SummarizationAccuracySemanticRobustness "
            "evaluate method",
        )
        is_custom_dataset_evaluation = False
        if dataset_config:
            is_custom_dataset_evaluation = True
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[self.eval_name]]

        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [MODEL_INPUT_COLUMN_NAME, TARGET_OUTPUT_COLUMN_NAME])
            if is_custom_dataset_evaluation:
                # TODO when user provide built-in DataConfig, we should provide default prompt_template
                util.require(
                    prompt_template,
                    f"Missing required input: prompt_template for evaluating custom dataset : {dataset_config}",
                )
            else:
                prompt_template = EVAL_PROMPT_TEMPLATES[self.eval_name, dataset_config.dataset_name]
                util.assert_condition(
                    prompt_template is not None,
                    f"No Prompt Template configured for ({self.eval_name}, {dataset_config.dataset_name})",
                )

            assert prompt_template  # to satisfy mypy
            dataset = generate_prompt_column_for_dataset(
                prompt_template, dataset, MODEL_INPUT_COLUMN_NAME, PROMPT_COLUMN_NAME
            )
            with timed_block(f"Computing score and aggregation on dataset {dataset_config.dataset_name}", logger):
                dataset = self.__add_scores(model, prompt_template, dataset)

                dataset_scores, category_scores = aggregate_evaluation_scores(
                    dataset, [DELTA_ROUGE_SCORE, DELTA_BERT_SCORE, DELTA_METEOR_SCORE], agg_method=MEAN
                )
                eval_outputs.append(
                    EvalOutput(
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                        prompt_template=prompt_template,
                        dataset_scores=dataset_scores,
                        category_scores=category_scores,
                        output_path=self._eval_results_path,
                    )
                )

            if save:
                save_dataset(
                    dataset=dataset,
                    score_names=[DELTA_ROUGE_SCORE, DELTA_BERT_SCORE, DELTA_METEOR_SCORE],
                    path=generate_output_dataset_path(
                        path_to_parent_dir=self._eval_results_path,
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                    ),
                )

        return eval_outputs

    def __add_scores(self, model: ModelRunner, prompt_template: str, dataset: Dataset) -> Dataset:  # pragma: no cover
        """
        Private method to encapsulate map call on evaluate sample. Specifically created for cleaner mocking in
        unit tests.
        :param model: model to be used for evaluation
        :param prompt_template: prompt template
        :param dataset: input ray dataset
        :returns: ray dataset with added score columns
        """
        evaluate_sample_fn = self.evaluate_sample

        class GenerateEvalScoresActor:  # pragma: no cover
            """
            This class represents the Ray Actor that gets eval scores for every row in ray dataset by
            calling evaluate_sample of the same class.

            We use Ray Actors instead of Tasks because the Actor approach minimizes
            the number of times the SummarizationAccuracy dependent class gets serialised/deserialized.
            With Tasks, Ray will serialise and deserialize for every single row evaluation. With Actors,
            class gets deserialized once per Actor when the Actor gets initialized.
            """

            def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
                assert prompt_template  # to satisfy mypy
                scores = evaluate_sample_fn(
                    row[MODEL_INPUT_COLUMN_NAME], row[TARGET_OUTPUT_COLUMN_NAME], model, prompt_template
                )
                for score in scores:
                    row[score.name] = score.value
                return row

        return dataset.map(
            GenerateEvalScoresActor, compute=ray.data.ActorPoolStrategy(size=get_num_actors())  # type: ignore[arg-type]
        ).materialize()
