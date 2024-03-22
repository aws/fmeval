import logging
import warnings
from collections import defaultdict

from typing import Any, Callable, List, Optional, Dict

from dataclasses import dataclass

import fmeval.util as util
from fmeval.constants import (
    DatasetColumns,
    MEAN,
    BUTTER_FINGER,
    RANDOM_UPPERCASE,
    ADD_REMOVE_WHITESPACE,
    PREFIX_FOR_DELTA_SCORES,
)
from fmeval.data_loaders.util import get_dataset
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.semantic_perturbation_utils import (
    ButterFingerConfig,
    RandomUpperCaseConfig,
    WhitespaceAddRemoveConfig,
)
from fmeval.eval_algorithms.util import (
    generate_prompt_column_for_dataset,
    aggregate_evaluation_scores,
    validate_dataset,
    save_dataset,
    generate_output_dataset_path,
    generate_mean_delta_score,
    generate_model_predict_response_for_dataset,
)
from fmeval.eval_algorithms.eval_algorithm import (
    EvalAlgorithmInterface,
    EvalAlgorithmConfig,
)
from fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalOutput,
    EvalScore,
    EVAL_DATASETS,
    DATASET_CONFIGS,
    get_default_prompt_template,
    DEFAULT_PROMPT_TEMPLATE,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.perf_util import timed_block
from fmeval.eval_algorithms.classification_accuracy import (
    convert_model_output_to_label,
    ClassificationAccuracy,
    ClassificationAccuracyConfig,
    CLASSIFICATION_ACCURACY_SCORE,
    UNIQUENESS_FACTOR,
)
from fmeval.eval_algorithms.semantic_perturbation_utils import ButterFinger, RandomUpperCase, WhitespaceAddRemove

# All the perturbation types supported by this eval algo
PERTURBATION_TYPE_TO_HELPER_CLASS = {
    BUTTER_FINGER: ButterFinger,
    RANDOM_UPPERCASE: RandomUpperCase,
    ADD_REMOVE_WHITESPACE: WhitespaceAddRemove,
}

PREFIX_FOR_DELTA_SCORES = "delta_"
DELTA_CLASSIFICATION_ACCURACY_SCORE = PREFIX_FOR_DELTA_SCORES + CLASSIFICATION_ACCURACY_SCORE

logger = logging.getLogger(__name__)


@dataclass
class ClassificationAccuracySemanticRobustnessConfig(EvalAlgorithmConfig):
    """
    Configuration for the Classification Accuracy Semantic Robustness Evaluation

    :param valid_labels: List of valid string label
    :param converter_fn: Function to process model output to labels, defaults to simple integer conversion
    :param perturbation_type: perturbation type for generating perturbed inputs
    :param num_perturbations: Number of perturbed inputs to be generated for robustness evaluation
    :param butter_finger_perturbation_prob: The probability that a given character will be perturbed. Used for
        butter_finger perturbation_type
    :param random_uppercase_corrupt_proportion: Fraction of characters to be changed to uppercase. Used for
        random_uppercase perturbation_type
    :param whitespace_remove_prob: Given a whitespace, remove it with this much probability. Used for
        add_remove_whitespace perturbation_type
    :param whitespace_add_prob: Given a non-whitespace, add a whitespace before it with this probability. Used for
        add_remove_whitespace perturbation_type
    """

    valid_labels: Optional[List[str]] = None
    converter_fn: Callable[[str, List[str]], str] = convert_model_output_to_label
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
        if self.valid_labels:
            for i, label in enumerate(self.valid_labels):
                if not isinstance(label, str):
                    warnings.warn("Valid labels should be strings, casting.")
                    self.valid_labels[i] = str(label)


CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS = EvalAlgorithm.CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS.value


class ClassificationAccuracySemanticRobustness(EvalAlgorithmInterface):
    """
    Classification Accuracy Eval algorithm
    """

    eval_name = EvalAlgorithm.CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS.value

    def __init__(
        self,
        eval_algorithm_config: ClassificationAccuracySemanticRobustnessConfig = ClassificationAccuracySemanticRobustnessConfig(),
    ):
        """Default constructor

        :param eval_algorithm_config: Classification Accuracy Semantic Robustness eval algorithm config.
        """
        self.eval_name = CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS
        self._eval_algorithm_config = eval_algorithm_config
        self._classification_accuracy_eval_algo = ClassificationAccuracy(
            eval_algorithm_config=ClassificationAccuracyConfig(
                valid_labels=self._eval_algorithm_config.valid_labels,
                converter_fn=self._eval_algorithm_config.converter_fn,
            )
        )

        if self._eval_algorithm_config.perturbation_type == BUTTER_FINGER:
            self._perturbation_config = ButterFingerConfig(self._eval_algorithm_config.butter_finger_perturbation_prob)
        elif self._eval_algorithm_config.perturbation_type == RANDOM_UPPERCASE:
            self._perturbation_config = RandomUpperCaseConfig(
                self._eval_algorithm_config.random_uppercase_corrupt_proportion
            )
        else:
            self._perturbation_config = WhitespaceAddRemoveConfig(
                self._eval_algorithm_config.whitespace_remove_prob, self._eval_algorithm_config.whitespace_add_prob
            )

        self._classification_accuracy_eval_algo = ClassificationAccuracy(
            eval_algorithm_config=ClassificationAccuracyConfig(
                valid_labels=eval_algorithm_config.valid_labels, converter_fn=eval_algorithm_config.converter_fn
            )
        )

    def __reduce__(self):  # pragma: no cover
        """
        Custom serializer method used by Ray when it serializes instances of this
        class during dataset.map() operations.
        """
        serialized_data = (self._eval_algorithm_config,)
        return ClassificationAccuracySemanticRobustness, serialized_data

    def evaluate(
        self,
        model: ModelRunner,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        save: bool = False,
        num_records=100,
    ) -> List[EvalOutput]:
        """
        Classification Accuracy Semantic Robustness evaluate.

        :param model: An instance of ModelRunner which is the model under evaluation
        :param dataset_config: Configures the single dataset used for evaluation. If not provided,
            evaluation will use all of it's supported built-in datasets
        :param prompt_template: A template which can be used to generate prompts, optional, if not provided defaults
            will be used.
        :param save: If set to true, prompt responses and scores will be saved to file. The output is written to
                     EvalAlgorithmInterface.EVAL_RESULTS_PATH
        :param num_records: The number of records to be sampled randomly from the input dataset to perform the
                            evaluation
        :returns: A List of EvalOutput objects.
        """
        util.require(
            model,
            "Missing required input: model i.e. ModelRunner, for ClassificationAccuracySemanticRobustness evaluate",
        )
        if dataset_config:
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[self.eval_name]]

        eval_outputs: List[EvalOutput] = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [DatasetColumns.TARGET_OUTPUT.value.name, DatasetColumns.MODEL_INPUT.value.name])
            dataset_prompt_template = (
                get_default_prompt_template(dataset_config.dataset_name) if not prompt_template else prompt_template
            )
            dataset = generate_prompt_column_for_dataset(
                prompt_template=dataset_prompt_template,
                data=dataset,
                model_input_column_name=DatasetColumns.MODEL_INPUT.value.name,
                prompt_column_name=DatasetColumns.PROMPT.value.name,
            )

            dataset = generate_model_predict_response_for_dataset(
                model=model,
                data=dataset,
                model_input_column_name=DatasetColumns.PROMPT.value.name,
                model_output_column_name=DatasetColumns.MODEL_OUTPUT.value.name,
            )

            config_valid_labels = self._eval_algorithm_config.valid_labels
            if not self._eval_algorithm_config.valid_labels:  # pragma: no branch
                self._eval_algorithm_config.valid_labels = dataset.unique(
                    column=DatasetColumns.TARGET_OUTPUT.value.name
                )
                row_count = dataset.count()
                assert self._eval_algorithm_config.valid_labels is not None  # to satisfy mypy
                if (
                    len(self._eval_algorithm_config.valid_labels) / (row_count + 1) < UNIQUENESS_FACTOR
                ):  # pragma: no cover
                    logger.warning(
                        f"The number of classes: {len(self._eval_algorithm_config.valid_labels)} in the dataset is too large "
                        f"for the number of rows in the dataset: {row_count}",
                    )
                self._classification_accuracy_eval_algo = ClassificationAccuracy(
                    eval_algorithm_config=ClassificationAccuracyConfig(
                        valid_labels=self._eval_algorithm_config.valid_labels,
                        converter_fn=self._eval_algorithm_config.converter_fn,
                    )
                )
            with timed_block(f"Computing score and aggregation on dataset {dataset_config.dataset_name}", logger):

                def _generate_score_columns(row: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
                    scores = self.evaluate_sample(
                        model_input=row[DatasetColumns.MODEL_INPUT.value.name],
                        model=model,
                        target_output=row[DatasetColumns.TARGET_OUTPUT.value.name],
                        model_output=row[DatasetColumns.MODEL_OUTPUT.value.name],
                        prompt_template=dataset_prompt_template,
                    )
                    for score in scores:
                        row[score.name] = score.value
                    return row

                dataset = dataset.map(_generate_score_columns).materialize()

                dataset_scores, category_scores = aggregate_evaluation_scores(
                    dataset, [CLASSIFICATION_ACCURACY_SCORE, DELTA_CLASSIFICATION_ACCURACY_SCORE], agg_method=MEAN
                )

                eval_outputs.append(
                    EvalOutput(
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                        prompt_template=dataset_prompt_template,
                        dataset_scores=dataset_scores,
                        category_scores=category_scores,
                        output_path=generate_output_dataset_path(
                            path_to_parent_dir=util.get_eval_results_path(),
                            eval_name=self.eval_name,
                            dataset_name=dataset_config.dataset_name,
                        ),
                    )
                )
            # set it back to the same value as before the start of evaluating this dataset
            self._eval_algorithm_config.valid_labels = config_valid_labels
            if save:
                save_dataset(
                    dataset=dataset,
                    score_names=[CLASSIFICATION_ACCURACY_SCORE, DELTA_CLASSIFICATION_ACCURACY_SCORE],
                    path=generate_output_dataset_path(
                        path_to_parent_dir=util.get_eval_results_path(),
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                    ),
                )

        return eval_outputs

    def evaluate_sample(
        self,
        model_input: str,
        model: ModelRunner,
        target_output: str,
        model_output: Optional[str] = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> List[EvalScore]:  # type: ignore[override]
        """
        Evaluate a single record for Classification Accuracy Semantic Robustness.

        :param model_input: text input for model
        :param model: An instance of ModelRunner which is the model under evaluation
        :param target_output: The expected responses from the model
        :param model_output: The output of a model that we want to evaluate
        :param prompt_template: A template which can be used to compose prompt using model_input
        :returns: A List of EvalScores computed for prompts and responses.
        """
        util.require(
            model_input,
            "Missing required input: model_input, for ClassificationAccuracySemanticRobustness evaluate_sample",
        )
        util.require(
            model,
            "Missing required input: model i.e. ModelRunner, for ClassificationAccuracySemanticRobustness evaluate_sample",
        )
        util.require(
            target_output,
            "Missing required input: target_output, for " "ClassificationAccuracySemanticRobustness evaluate_sample",
        )

        prompt_composer = PromptComposer(prompt_template)
        original_prompt = prompt_composer.compose(model_input)
        original_model_output = model_output if model_output else model.predict(original_prompt)[0]

        perturbation = PERTURBATION_TYPE_TO_HELPER_CLASS[self._eval_algorithm_config.perturbation_type]()

        perturbed_inputs = perturbation.perturb(
            text=model_input,
            config=self._perturbation_config,
            num_perturbations=self._eval_algorithm_config.num_perturbations,
        )

        perturbed_input_prompts = [prompt_composer.compose(perturbed_input) for perturbed_input in perturbed_inputs]
        perturbed_input_outputs = [model.predict(prompt)[0] for prompt in perturbed_input_prompts]

        original_classification_accuracy_scores = self._classification_accuracy_eval_algo.evaluate_sample(
            target_output=target_output, model_output=original_model_output
        )

        perturbed_outputs_classification_accuracy_scores = defaultdict(lambda: [])
        for perturbed_input_output in perturbed_input_outputs:
            accuracy_scores = self._classification_accuracy_eval_algo.evaluate_sample(
                target_output=target_output, model_output=perturbed_input_output
            )
            for accuracy_score in accuracy_scores:
                perturbed_outputs_classification_accuracy_scores[accuracy_score.name].append(accuracy_score)

        delta_scores = [
            EvalScore(
                name=PREFIX_FOR_DELTA_SCORES + original_score.name,
                value=generate_mean_delta_score(
                    original_score, perturbed_outputs_classification_accuracy_scores[original_score.name]
                ),
            )
            for original_score in original_classification_accuracy_scores
        ]

        return original_classification_accuracy_scores + delta_scores
