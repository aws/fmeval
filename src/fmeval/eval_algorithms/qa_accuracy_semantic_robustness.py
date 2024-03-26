import logging
from collections import defaultdict


from typing import Any, Dict, List, Optional

from dataclasses import dataclass


import fmeval.util as util
from fmeval.constants import (
    DatasetColumns,
    MEAN,
    BUTTER_FINGER,
    RANDOM_UPPER_CASE,
    WHITESPACE_ADD_REMOVE,
    PREFIX_FOR_DELTA_SCORES,
)
from fmeval.data_loaders.util import get_dataset
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.semantic_perturbation_utils import (
    ButterFinger,
    RandomUpperCase,
    WhitespaceAddRemove,
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
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmConfig, EvalAlgorithmInterface
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
from fmeval.eval_algorithms.qa_accuracy import (
    F1_SCORE,
    EXACT_MATCH_SCORE,
    QUASI_EXACT_MATCH_SCORE,
    PRECISION_OVER_WORDS,
    RECALL_OVER_WORDS,
    QAAccuracy,
    QAAccuracyConfig,
)

# All the perturbation types supported by this eval algo
PERTURBATION_TYPE_TO_HELPER_CLASS = {
    BUTTER_FINGER: ButterFinger,
    RANDOM_UPPER_CASE: RandomUpperCase,
    WHITESPACE_ADD_REMOVE: WhitespaceAddRemove,
}

DELTA_F1_SCORE = PREFIX_FOR_DELTA_SCORES + F1_SCORE
DELTA_EXACT_MATCH_SCORE = PREFIX_FOR_DELTA_SCORES + EXACT_MATCH_SCORE
DELTA_QUASI_EXACT_MATCH_SCORE = PREFIX_FOR_DELTA_SCORES + QUASI_EXACT_MATCH_SCORE
DELTA_PRECISION_OVER_WORDS = PREFIX_FOR_DELTA_SCORES + PRECISION_OVER_WORDS
DELTA_RECALL_OVER_WORDS = PREFIX_FOR_DELTA_SCORES + RECALL_OVER_WORDS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QAAccuracySemanticRobustnessConfig(EvalAlgorithmConfig):
    """
    Configuration for the QA Accuracy Semantic Robustness Evaluation

    :param target_output_delimiter: Target Output can have multiple answers. We expect customer to combine all the
        possible answers into a single string and use the delimiter to separate them. For instance,
        if the answers are ["UK", "England"] and the delimiter="<OR>", then the target_output should be "UK<OR>England".
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

    target_output_delimiter: Optional[str] = "<OR>"
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
        if self.target_output_delimiter == "":
            raise EvalAlgorithmClientError(
                "Empty target_output_delimiter is provided. Please either provide a non-empty string, or set it to None."
            )


QA_ACCURACY_SEMANTIC_ROBUSTNESS = EvalAlgorithm.QA_ACCURACY_SEMANTIC_ROBUSTNESS.value


class QAAccuracySemanticRobustness(EvalAlgorithmInterface):
    """Semantic Robustness evaluation algorithm for QA Accuracy

    This evaluation measures how much QA Accuracy changes as a result of semantic preserving
    perturbations on the input. For example, if we apply the whitespace perturbation (adding extra whitepaces at random) to the input text,
    how much does the quality of the model answer change.

    The output difference is measured by computing the QA Accuracy metrics before after perturbing the inputs. We report the absolute value of the difference in scores
    on average over N (`num_perturbations`) perturbed inputs: $$ \frac{1}{P} \sum_{i=1}^{P} |s - \bar{s}_i|,$$
    where $s$ is the score produced by the original metric (i.e., exact match, quasi-exact match, precision over words, recall over words and F1 over words), and $\bar{s_i}$ is the metric evaluated after the i-th perturbation has been applied.

    For details on the QA Accuracy metrics, see the QA Accuracy evaluation. For details on perturbations, see the GeneralSemanticRobustness evaluation.
    """

    def __init__(
        self, eval_algorithm_config: QAAccuracySemanticRobustnessConfig = QAAccuracySemanticRobustnessConfig()
    ):
        """Default constructor

        :param eval_algorithm_config: QA Accuracy Semantic Robustness eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.eval_name = QA_ACCURACY_SEMANTIC_ROBUSTNESS
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

        self._qa_accuracy_eval_algo = QAAccuracy(
            eval_algorithm_config=QAAccuracyConfig(
                target_output_delimiter=eval_algorithm_config.target_output_delimiter
            )
        )

    def __reduce__(self):  # pragma: no cover
        """
        Custom serializer method used by Ray when it serializes instances of this
        class during dataset.map() operations.
        """
        serialized_data = (self._eval_algorithm_config,)
        return QAAccuracySemanticRobustness, serialized_data

    def evaluate(
        self,
        model: ModelRunner,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        save: bool = False,
        num_records=100,
    ) -> List[EvalOutput]:
        """
        QA Accuracy Semantic Robustness evaluate.

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
        util.require(model, "Missing required input: model i.e. ModelRunner, for QAAccuracySemanticRobustness evaluate")
        if dataset_config:
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[self.eval_name]]

        eval_outputs: List[EvalOutput] = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [DatasetColumns.MODEL_INPUT.value.name, DatasetColumns.TARGET_OUTPUT.value.name])
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
                    dataset,
                    [
                        F1_SCORE,
                        EXACT_MATCH_SCORE,
                        QUASI_EXACT_MATCH_SCORE,
                        PRECISION_OVER_WORDS,
                        RECALL_OVER_WORDS,
                        DELTA_F1_SCORE,
                        DELTA_EXACT_MATCH_SCORE,
                        DELTA_QUASI_EXACT_MATCH_SCORE,
                        DELTA_PRECISION_OVER_WORDS,
                        DELTA_RECALL_OVER_WORDS,
                    ],
                    agg_method=MEAN,
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
            if save:
                save_dataset(
                    dataset=dataset,
                    score_names=[
                        F1_SCORE,
                        EXACT_MATCH_SCORE,
                        QUASI_EXACT_MATCH_SCORE,
                        PRECISION_OVER_WORDS,
                        RECALL_OVER_WORDS,
                        DELTA_F1_SCORE,
                        DELTA_EXACT_MATCH_SCORE,
                        DELTA_QUASI_EXACT_MATCH_SCORE,
                        DELTA_PRECISION_OVER_WORDS,
                        DELTA_RECALL_OVER_WORDS,
                    ],
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
        Evaluate a single QA record for Semantic Robustness.

        :param model_input: text input for model
        :param model: An instance of ModelRunner which is the model under evaluation
        :param target_output: The expected responses from the model
        :param model_output: The output of a model that we want to evaluate.
        :param prompt_template: A template which can be used to compose prompt using model_input
        :returns: A List of EvalScores computed for prompts and responses.
        """
        util.require(
            model_input, "Missing required input: model_input, for QAAccuracySemanticRobustness evaluate_sample"
        )
        util.require(
            model, "Missing required input: model i.e. ModelRunner, for QAAccuracySemanticRobustness evaluate_sample"
        )
        util.require(
            target_output, "Missing required input: target_output, for " "QAAccuracySemanticRobustness evaluate_sample"
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

        original_qa_accuracy_scores = self._qa_accuracy_eval_algo.evaluate_sample(
            target_output=target_output, model_output=original_model_output
        )

        perturbed_outputs_qa_accuracy_scores = defaultdict(list)
        for perturbed_input_output in perturbed_input_outputs:
            accuracy_scores = self._qa_accuracy_eval_algo.evaluate_sample(
                target_output=target_output, model_output=perturbed_input_output
            )
            for accuracy_score in accuracy_scores:
                perturbed_outputs_qa_accuracy_scores[accuracy_score.name].append(accuracy_score)

        delta_scores = [
            EvalScore(
                name=PREFIX_FOR_DELTA_SCORES + original_score.name,
                value=generate_mean_delta_score(
                    original_score, perturbed_outputs_qa_accuracy_scores[original_score.name]
                ),
            )
            for original_score in original_qa_accuracy_scores
        ]
        return original_qa_accuracy_scores + delta_scores
