import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ray.data import Dataset
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

import fmeval.util as util
from fmeval.constants import (
    DatasetColumns,
    MEAN,
)
from fmeval.data_loaders.util import get_dataset
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.common import save_dataset
from fmeval.eval_algorithms.eval_algorithm import (
    EvalAlgorithmInterface,
    EvalAlgorithmConfig,
)
from fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalOutput,
    EvalScore,
    CategoryScore,
    get_default_prompt_template,
)
from fmeval.eval_algorithms.save_strategy import SaveStrategy, FileSaveStrategy
from fmeval.eval_algorithms.util import (
    validate_dataset,
    category_wise_aggregation,
    generate_output_dataset_path,
    get_dataset_configs,
    create_model_invocation_pipeline,
)
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.perf_util import timed_block
from fmeval.transforms.transform import Transform
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.transforms.util import validate_call

CLASSIFICATION_ACCURACY_SCORE = "classification_accuracy_score"
BALANCED_ACCURACY_SCORE = "balanced_accuracy_score"
PRECISION_SCORE = "precision_score"
RECALL_SCORE = "recall_score"
UNKNOWN_LABEL = "unknown"
CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME = "classified_model_output"
OUTPUT_KEYS = [CLASSIFICATION_ACCURACY_SCORE, CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME]
CLASSIFICATION_ACCURACY_SCORES_TO_FUNCS: Dict[str, Callable[..., float]] = {
    BALANCED_ACCURACY_SCORE: balanced_accuracy_score,
    PRECISION_SCORE: precision_score,
    RECALL_SCORE: recall_score,
}
UNIQUENESS_FACTOR = 0.05

logger = logging.getLogger(__name__)


def convert_model_output_to_label(model_output: str, valid_labels: List[str]) -> str:
    """Convert model output to string class label. The model is expected to return a label directly (if it has a
    classification head), or a string containing a label (if it has a language modelling head). In the latter case we
    strip any additional text (e.g. "The answer is 2." --> "2"). If no valid labels is contained in the
    `model_output` an "unknown" label is returned. Users can define other `converter_fn`s, e.g. to translate a text
    label to string ("NEGATIVE" --> "0").

    :param model_output: Value returned by the model.
    :param valid_labels: Valid labels.
    :return: `model_output` transformed into a label
    """
    # normalise to lowercase & strip
    valid_labels = [label.lower().strip() for label in valid_labels]

    response_words = model_output.split(" ")
    predicted_labels = [word.lower().strip() for word in response_words if word.lower().strip() in valid_labels]
    # if there is more than one label in the model output we pick the first
    string_label = predicted_labels[0] if predicted_labels else UNKNOWN_LABEL

    return string_label


class ClassificationAccuracyScores(Transform):
    """This transform augments its input record with computed classification accuracy scores."""

    def __init__(
        self,
        target_output_key: str = DatasetColumns.TARGET_OUTPUT.value.name,
        model_output_key: str = DatasetColumns.MODEL_OUTPUT.value.name,
        classified_model_output_key: str = CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME,
        classification_accuracy_score_key: str = CLASSIFICATION_ACCURACY_SCORE,
        valid_labels: Optional[List[str]] = None,
        converter_fn: Callable[[str, List[str]], str] = convert_model_output_to_label,
    ):
        """ClassificationAccuracyScores initializer.

        :param target_output_key: The record key corresponding to the target output.
        :param model_output_key: The record key corresponding to the model output.
        :param classified_model_output_key: The key to use for the classified model output
            that will be added to the record.
        :param classification_accuracy_score_key: The key to use for the classification accuracy
            score that will be added to the record.
        :param valid_labels: See corresponding parameter in ClassificationAccuracyConfig.
        :param converter_fn: See corresponding parameter in ClassificationAccuracyConfig.
        """
        super().__init__(
            target_output_key,
            model_output_key,
            classified_model_output_key,
            classification_accuracy_score_key,
            valid_labels,
            converter_fn,
        )
        self.register_input_output_keys(
            input_keys=[target_output_key, model_output_key],
            output_keys=[classified_model_output_key, classification_accuracy_score_key],
        )
        self.target_output_key = target_output_key
        self.model_output_key = model_output_key
        self.classified_model_output_key = classified_model_output_key
        self.classification_accuracy_score_key = classification_accuracy_score_key
        self.valid_labels = valid_labels
        self.converter_fn = converter_fn

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with computed classification accuracy scores.

        :param record: The input record.
        :returns: The input record with the classification accuracy score
            and the classified model output added in.
        """
        target_output = record[self.target_output_key]
        model_output = record[self.model_output_key]
        record[self.classified_model_output_key] = self.converter_fn(model_output, self.valid_labels)  # type: ignore
        record[self.classification_accuracy_score_key] = int(
            record[self.classified_model_output_key] == str(target_output)
        )
        return record


@dataclass(frozen=True)
class ClassificationAccuracyConfig(EvalAlgorithmConfig):
    """Configures the Classification Accuracy evaluation algorithm.

    :param valid_labels: The labels of the classes predicted from the model.
    :param converter_fn: Function to process model output to labels, defaults to simple integer conversion.
    :param multiclass_average_strategy: `average` to be passed to sklearn's precision and recall scores.
        This determines how scores are aggregated in the multiclass classification setting
        (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html).
        Options are {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, default='micro'.
    """

    valid_labels: Optional[List[str]] = None
    converter_fn: Callable[[str, List[str]], str] = convert_model_output_to_label
    multiclass_average_strategy: Optional[str] = "micro"

    def __post_init__(self):
        if self.valid_labels:
            for i, label in enumerate(self.valid_labels):
                if not isinstance(label, str):
                    warnings.warn("Valid labels should be strings, casting.")
                    self.valid_labels[i] = str(label)


class ClassificationAccuracy(EvalAlgorithmInterface):
    """This evaluation measures how accurately a model performs in text classification tasks. Our built-in example task is sentiment classification where the model predicts whether a user review is positive or negative.
    The accuracy of its response is measured by comparing model output to target answer under different metrics:

    1. Classification accuracy: Is `model_output == target_answer`? This metric is computed for each datapoint as well as on average over the whole dataset.
    2. Precision: true positives / (true positives + false positives), computed once for the whole dataset. Its parameter `multiclass_average_stategy` can be set in the `ClassificationAccuracyConfig`.
    3. Recall: true positives / (true positives + false negatives), computed once for the whole dataset. Its parameter `multiclass_average_stategy` can be set in the `ClassificationAccuracyConfig`.
    4. Balanced classification accuracy: Same as accuracy in the binary case, otherwise averaged recall per class. This metric is computed once for the whole dataset.

    All metrics are reported on average over `num_records` datapoints and per category, resulting in a number between 0
    (worst) and 1 (best) for each metric.

    """

    eval_name = EvalAlgorithm.CLASSIFICATION_ACCURACY.value

    def __init__(self, eval_algorithm_config: ClassificationAccuracyConfig = ClassificationAccuracyConfig()):
        """Default constructor

        :param eval_algorithm_config: Classification Accuracy eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.valid_labels = eval_algorithm_config.valid_labels
        self.converter_fn = eval_algorithm_config.converter_fn
        self.multiclass_average_strategy = eval_algorithm_config.multiclass_average_strategy

    def evaluate_sample(self, target_output: str, model_output: str) -> List[EvalScore]:
        """Compute classification accuracy metrics for a single sample.

        :param target_output: The expected/desired model output.
        :param model_output: The actual model output.
        :returns: A single-element list with an EvalScore for the classification accuracy score.
        """
        util.require(
            self.valid_labels,
            "ClassificationAccuracy evaluate_sample method requires the `valid_labels` "
            "attribute of the ClassificationAccuracy instance to be set.",
        )
        sample = {
            DatasetColumns.TARGET_OUTPUT.value.name: target_output,
            DatasetColumns.MODEL_OUTPUT.value.name: model_output,
        }
        pipeline = self._build_pipeline(self.valid_labels)
        result = pipeline.execute_record(sample)
        return [
            EvalScore(
                name=CLASSIFICATION_ACCURACY_SCORE,
                value=result[CLASSIFICATION_ACCURACY_SCORE],  # type: ignore
            )
        ]

    def _build_pipeline(self, valid_labels: Optional[List[str]]) -> TransformPipeline:
        return TransformPipeline(
            [ClassificationAccuracyScores(valid_labels=valid_labels, converter_fn=self.converter_fn)]
        )

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[Union[DataConfig, List[DataConfig]]] = None,
        prompt_template: Optional[str] = None,
        num_records: int = 100,
        save: bool = False,
        save_strategy: Optional[SaveStrategy] = None,
    ) -> List[EvalOutput]:
        """Compute classification accuracy metrics on one or more datasets.

        :param model: An instance of ModelRunner representing the model under evaluation.
            If this argument is None, the `dataset_config` argument must not be None,
            and must correspond to a dataset that already contains a column with model outputs.
        :param dataset_config: Configures a single dataset or list of datasets used for the
            evaluation. If not provided, this method will run evaluations using all of its
            supported built-in datasets.
        :param prompt_template: A template used to generate prompts that are fed to the model.
            If not provided, defaults will be used. If provided, `model` must not be None.
        :param num_records: The number of records to be sampled randomly from the input dataset(s)
            used to perform the evaluation(s).
        :param save: If set to true, prompt responses and scores will be saved to a file.
        :param save_strategy: Specifies the strategy to use to save the localized outputs of the evaluations. If not
            specified, it will save it to the path that can be configured by the EVAL_RESULTS_PATH environment variable.
            If that environment variable is also not configured, it will be saved to the default path `/tmp/eval_results/`.

        :return: A list of EvalOutput objects.
        """
        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        eval_outputs: List[EvalOutput] = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)

            validate_dataset(dataset, [DatasetColumns.TARGET_OUTPUT.value.name])
            valid_labels = (
                self.valid_labels
                if self.valid_labels
                else dataset.unique(column=DatasetColumns.TARGET_OUTPUT.value.name)
            )
            row_count = dataset.count()
            if len(valid_labels) / (row_count + 1) < UNIQUENESS_FACTOR:  # pragma: no cover
                logger.warning(
                    f"The number of classes: {len(valid_labels)} in the dataset is too large "
                    f"for the number of rows in the dataset: {row_count}",
                )

            pipeline = self._build_pipeline(valid_labels)
            dataset_prompt_template = None
            if DatasetColumns.MODEL_OUTPUT.value.name not in dataset.columns():
                util.require(model, "No ModelRunner provided. ModelRunner is required for inference on model_inputs")
                validate_dataset(dataset, [DatasetColumns.MODEL_INPUT.value.name])
                dataset_prompt_template = (
                    get_default_prompt_template(dataset_config.dataset_name) if not prompt_template else prompt_template
                )
                model_invocation_pipeline = create_model_invocation_pipeline(model, dataset_prompt_template)
                pipeline = TransformPipeline([model_invocation_pipeline, pipeline])

            output_path = generate_output_dataset_path(
                path_to_parent_dir=util.get_eval_results_path(),
                eval_name=self.eval_name,
                dataset_name=dataset_config.dataset_name,
            )
            with timed_block(f"Computing score and aggregation on dataset {dataset_config.dataset_name}", logger):
                dataset = pipeline.execute(dataset)
                dataset_scores, category_scores = self._generate_dataset_and_category_level_scores(dataset)
                eval_outputs.append(
                    EvalOutput(
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                        prompt_template=dataset_prompt_template,
                        dataset_scores=dataset_scores,
                        category_scores=category_scores,
                        output_path=output_path,
                    )
                )

            if save:
                save_dataset(
                    dataset=dataset,
                    score_names=[CLASSIFICATION_ACCURACY_SCORE],
                    save_strategy=save_strategy if save_strategy else FileSaveStrategy(output_path),
                )

        return eval_outputs

    def _get_score(self, y_true, y_pred, score_fn: Callable[..., float]) -> float:
        """
        Method to generate accuracy score
        :param y_true: Ground truth (correct) target values.
        :param y_pred: Estimated targets as returned by a classifier.
        :param score_fn: Function for computing one of the classification accuracy scores.
        :returns: Computed score
        """
        if score_fn == recall_score or score_fn == precision_score:
            return score_fn(y_true, y_pred, average=self.multiclass_average_strategy)
        return score_fn(y_true, y_pred)

    def _generate_dataset_and_category_level_scores(
        self, dataset: Dataset
    ) -> Tuple[List[EvalScore], Optional[List[CategoryScore]]]:
        df = dataset.to_pandas()
        dataset_scores = [
            EvalScore(name=CLASSIFICATION_ACCURACY_SCORE, value=dataset.mean(CLASSIFICATION_ACCURACY_SCORE))
        ]

        for eval_score, score_fn in CLASSIFICATION_ACCURACY_SCORES_TO_FUNCS.items():
            dataset_scores.append(
                EvalScore(
                    name=eval_score,
                    value=self._get_score(
                        # TODO dataloader should ensure target output is string
                        y_true=df[DatasetColumns.TARGET_OUTPUT.value.name],
                        y_pred=df[CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME],
                        score_fn=score_fn,
                    ),
                )
            )

        category_scores: Optional[Dict[str, CategoryScore]] = None
        if DatasetColumns.CATEGORY.value.name in dataset.columns():
            category_scores = {
                name: CategoryScore(name=name, scores=[]) for name in dataset.unique(DatasetColumns.CATEGORY.value.name)
            }
            category_aggregate: Dataset = category_wise_aggregation(dataset, CLASSIFICATION_ACCURACY_SCORE, MEAN)
            for row in category_aggregate.iter_rows():
                category_scores[row[DatasetColumns.CATEGORY.value.name]].scores.append(
                    EvalScore(name=CLASSIFICATION_ACCURACY_SCORE, value=row[f"mean({CLASSIFICATION_ACCURACY_SCORE})"])
                )
                categorical_y_true = df.loc[
                    df[DatasetColumns.CATEGORY.value.name] == row[DatasetColumns.CATEGORY.value.name],
                    DatasetColumns.TARGET_OUTPUT.value.name,
                ]
                categorical_y_pred = df.loc[
                    df[DatasetColumns.CATEGORY.value.name] == row[DatasetColumns.CATEGORY.value.name],
                    CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME,
                ]
                for eval_score, score_fn in CLASSIFICATION_ACCURACY_SCORES_TO_FUNCS.items():
                    category_scores[row[DatasetColumns.CATEGORY.value.name]].scores.append(
                        EvalScore(
                            name=eval_score,
                            value=self._get_score(
                                y_true=categorical_y_true, y_pred=categorical_y_pred, score_fn=score_fn
                            ),
                        )
                    )

        return dataset_scores, list(category_scores.values()) if category_scores else None
