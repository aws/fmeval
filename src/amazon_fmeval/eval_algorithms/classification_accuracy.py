import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd
from ray.data import Dataset
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

import amazon_fmeval.util as util
from amazon_fmeval.constants import (
    MODEL_OUTPUT_COLUMN_NAME,
    MODEL_INPUT_COLUMN_NAME,
    TARGET_OUTPUT_COLUMN_NAME,
    MEAN,
    CATEGORY_COLUMN_NAME,
)
from amazon_fmeval.data_loaders.util import get_dataset
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.eval_algorithms.eval_algorithm import (
    EvalAlgorithmInterface,
    EvalAlgorithmConfig,
)
from amazon_fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalOutput,
    EvalScore,
    EVAL_DATASETS,
    EVAL_PROMPT_TEMPLATES,
    DATASET_CONFIGS,
    CategoryScore,
)
from amazon_fmeval.eval_algorithms.util import (
    generate_prompt_column_for_dataset,
    generate_model_predict_response_for_dataset,
    validate_dataset,
    category_wise_aggregation,
    save_dataset,
    generate_output_dataset_path,
)
from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.model_runner import ModelRunner


CLASSIFICATION_ACCURACY_SCORE = "classification_accuracy_score"
BALANCED_ACCURACY_SCORE = "balanced_accuracy_score"
PRECISION_SCORE = "precision_score"
RECALL_SCORE = "recall_score"

UNKNOWN_LABEL = "unknown"

PROMPT_COLUMN_NAME = "prompt"

CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME = "classified_model_output"


CLASSIFICATION_ACCURACY_SCORES_TO_FUNCS: Dict[str, Callable[..., float]] = {
    BALANCED_ACCURACY_SCORE: balanced_accuracy_score,
    PRECISION_SCORE: precision_score,
    RECALL_SCORE: recall_score,
}


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


@dataclass(frozen=True)
class ClassificationAccuracyConfig(EvalAlgorithmConfig):
    """
    Configuration for the Classification Accuracy Evaluation

    :param multiclass_average_strategy: `average` to be passed to sklearn's precision and recall scores.
        This determines how scores are aggregated in the multiclass classification setting (see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html).
        Options are {'micro', 'macro', 'samples', 'weighted', 'binary'} or None, default='micro'.
    :param converter_fn: Function to process model output to labels, defaults to simple integer conversion.
    """

    valid_labels: List[str]
    converter_fn: Callable[[str, List[str]], str] = convert_model_output_to_label
    multiclass_average_strategy: Optional[str] = "micro"

    def __post_init__(self):
        for i, label in enumerate(self.valid_labels):
            if not isinstance(label, str):
                warnings.warn("Valid labels should be strings, casting.")
                self.valid_labels[i] = str(label)


class ClassificationAccuracy(EvalAlgorithmInterface):
    eval_name = EvalAlgorithm.CLASSIFICATION_ACCURACY.value

    def __init__(self, eval_algorithm_config: ClassificationAccuracyConfig):
        """Default constructor

        :param eval_algorithm_config: Classification Accuracy eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self._eval_algorithm_config = eval_algorithm_config

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        save: bool = False,
    ) -> List[EvalOutput]:
        is_custom_dataset_evaluation = False
        if dataset_config:
            is_custom_dataset_evaluation = True
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[self.eval_name]]
        eval_outputs: List[EvalOutput] = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config)
            validate_dataset(dataset, [TARGET_OUTPUT_COLUMN_NAME, MODEL_INPUT_COLUMN_NAME])
            if MODEL_OUTPUT_COLUMN_NAME not in dataset.columns():
                util.require(model, "No ModelRunner provided. ModelRunner is required for inference on model_inputs")
                if is_custom_dataset_evaluation:
                    # TODO when user provide built-in DataConfig, we should provide default prompt_template
                    util.require(
                        prompt_template is not None,
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
                    prompt_template=prompt_template,
                    data=dataset,
                    model_input_column_name=MODEL_INPUT_COLUMN_NAME,
                    prompt_column_name=PROMPT_COLUMN_NAME,
                )
                assert model  # to satisfy mypy
                dataset = generate_model_predict_response_for_dataset(
                    model=model,
                    data=dataset,
                    model_input_column_name=PROMPT_COLUMN_NAME,
                    model_output_column_name=MODEL_OUTPUT_COLUMN_NAME,
                )

            def _generate_classified_model_output_column(df: pd.DataFrame) -> pd.Series:  # pragma: no cover
                return pd.Series(
                    data=[
                        self._eval_algorithm_config.converter_fn(
                            row[MODEL_OUTPUT_COLUMN_NAME], self._eval_algorithm_config.valid_labels
                        )
                        for index, row in df.iterrows()
                    ]
                )

            dataset = dataset.add_column(CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME, _generate_classified_model_output_column)
            dataset = dataset.materialize()

            def _generate_classification_accuracy_column(df: pd.DataFrame) -> pd.Series:  # pragma: no cover
                return pd.Series(
                    data=[
                        int(row[CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME] == str(row[TARGET_OUTPUT_COLUMN_NAME]))
                        for index, row in df.iterrows()
                    ]
                )

            dataset = dataset.add_column(CLASSIFICATION_ACCURACY_SCORE, _generate_classification_accuracy_column)
            dataset = dataset.materialize()

            df = dataset.to_pandas()
            dataset_scores = [
                EvalScore(name=CLASSIFICATION_ACCURACY_SCORE, value=dataset.mean(CLASSIFICATION_ACCURACY_SCORE))
            ]

            for eval_score, eval_fn in CLASSIFICATION_ACCURACY_SCORES_TO_FUNCS.items():
                dataset_scores.append(
                    EvalScore(
                        name=eval_score,
                        value=self._get_score(
                            # TODO dataloader should ensure target output is string
                            y_true=df[TARGET_OUTPUT_COLUMN_NAME],
                            y_pred=df[CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME],
                            eval_fn=eval_fn,
                        ),
                    )
                )

            category_scores: Optional[Dict[str, CategoryScore]] = None
            if CATEGORY_COLUMN_NAME in dataset.columns():
                category_scores = {
                    name: CategoryScore(name=name, scores=[]) for name in dataset.unique(CATEGORY_COLUMN_NAME)
                }
                category_aggregate: Dataset = category_wise_aggregation(dataset, CLASSIFICATION_ACCURACY_SCORE, MEAN)
                for row in category_aggregate.iter_rows():
                    category_scores[row[CATEGORY_COLUMN_NAME]].scores.append(
                        EvalScore(
                            name=CLASSIFICATION_ACCURACY_SCORE, value=row[f"mean({CLASSIFICATION_ACCURACY_SCORE})"]
                        )
                    )
                    categorical_y_true = df.loc[
                        df[CATEGORY_COLUMN_NAME] == row[CATEGORY_COLUMN_NAME], TARGET_OUTPUT_COLUMN_NAME
                    ]
                    categorical_y_pred = df.loc[
                        df[CATEGORY_COLUMN_NAME] == row[CATEGORY_COLUMN_NAME], CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME
                    ]
                    for eval_score, eval_fn in CLASSIFICATION_ACCURACY_SCORES_TO_FUNCS.items():
                        category_scores[row[CATEGORY_COLUMN_NAME]].scores.append(
                            EvalScore(
                                name=eval_score,
                                value=self._get_score(
                                    y_true=categorical_y_true, y_pred=categorical_y_pred, eval_fn=eval_fn
                                ),
                            )
                        )

            eval_outputs.append(
                EvalOutput(
                    eval_name=self.eval_name,
                    dataset_name=dataset_config.dataset_name,
                    prompt_template=prompt_template,
                    dataset_scores=dataset_scores,
                    category_scores=list(category_scores.values()) if category_scores else None,
                    output_path=self._eval_results_path,
                )
            )
            if save:
                save_dataset(
                    dataset=dataset,
                    score_names=[CLASSIFICATION_ACCURACY_SCORE],
                    path=generate_output_dataset_path(
                        path_to_parent_dir=self._eval_results_path,
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                    ),
                )

        return eval_outputs

    def _get_score(self, y_true, y_pred, eval_fn: Callable[..., float]) -> float:
        """
        Method to generate accuracy score
        :param y_true: Ground truth (correct) target values.
        :param y_pred: Estimated targets as returned by a classifier.
        :param eval_fn: Score evaluate function.
        :returns: Computed score
        """
        if eval_fn == recall_score or eval_fn == precision_score:
            return eval_fn(y_true, y_pred, average=self._eval_algorithm_config.multiclass_average_strategy)
        return eval_fn(y_true, y_pred)

    def evaluate_sample(self, target_output: str, model_output: str) -> List[EvalScore]:  # type: ignore[override]
        """
        Evaluate a single Classification record.

        :param model_output: An instance of ModelOutput which contains the responses from the model needed for this
                             evaluation.
        :param target_output: The expected responses from the model.
        :returns: A List of EvalScores computed for prompts and responses.
        """
        if target_output is None:
            raise EvalAlgorithmClientError(
                "Missing required input: target_output, for Classification Accuracy evaluate_sample"
            )
        if model_output is None:
            raise EvalAlgorithmClientError(
                "Missing required input: model_output, for Classification Accuracy evaluate_sample"
            )
        return [
            EvalScore(
                name=CLASSIFICATION_ACCURACY_SCORE,
                value=int(
                    self._eval_algorithm_config.converter_fn(model_output, self._eval_algorithm_config.valid_labels)
                    == str(target_output)
                ),
            )
        ]
