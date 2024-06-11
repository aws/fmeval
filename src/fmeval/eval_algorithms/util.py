import json
import logging
import os
import ray.data

import fmeval.util as util

from ray.data import Dataset
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from fmeval.constants import (
    DatasetColumns,
    EVAL_OUTPUT_RECORDS_BATCH_SIZE,
    MEAN,
    NUM_ROWS_DETERMINISTIC,
    DATASET_COLUMNS,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms import (
    EvalScore,
    CategoryScore,
    DATASET_CONFIGS,
    EVAL_DATASETS,
    EvalOutput,
    get_default_prompt_template,
)
from fmeval.exceptions import EvalAlgorithmInternalError, EvalAlgorithmClientError
from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.perf_util import timed_block
from fmeval.transforms.common import GeneratePrompt, GetModelOutputs
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.util import get_num_actors

logger = logging.getLogger(__name__)


def get_dataset_configs(data_config: Optional[DataConfig], eval_name: str) -> List[DataConfig]:
    return (
        [data_config] if data_config else [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[eval_name]]
    )


def generate_model_predict_response_for_dataset(
    model: ModelRunner,
    data: Dataset,
    model_input_column_name: str,
    model_output_column_name: Optional[str] = None,
    model_log_probability_column_name: Optional[str] = None,
) -> Dataset:
    """
    Runs the model on the given data. Output will be written to the
    `model_output_column_name` column, and log_probability will be
    written to the `model_log_probability_column_name` column.

    :param model: ModelRunner to get predictions from.
    :param data: The dataset containing model inputs to feed to `model`.
    :param model_input_column_name: The name of the column containing the model input.
    :param model_output_column_name: The name of the column to write the model output to.
    :param model_log_probability_column_name: The name of the column to write the model log probability to.
    :return: The dataset with a model output column and model log probability column added.
        Note that both columns are optional, i.e. it is possible that a model output
        column is added, but a log probability column is not added (and vice versa).
    """
    with timed_block(f"Performing inference on dataset on {model}", logger):

        class ModelRunnerWrapper:  # pragma: no cover
            """
            This class represents the Ray Actor that gets model predictions
            by feeding model inputs from the dataset to the model runner.

            We use Ray Actors instead of Tasks because the Actor approach minimizes
            the number of times that the ModelRunner `model` gets deserialized.
            With Tasks, Ray will serialize and deserialize `model` for every single
            prediction. With Actors, `model` gets deserialized once per Actor when
            the Actor gets initialized.
            """

            def __init__(self):
                self.model_runner = model
                logger.setLevel(logging.DEBUG)

            def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
                predict_output = self.model_runner.predict(row[model_input_column_name])
                if model_output_column_name:
                    row[model_output_column_name] = predict_output[0]
                if model_log_probability_column_name:
                    row[model_log_probability_column_name] = predict_output[1]
                return row

        data = data.map(
            ModelRunnerWrapper, compute=ray.data.ActorPoolStrategy(size=get_num_actors())  # type: ignore[arg-type]
        ).materialize()
    return data


def generate_prompt_column_for_dataset(
    prompt_template: str, data: Dataset, model_input_column_name: str, prompt_column_name: str
) -> Dataset:
    """
    Generates prompts column for a given input dataset and prompt_template
    :param prompt_template: Prompt template
    :param data: the dataset where each instance is a row in the dataset.
    :param model_input_column_name: the name of the column containing the model input.
    :param prompt_column_name: Output column name to which composed prompts are added
    :return: the dataset with the composed prompts added.
    """
    with timed_block(f"Generating prompt column", logger):
        prompt_composer = PromptComposer(prompt_template)

        def _generate_prompt_column(row: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
            """
            Map function for generating the prompt column value given a dataset row.
            """
            row[prompt_column_name] = prompt_composer.compose(row[model_input_column_name])
            return row

        data = data.map(_generate_prompt_column).materialize()
    return data


def validate_dataset(dataset: Dataset, column_names: List[str]):
    """
    Util function to validate that dataset contains the required column names.

    :param dataset: Input ray dataset
    :param column_names: names of the columns that must be present in the dataset
    :raises: EvalAlgorithmClientError for an invalid dataset
    """
    for column_name in column_names:
        util.require(
            column_name in dataset.columns(),
            f"Missing required column: {column_name}, for evaluate() method",
        )


def aggregate_evaluation_scores(
    dataset: Dataset, score_column_names: List[str], agg_method: str
) -> Tuple[List[EvalScore], Optional[List[CategoryScore]]]:
    """
    The method aggregates scores at the dataset level and optionally at the category level if
     categories are available in the dataset.

    :param dataset: ray dataset with eval scores
    :param score_column_names: a list of column names which contain the scores to aggregate
    :param agg_method: the name of the aggregation to perform
    :return: a tuple containing 1) dataset-level scores and
                                2) a list of category-level scores if categories are available, `None` otherwise
    """
    dataset_scores = [
        EvalScore(name=score_column_name, value=dataset_aggregation(dataset, score_column_name, agg_method))
        for score_column_name in score_column_names
    ]
    category_scores: Optional[Dict[str, CategoryScore]] = None
    if DatasetColumns.CATEGORY.value.name in dataset.columns():
        category_scores = {
            name: CategoryScore(name=name, scores=[]) for name in dataset.unique(DatasetColumns.CATEGORY.value.name)
        }
        for score_column_name in score_column_names:
            category_aggregate: Dataset = category_wise_aggregation(dataset, score_column_name, agg_method)
            for row in category_aggregate.iter_rows():
                category_scores[row[DatasetColumns.CATEGORY.value.name]].scores.append(
                    EvalScore(name=score_column_name, value=row[f"mean({score_column_name})"])
                )

    return dataset_scores, list(category_scores.values()) if category_scores else None


def dataset_aggregation(dataset: Dataset, score_column_name: str, agg_method: str) -> float:
    if agg_method == MEAN:
        aggregate = dataset.mean(score_column_name)
        assert isinstance(aggregate, float)
        return aggregate
    else:
        raise EvalAlgorithmInternalError(f"Aggregation method {agg_method} is not supported")


def category_wise_aggregation(dataset: Dataset, score_column_name: str, agg_method: str) -> Dataset:
    category_aggregate: Dataset = dataset.groupby(DatasetColumns.CATEGORY.value.name)  # type: ignore
    if agg_method == MEAN:
        category_aggregate = category_aggregate.mean(score_column_name)
    else:
        raise EvalAlgorithmInternalError(f"Aggregation method {agg_method} is not supported")
    return category_aggregate


@dataclass
class EvalOutputRecord:
    """
    This class represents a single record that gets written by the `save_dataset` method.
    In other words, it represents a single row from the Ray Dataset that is being saved.

    :param scores: A list of EvalScores, where each EvalScore corresponds
        to one of the score columns in the Ray Dataset being saved.
    :param dataset_columns: Maps a column name to its contents in the current row
        (recall that an EvalOutputRecord corresponds to a single Ray Dataset row).

        Note: the keys in `dataset_columns` must belong to constants.COLUMN_NAMES,
        because constants.COLUMN_NAMES defines which (non-score) columns are allowed
        to appear in the saved output, i.e. it defines the schema for an output record.
    """

    scores: List[EvalScore]
    dataset_columns: Dict[str, Union[str, float, int]]

    def __post_init__(self):
        for col in self.dataset_columns:
            util.assert_condition(
                col in DATASET_COLUMNS,
                f"Attempting to initialize an EvalOutputRecord with invalid non-score column {col}.",
            )

    def __str__(self):
        return json.dumps(self._to_dict())

    def _to_dict(self) -> OrderedDict[str, Union[str, float, int]]:
        """
        Returns a dictionary representation of this instance,
        to be used when writing this object to JSON Lines.

        Note that we use an OrderedDict to maintain consistency
        in the ordering of columns. The score columns always come
        at the end, and the non-score columns are ordered according
        to constants.COLUMN_NAMES.
        """
        json_obj = OrderedDict(
            (col_name, self.dataset_columns[col_name])
            for col_name in DATASET_COLUMNS
            if col_name in self.dataset_columns
        )
        json_obj["scores"] = [eval_score.__dict__ for eval_score in self.scores]
        return json_obj

    @staticmethod
    def from_row(row: Dict[str, Union[str, float, int]], score_names: List[str]) -> "EvalOutputRecord":
        """
        Returns an instance of EvalOutputRecord, created from a Ray Dataset row (represented as a dict).

        Example input:
            row = {
                "model_input": "input",
                "model_output": "output",
                "column_that_wont_be_included": "hello",
                "rouge": 0.42,
                "bert": 0.162
            }

        Corresponding output:
            EvalOutputRecord(
                scores=[
                    EvalScore(name="rouge", value=0.42),
                    EvalScore(name="bert", value=0.162)
                ],
                dataset_columns={
                    "model_input": "input",
                    "model_output": "output"
                }
            )

        Note how "column_that_wont_be_included" is not included in the produced EvalOutputRecord.
        This is because only columns in constants.COLUMN_NAMES are considered to be valid columns
        in the saved output file generated by `save_dataset`. The reason why it's even possible
        for a column name that doesn't belong to constants.COLUMN_NAMES to appear in `row` is that
        the Ray Dataset that `row` belongs to can contain columns used to store intermediate computations.
        For example, ClassificationAccuracy generates a column named CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME
        that is used to compute CLASSIFICATION_ACCURACY_SCORE, which is one of the score columns.

        :param row: a Ray Dataset row represented as a dict
        :param score_names: column names included in the Ray Dataset that `row`
            is a sample of that correspond to evaluation algorithm scores
        :returns: an instance of EvalOutputRecord corresponding to `row`
        """
        dataset_columns = {}
        scores = []
        for column_name, value in row.items():
            if column_name not in score_names:  # pragma: no branch
                if column_name in DATASET_COLUMNS:  # pragma: no branch
                    dataset_columns[column_name] = value
            else:
                assert isinstance(value, float) or isinstance(value, int)  # to satisfy Mypy
                scores.append(EvalScore(name=column_name, value=value))

        return EvalOutputRecord(
            scores=scores,
            dataset_columns=dataset_columns,
        )


def save_dataset(dataset: Dataset, score_names: List[str], path: str) -> None:  # pragma: no cover
    """
    Writes the dataset to a JSON Lines file, where each JSON Lines object
    is the JSON representation of an `EvalOutputRecord`.

    :param dataset: a Ray Dataset that is produced during the execution of
        an EvalAlgorithm's `evaluate` method. This dataset is expected
        to include columns for every score computed by the evaluation algorithm.
    :param score_names: the names of the score columns in `dataset`.
    :param path: a local file path to write the dataset to. The file name specified
        by this argument may not end in the extension `.jsonl`. In this case,
        we append the extension ourselves.


        Example Dataset:
         ________________________________________________
        | "model_input" | "aux" | "rouge" | "bert_score"|
        -------------------------------------------------
        |    "hello"    | 0.189 |   0.5   |     0.42    |
        -------------------------------------------------
        |    "world"    | 0.162 |  0.314  |    0.271    |
        -------------------------------------------------

        Note that the "aux" column name does not belong to constants.COLUMN_NAMES, meaning that this column
        won't get included in the saved outputs. See the docstring for EvalOutputRecord.from_row for more details.

        Corresponding Json Lines file contents:
        {"model_input" : "hello", "scores" : [{"name": "rouge", "value": 0.5}, {"name": "bert_score", "value": 0.42}]}
        {"model_input" : "world", "scores" : [{"name": "rouge", "value": 0.314}, {"name": "bert_score", "value": 0.271}]}


    """
    with timed_block(f"Saving dataset to file", logger):
        # We need the outer dict that wraps the EvalOutputRecord because map() requires
        # whatever is returned from the lambda function to be a dict
        dataset = dataset.map(lambda row: {"record": EvalOutputRecord.from_row(row, score_names)})
        # Without this line, dataset.iter_rows() below is not guaranteed to return the rows
        # in the same order that they appear in `dataset`.
        dataset.materialize()

        path_to_parent_dir = os.path.dirname(path)
        file_name = os.path.basename(path)
        file_name_without_extension = os.path.splitext(file_name)[0]
        full_path = f"{path_to_parent_dir}/{file_name_without_extension}.jsonl"
        with open(full_path, "w") as fh:
            records = []
            for dataset_row in dataset.iter_rows():
                record = dataset_row["record"]
                records.append(str(record))
                if len(records) == EVAL_OUTPUT_RECORDS_BATCH_SIZE:
                    fh.write("\n".join(records) + "\n")
                    records = []
            if records:  # pragma: no branch
                fh.write("\n".join(records))  # handle the last batch


def generate_output_dataset_path(path_to_parent_dir: str, eval_name: str, dataset_name) -> str:
    """
    Returns the path to be used by an EvalAlgorithm when calling `save_dataset`.

    :param path_to_parent_dir: The path to the parent directory of the file to be saved.
    :param eval_name: The evaluation name provided by the EvalAlgorithm.
    :param dataset_name: The name of the dataset.
    :returns: A path that is unique to an evaluation/dataset pair for a given job.
    """
    return os.path.join(path_to_parent_dir, f"{eval_name}_{dataset_name}.jsonl")


def generate_mean_delta_score(original_score: EvalScore, perturbed_input_scores: List[EvalScore]) -> float:
    """
    Util method to generate mean of difference between original and perturbed input scores
    :param original_score: Original score
    :param perturbed_input_scores: List of scores for model inference outputs on perturbed inputs
    :returns: mean of delta between the scores
    """
    return sum([abs(original_score.value - reference_score.value) for reference_score in perturbed_input_scores]) / len(
        perturbed_input_scores
    )


def verify_model_determinism(
    model: ModelRunner,
    dataset: Dataset,
    prompt_template: str,
    model_input_column_name: str = DatasetColumns.MODEL_INPUT.value.name,
) -> bool:
    """Heuristic for whether model is deterministic.

    This function invokes the provided model twice on each of the first
    NUM_ROWS_DETERMINISTIC rows in the dataset. If the two model outputs
    for each input are the same for all rows, the model is considered deterministic.

    :param model: A ModelRunner instance representing the model under investigation.
    :param dataset: A Ray Dataset that includes a model input column.
    :param prompt_template: The template used to compose the prompt from the model input.
    :param model_input_column_name: Model input column name.
    :returns: Whether the model is deterministic.
    """
    prompt_composer = PromptComposer(prompt_template)
    for row in dataset.limit(NUM_ROWS_DETERMINISTIC).iter_rows():
        prompt = prompt_composer.compose(row[model_input_column_name])
        model_output = model.predict(prompt)[0]
        if model.predict(prompt)[0] != model_output:
            return False
    return True


def create_model_invocation_pipeline(model: ModelRunner, prompt_template: str) -> TransformPipeline:
    """Create a transform pipeline for performing the standard action of invoking a model on a prompt.

    :param model: The model to be invoked.
    :param prompt_template: The template used for constructing prompts (out of raw inputs)
        that will be fed to the model.
    :returns: A TransformPipeline instance containing a GeneratePrompt transform that uses `prompt_template`
        and a GetModelOutputs transform for invoking the model on the generated prompts.
    """
    gen_prompt = GeneratePrompt(
        input_keys=[DatasetColumns.MODEL_INPUT.value.name],
        output_keys=[DatasetColumns.PROMPT.value.name],
        prompt_template=prompt_template,
    )
    get_model_outputs = GetModelOutputs(
        input_to_output_keys={DatasetColumns.PROMPT.value.name: [DatasetColumns.MODEL_OUTPUT.value.name]},
        model_runner=model,
    )
    return TransformPipeline([gen_prompt, get_model_outputs])


def evaluate_dataset(
    dataset: Dataset,
    pipeline: TransformPipeline,
    dataset_name: str,
    eval_name: str,
    metric_names: List[str],
    eval_results_path: str,
    model: Optional[ModelRunner] = None,
    prompt_template: Optional[str] = None,
    agg_method: str = MEAN,
    save: bool = False,
) -> EvalOutput:
    """Execute an evaluation algorithm's pipeline on a dataset.

    :param dataset: The dataset to be evaluated.
    :param pipeline: The evaluation algorithm's pipeline, to be executed on the dataset.
    :param dataset_name: The name of the dataset being evaluated. This is metadata that
        will be included in the returned EvalOutput object.
    :param eval_name: The name of the evaluation algorithm.
    :param metric_names: The names of the metrics that this evaluation algorithm computes.
        prior to performing any evaluation logic. This parameter is algorithm-specific.
    :param eval_results_path: A file containing evaluation results will be stored at this path.
    :param model: An instance of ModelRunner representing the model under evaluation.
        If this argument is None, model responses cannot be obtained. In such cases,
        the dataset configured by `dataset_config` should already contain a column for
        model outputs.
    :param prompt_template: A template used to generate prompts that are fed to the model.
        If set to None, a default value will be used. Note that if this argument is not None,
        `model` must also not be None.
    :param agg_method: The aggregation method to use when aggregating the computed metric values.
        Currently, only MEAN is supported.
    :param save: If set to true, prompt responses and scores will be saved to a file.
        The path that this file is stored at is configured by `eval_results_path`.

    :return: An EvalOutput object encapsulating the results of the evaluation.
    """
    if model:
        try:
            validate_dataset(dataset, [DatasetColumns.MODEL_INPUT.value.name])
        except EvalAlgorithmClientError:
            raise EvalAlgorithmClientError(
                "evaluate_dataset has been given a ModelRunner to obtain outputs from "
                "but the provided dataset does not contain a model input column."
            )
        prompt_template = get_default_prompt_template(dataset_name) if not prompt_template else prompt_template
        model_invocation_pipeline = create_model_invocation_pipeline(model, prompt_template)
        pipeline = TransformPipeline([model_invocation_pipeline, pipeline])
    else:
        if prompt_template:
            logger.warning(
                "A prompt template, but no corresponding model, was provided."
                "Model outputs from the dataset will be used, and this prompt template will be ignored."
            )
        try:
            validate_dataset(dataset, [DatasetColumns.MODEL_OUTPUT.value.name])
        except EvalAlgorithmClientError:
            raise EvalAlgorithmClientError(
                "evaluate_dataset has been given a dataset with no model output column "
                "and no ModelRunner to obtain outputs from. Please either provide a model "
                "or use a dataset that contains model outputs already."
            )

    with (timed_block(f"Computing score and aggregation on dataset {dataset_name}", logger)):
        dataset = pipeline.execute(dataset)
        dataset_scores, category_scores = aggregate_evaluation_scores(dataset, metric_names, agg_method=agg_method)

        output_path = generate_output_dataset_path(
            path_to_parent_dir=eval_results_path,
            eval_name=eval_name,
            dataset_name=dataset_name,
        )
        eval_output = EvalOutput(
            eval_name=eval_name,
            dataset_name=dataset_name,
            prompt_template=prompt_template,
            dataset_scores=dataset_scores,
            category_scores=category_scores,
            output_path=output_path,
        )

        if save:  # pragma: no branch
            save_dataset(
                dataset=dataset,
                score_names=metric_names,
                path=output_path,
            )

        return eval_output
