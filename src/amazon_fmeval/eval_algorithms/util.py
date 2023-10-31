import json
import logging
import os
import pandas as pd
import ray.data

import amazon_fmeval.util as util

from ray.data import Dataset
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from amazon_fmeval.constants import (
    CATEGORY_COLUMN_NAME,
    EVAL_OUTPUT_RECORDS_BATCH_SIZE,
    MEAN,
    NUM_ROWS_DETERMINISTIC,
)
from amazon_fmeval.eval_algorithms import EvalScore, CategoryScore
from amazon_fmeval.exceptions import EvalAlgorithmInternalError
from amazon_fmeval.model_runners.composers.composers import PromptComposer
from amazon_fmeval.model_runners.model_runner import ModelRunner
from amazon_fmeval.perf_util import timed_block
from amazon_fmeval.util import get_num_actors

logger = logging.getLogger(__name__)


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

        def _generate_prompt_column(df: pd.DataFrame) -> pd.Series:  # pragma: no cover
            """
            Map function generating the prompt column values given a batch of records in pandas format.
            """
            return pd.Series(
                data=[prompt_composer.compose(row[model_input_column_name]) for index, row in df.iterrows()]
            )

        data = data.add_column(prompt_column_name, _generate_prompt_column).materialize()
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
    Factual knowledge eval algo aggregation method.

    :param dataset: ray dataset with eval scores
    :param score_column_names: a list of column names which contain the scores

    :return a tuple containing dataset and category level scores
    """
    dataset_scores = [
        EvalScore(name=score_column_name, value=dataset_aggregation(dataset, score_column_name, agg_method))
        for score_column_name in score_column_names
    ]
    category_scores: Optional[Dict[str, CategoryScore]] = None
    if CATEGORY_COLUMN_NAME in dataset.columns():
        category_scores = {name: CategoryScore(name=name, scores=[]) for name in dataset.unique(CATEGORY_COLUMN_NAME)}
        for score_column_name in score_column_names:
            category_aggregate: Dataset = category_wise_aggregation(dataset, score_column_name, agg_method)
            for row in category_aggregate.iter_rows():
                category_scores[row[CATEGORY_COLUMN_NAME]].scores.append(
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
    category_aggregate: Dataset = dataset.groupby(CATEGORY_COLUMN_NAME)  # type: ignore
    if agg_method == MEAN:
        category_aggregate = category_aggregate.mean(score_column_name)
    else:
        raise EvalAlgorithmInternalError(f"Aggregation method {agg_method} is not supported")
    return category_aggregate


@dataclass
class EvalOutputRecord:
    """
    The schema used to define the records that get written
    to a JSON Lines file when `save_dataset` is called.

    :param model_input: the model input
    :param model_output: the model output
    :param model_log_probability: the model log probability
    :param target_output: the target output
    :param category: the category
    :param sent_more_input: the "sent more" input (used by Prompt stereotyping)
    :param sent_less_input: the "sent less" input (used by Prompt stereotyping)
    :param sent_more_input_prob: the "sent more" input probability (used by Prompt stereotyping)
    :param sent_less_input_prob: the "sent less" input probability (used by Prompt stereotyping)
    :param sent_more_output: the "sent more" output (used by Prompt stereotyping)
    :param sent_less_output: the "sent less" output (used by Prompt stereotyping)

    IMPORTANT:
        The attributes of this class MUST match the values of the
        column name constants in COLUMN_NAMES in src/constants.py.

        Reason:
        The `from_row` method validates the column names included
        in its `row` input, making sure that these column names
        match the attribute names of this class (this validation
        only occurs for column names that don't correspond to score
        names).

        Since the `row` input comes from a Ray Dataset produced by
        the `evaluate` method of an `EvalAlgorithmInterface`, the column
        names in the row must come from COLUMN_NAMES in src/constants.py.

        Thus, the attribute names of this class must match the constants
        in COLUMN_NAMES in order for the validation to make sense.
    """

    scores: List[EvalScore]
    model_input: Optional[str] = None
    model_output: Optional[str] = None
    model_log_probability: Optional[float] = None
    target_output: Optional[str] = None
    category: Optional[str] = None
    sent_more_input: Optional[str] = None
    sent_less_input: Optional[str] = None
    sent_more_input_prob: Optional[str] = None
    sent_less_input_prob: Optional[str] = None
    sent_more_output: Optional[str] = None
    sent_less_output: Optional[str] = None
    prompt: Optional[str] = None
    sent_more_prompt: Optional[str] = None
    sent_less_prompt: Optional[str] = None

    def __str__(self):
        return json.dumps(self._to_dict())

    def _to_dict(self):
        """
        Returns a dictionary representation of this instance,
        to be used when writing this object to JSON Lines.

        Note that attributes with value None are not included
        in the JSON representation. Additionally, we want the
        key "scores" to appear last in the JSON representation,
        but this attribute appears first in EvalOutputRecord's
        class definition, hence the sorting code below.
        """
        attributes = list(self.__dict__.keys())
        attributes.sort(key=lambda x: x == "scores")
        json_obj = OrderedDict(  # regular Dicts don't guarantee key order
            (attr, self.__dict__[attr]) for attr in attributes if self.__dict__[attr] is not None
        )
        json_obj["scores"] = [eval_score.__dict__ for eval_score in json_obj["scores"]]
        return json_obj

    @staticmethod
    def from_row(row: Dict[str, Union[str, float]], score_names: List[str]) -> "EvalOutputRecord":
        """
        Returns an instance of EvalOutputRecord, created from a Ray Dataset row (represented as a dict).

        Example input:
            row = {
                "model_input": "input",
                "model_output": "output",
                "rouge": 0.42,
                "bert": 0.162
            }

        Corresponding output:
            EvalOutputRecord(
                model_input="input",
                model_output="output",
                scores=[
                    EvalScore(name="rouge", value=0.42),
                    EvalScore(name="bert", value=0.162)
                ]
            )

        :param row: a Ray Dataset row represented as a dict
        :param score_names: column names included in the Ray Dataset that `row`
            is a sample of that correspond to evaluation algorithm scores
        :returns: an instance of EvalOutputRecord corresponding to `row`
        """
        eval_output_record_attribute_names = set(EvalOutputRecord.__annotations__.keys())
        non_score_columns = {}
        scores = []
        for column_name, value in row.items():
            if column_name not in score_names:  # pragma: no branch
                if column_name in eval_output_record_attribute_names:  # pragma: no branch
                    non_score_columns[column_name] = value
            else:
                assert isinstance(value, float) or isinstance(value, int)  # to satisfy Mypy
                scores.append(EvalScore(name=column_name, value=value))

        return EvalOutputRecord(
            scores=scores,
            **non_score_columns,  # type: ignore
        )


def save_dataset(dataset: Dataset, score_names: List[str], path: str) -> None:  # pragma: no cover
    """
    Writes the dataset to a JSON Lines file, where each JSON Lines object
    follows the schema defined by `EvalOutputRecord`.

    :param dataset: a Ray Dataset that is produced during the execution of
        an EvalAlgorithmInterface's `evaluate` method. This dataset is expected
        to include columns for every score computed by the evaluation algorithm.
    :param score_names: the names of the score columns in `dataset`
    :param path: a local file path to write the dataset to. The file name specified
        by this argument may not end in the extension `.jsonl`. In this case,
        we append the extension ourselves.


        Example Dataset:
         ________________________________________
        | "model_input" | "rouge" | "bert_score"|
        ----------------------------------------
        |   "hello"    |   0.5   |     0.42    |
        ---------------------------------------
        |   "world"   |  0.314  |    0.271    |
        ---------------------------------------

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
    Returns the path to be used by an EvalAlgorithmInterface when calling `save_dataset`.

    :param path_to_parent_dir: The path to the parent directory of the file to be saved.
    :param eval_name: The evaluation name provided by the EvalAlgorithmInterface.
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
    return sum([original_score.value - reference_score.value for reference_score in perturbed_input_scores]) / len(
        perturbed_input_scores
    )


def verify_model_determinism(model: ModelRunner, dataset: Dataset, prompt_column_name: str) -> bool:
    """
    Check model is not deterministic for first NUM_ROWS_DETERMINISTIC rows
    :param model: An instance of ModelRunner which is the model under evaluation
    :param dataset: a Ray Dataset that expected to include columns for prompts
    :param prompt_column_name: Prompt column name
    :return True if model is deterministic, False otherwise
    """
    for row in dataset.limit(NUM_ROWS_DETERMINISTIC).iter_rows():
        original_prompt = row[prompt_column_name]
        original_model_output = model.predict(original_prompt)[0]
        if model.predict(original_prompt)[0] != original_model_output:
            return False
    return True
