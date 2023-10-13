import json
import os
import pandas as pd
import sagemaker
import ray.data
import amazon_fmeval.util as util
import multiprocessing as mp

from ray.data import Dataset
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from amazon_fmeval.constants import CATEGORY_COLUMN_NAME, EVAL_OUTPUT_RECORDS_BATCH_SIZE, MEAN, MIME_TYPE_JSON
from amazon_fmeval.eval_algorithms import EvalScore, CategoryScore
from amazon_fmeval.exceptions import EvalAlgorithmInternalError
from amazon_fmeval.model_runners.composers.composers import PromptComposer
from amazon_fmeval.model_runners.model_runner import ModelRunner
from amazon_fmeval.model_runners.sm_jumpstart_model_runner import JumpStartModelRunner
from amazon_fmeval.model_runners.sm_model_runner import SageMakerModelRunner
from amazon_fmeval.model_runners.util import get_sagemaker_session, is_endpoint_in_service


def generate_model_predict_response_for_dataset(
    model: ModelRunner,
    data: Dataset,
    model_input_column_name: str,
    model_output_column_name: Optional[str] = None,
    model_log_probability_column_name: Optional[str] = None,
) -> Dataset:
    """
    Runs the model on the given data. Output will be written
    to the `model_output_column_name` column, and log_probability will be written to the
    `model_log_probability_column_name`
    :param model: ModelRunner
    :param data: the dataset where each instance is a row in the dataset.
    :param model_input_column_name: the name of the column containing the model input.
    :param model_output_column_name: the name of the column to write the model output to.
    :param model_log_probability_column_name: the name of the column to write the model log probability to.
    :return: the dataset with the model output added.
    """
    if isinstance(model, JumpStartModelRunner) or isinstance(model, SageMakerModelRunner):  # pragma: no cover
        predictor_back_up = model.predictor
        sagemaker_session_back_up = model.sagemaker_session
        model.predictor = None
        model.sagemaker_session = None

    class ModelRunnerWrapper:  # pragma: no cover
        def __init__(self):
            self.model_runner = model
            self.sagemaker_session = get_sagemaker_session()
            if isinstance(model, JumpStartModelRunner) or isinstance(model, SageMakerModelRunner):
                self.model_runner.predictor = sagemaker.predictor.retrieve_default(
                    endpoint_name=model.endpoint_name,
                    model_id=model.model_id,
                    model_version=model.model_version,
                    sagemaker_session=self.sagemaker_session,
                )
                util.require(
                    self.model_runner.predictor.accept == MIME_TYPE_JSON,
                    f"Model accept type `{self.model_runner.predictor.accept}` is not supported.",
                )
                util.require(
                    self.model_runner.predictor.content_type == MIME_TYPE_JSON,
                    f"Model content type `{self.model_runner.predictor.content_type}` is not supported.",
                )
                util.require(
                    is_endpoint_in_service(self.sagemaker_session, self.model_runner.endpoint_name),
                    "Endpoint is not in service",
                )

        def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
            predict_output = self.model_runner.predict(row[model_input_column_name])
            if model_output_column_name:
                row[model_output_column_name] = predict_output[0]
            if model_log_probability_column_name:
                row[model_log_probability_column_name] = predict_output[1]
            return row

    result_data = data.map(ModelRunnerWrapper, compute=ray.data.ActorPoolStrategy(size=mp.cpu_count() - 1)).materialize()  # type: ignore [arg-type]
    if isinstance(model, JumpStartModelRunner) or isinstance(model, SageMakerModelRunner):  # pragma: no cover
        model.predictor = predictor_back_up
        model.sagemaker_session = sagemaker_session_back_up
    return result_data


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
    prompt_composer = PromptComposer(prompt_template)

    def _generate_prompt_column(df: pd.DataFrame) -> pd.Series:  # pragma: no cover
        """
        Map function generating the prompt column values given a batch of records in pandas format.
        """
        return pd.Series(data=[prompt_composer.compose(row[model_input_column_name]) for index, row in df.iterrows()])

    return data.add_column(prompt_column_name, _generate_prompt_column)


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

    Attributes:
        model_input: the model input
        model_output: the model output
        model_log_probability: the model log probability
        target_output: the target output
        category: the category
        sent_more_input: the "sent more" input (used by Prompt stereotyping)
        sent_less_input: the "sent less" input (used by Prompt stereotyping)
        sent_more_input_prob: the "sent more" input probability (used by Prompt stereotyping)
        sent_less_input_prob: the "sent less" input probability (used by Prompt stereotyping)
        sent_more_output: the "sent more" output (used by Prompt stereotyping)
        sent_less_output: the "sent less" output (used by Prompt stereotyping)

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


ADDITIONAL_MODEL_OUTPUT_COLUMN_NAME = "additional_model_output"


def is_predictor_deterministic(
    dataset: Dataset,
    model: ModelRunner,
    prompt_column_name: str,
    model_output_column_name: str,
    num_additional_predictions: int = 1,
) -> bool:
    """
    Util to determine if a model is deterministic. Method invokes model with inputs in a dataset multiple times
    and compares predictions.
    :param dataset: Ray dataset which contains input prompts and original output
    :param model: ModelRunner
    :param prompt_column_name: name of input prompt column in input dataset
    :param model_output_column_name: name of input model output column in input dataset
    :param num_additional_predictions: Number of additional inference requests to make
    :returns: True is model response is deterministic else False.

    Note: Currently we are comparing only model outputs in this utility. We can compare input log probabilities too
    if we get a use case for that in eval algos.
    """
    for idx in range(num_additional_predictions):
        dataset = generate_model_predict_response_for_dataset(
            model, dataset, prompt_column_name, f"{ADDITIONAL_MODEL_OUTPUT_COLUMN_NAME}_{idx}"
        )

    for row in dataset.iter_rows():
        is_prediction_same = all(
            [
                row[model_output_column_name] == row[f"{ADDITIONAL_MODEL_OUTPUT_COLUMN_NAME}_{idx}"]
                for idx in range(num_additional_predictions)
            ]
        )
        if not is_prediction_same:
            return False
    return True
