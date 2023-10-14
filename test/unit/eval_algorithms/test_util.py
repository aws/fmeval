import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple, Optional
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import ray

from amazon_fmeval.constants import CATEGORY_COLUMN_NAME, EVAL_OUTPUT_RECORDS_BATCH_SIZE
from amazon_fmeval.eval_algorithms.eval_algorithm import EvalScore
from amazon_fmeval.eval_algorithms.util import (
    generate_model_predict_response_for_dataset,
    generate_prompt_column_for_dataset,
    category_wise_aggregation,
    dataset_aggregation,
    save_dataset,
    EvalOutputRecord,
    generate_output_dataset_path,
)
from amazon_fmeval.exceptions import EvalAlgorithmInternalError
from amazon_fmeval.util import camel_to_snake

MODEL_INPUT_COLUMN_NAME = "model_input"
MODEL_OUTPUT_COLUMN_NAME = "model_output"
MODEL_LOG_PROBABILITY_COLUMN_NAME = "model_log_probability"
PROMPT_COLUMN_NAME = "prompt_column"


def test_camel_to_snake():
    assert camel_to_snake("camel2_camel2_case") == "camel2_camel2_case"
    assert camel_to_snake("getHTTPResponseCode") == "get_http_response_code"
    assert camel_to_snake("HTTPResponseCodeXYZ") == "http_response_code_xyz"


class TestCaseGenerateModelOutput(NamedTuple):
    num_rows: int
    input_dataset: List[Dict[str, Any]]
    expected_dataset: List[Dict[str, Any]]
    model_log_probability: Optional[float]


# TODO: Change
def model_runner_return_value():
    return "output", 1.0


@pytest.mark.parametrize(
    "test_case",
    [
        TestCaseGenerateModelOutput(
            input_dataset=[
                {
                    "id": 1,
                    "model_input": "a",
                },
                {
                    "id": 2,
                    "model_input": "b",
                },
                {
                    "id": 3,
                    "model_input": "c",
                },
            ],
            num_rows=3,
            expected_dataset=[
                {"id": 1, "model_input": "a", "model_output": "output", "model_log_probability": 1.0},
                {"id": 2, "model_input": "b", "model_output": "output", "model_log_probability": 1.0},
                {"id": 3, "model_input": "c", "model_output": "output", "model_log_probability": 1.0},
            ],
            model_log_probability=1.0,
        ),
    ],
)
def test_generate_model_predict_response_for_dataset(test_case):
    """
    GIVEN the assumption that ModelRunner.predict() invokes the model and extract model response as intended
    WHEN generate_model_output_for_dataset is called
    THEN dataset with model output attached
    """
    # GIVEN
    mock_model_runner = Mock()
    mock_model_runner.predict.return_value = model_runner_return_value()
    dataset = ray.data.from_items(test_case.input_dataset)
    # WHEN
    returned_dataset = generate_model_predict_response_for_dataset(
        model=mock_model_runner,
        data=dataset,
        model_input_column_name=MODEL_INPUT_COLUMN_NAME,
        model_output_column_name=MODEL_OUTPUT_COLUMN_NAME,
        model_log_probability_column_name=MODEL_LOG_PROBABILITY_COLUMN_NAME,
    )
    # THEN
    assert returned_dataset.count() == test_case.num_rows
    assert sorted(returned_dataset.take(test_case.num_rows), key=lambda x: x["id"]) == test_case.expected_dataset


class TestCaseGeneratePromptColumn(NamedTuple):
    num_rows: int
    input_dataset: List[Dict[str, Any]]
    prompt_template: str
    expected_dataset: List[Dict[str, Any]]


@pytest.mark.parametrize(
    "test_case",
    [
        TestCaseGeneratePromptColumn(
            input_dataset=[
                {
                    "id": 1,
                    "model_input": "a",
                },
                {
                    "id": 2,
                    "model_input": "b",
                },
                {
                    "id": 3,
                    "model_input": "c",
                },
            ],
            num_rows=3,
            prompt_template="Summarise: $feature",
            expected_dataset=[
                {
                    "id": 1,
                    "model_input": "a",
                    "prompt_column": "Summarise: a",
                },
                {
                    "id": 2,
                    "model_input": "b",
                    "prompt_column": "Summarise: b",
                },
                {
                    "id": 3,
                    "model_input": "c",
                    "prompt_column": "Summarise: c",
                },
            ],
        ),
    ],
)
def test_generate_prompt_column_for_dataset(test_case):
    """
    GIVEN an input dataset and a prompt_template
    WHEN generate_prompt_column_for_dataset is called
    THEN dataset with prompt column added is returned
    """
    dataset = ray.data.from_items(test_case.input_dataset)
    returned_dataset = generate_prompt_column_for_dataset(
        prompt_template=test_case.prompt_template,
        data=dataset,
        model_input_column_name=MODEL_INPUT_COLUMN_NAME,
        prompt_column_name=PROMPT_COLUMN_NAME,
    )
    assert returned_dataset.count() == test_case.num_rows
    assert sorted(returned_dataset.take(test_case.num_rows), key=lambda x: x["id"]) == test_case.expected_dataset


def test_aggregate_dataset_invalid_agg():
    with pytest.raises(EvalAlgorithmInternalError, match="Aggregation method median is not supported"):
        dataset_aggregation(
            dataset=ray.data.from_pandas(pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])),
            score_column_name="a",
            agg_method="median",
        )


def test_category_wise_aggregate_invalid_agg():
    pandas_df = pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])
    category = pd.Series(["A", "A", "C", "B", "B", "C", "C", "A", "B", "C"])
    pandas_df[CATEGORY_COLUMN_NAME] = category
    with pytest.raises(EvalAlgorithmInternalError, match="Aggregation method median is not supported"):
        category_wise_aggregation(dataset=ray.data.from_pandas(pandas_df), score_column_name="a", agg_method="median")


def test_category_wise_aggregate():
    pandas_df = pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])
    category = pd.Series(["A", "A", "C", "B", "B", "C", "C", "A", "B", "C"])
    pandas_df[CATEGORY_COLUMN_NAME] = category
    category_aggregates = category_wise_aggregation(
        dataset=ray.data.from_pandas(pandas_df), score_column_name="a", agg_method="mean"
    )
    for row in category_aggregates.iter_rows():
        assert row[f"mean(a)"] == pandas_df.loc[pandas_df.category == row[CATEGORY_COLUMN_NAME]]["a"].mean()


def test_eval_output_record_str():
    """
    GIVEN an EvalOutputRecord
    WHEN its __str__ method is called
    THEN the correct string representation of EvalOutputRecord,
        where attributes are sorted according to their order
        in the class definition (but with "scores" coming last)
        is returned
    """
    record = EvalOutputRecord(
        model_output="output",
        scores=[EvalScore(name="rouge", value=0.5), EvalScore(name="bert", value=0.4)],
        model_input="input",
    )
    expected_record = OrderedDict(
        [
            ("model_input", "input"),
            ("model_output", "output"),
            ("scores", [{"name": "rouge", "value": 0.5}, {"name": "bert", "value": 0.4}]),
        ]
    )
    assert json.loads(str(record), object_pairs_hook=OrderedDict) == expected_record


def test_eval_output_record_from_row_success():
    """
    GIVEN a row with valid keys (i.e. column names)
    WHEN EvalOutputRecord.from_row is called
    THEN the correct EvalOutputRecord is returned
    """
    row = {MODEL_INPUT_COLUMN_NAME: "input", MODEL_OUTPUT_COLUMN_NAME: "output", "rouge": 0.42, "bert": 0.162}
    expected_record = EvalOutputRecord(
        model_input="input",
        model_output="output",
        scores=[EvalScore(name="rouge", value=0.42), EvalScore(name="bert", value=0.162)],
    )

    assert EvalOutputRecord.from_row(row, score_names=["rouge", "bert"]) == expected_record


@pytest.mark.parametrize("parent_dir_path", ["path/to/parent", "path/to/parent/"])
def test_generate_output_dataset_path(parent_dir_path):
    eval_name = "eval"
    dataset_name = "dataset"
    actual = generate_output_dataset_path(parent_dir_path, eval_name, dataset_name)
    assert actual == "path/to/parent/eval_dataset.jsonl"


@pytest.mark.parametrize("file_name", ["my_dataset.jsonl", "my_dataset"])
def test_save_dataset(tmp_path, file_name):
    """
    Given a Ray Dataset, a list of score names, and a local path
    WHEN save_dataset is called
    THEN a JSON Lines file that adheres to the correct schema gets
        written to the local path
    """
    # GIVEN
    ds_items = [
        {MODEL_INPUT_COLUMN_NAME: "hello", CATEGORY_COLUMN_NAME: "Age", "rouge": 0.5, "bert_score": 0.42},
        {MODEL_INPUT_COLUMN_NAME: "world", CATEGORY_COLUMN_NAME: "Gender", "rouge": 0.314, "bert_score": 0.271},
    ]
    dataset = ray.data.from_items(ds_items)
    score_names = ["rouge", "bert_score"]

    # WHEN
    full_path_to_file = str(tmp_path / file_name)
    save_dataset(dataset, score_names, full_path_to_file)

    # THEN
    expected_full_path_to_created_file = str(tmp_path / "my_dataset.jsonl")
    assert os.path.isfile(expected_full_path_to_created_file)
    with open(expected_full_path_to_created_file) as file_handle:
        json_objects = [json.loads(line, object_pairs_hook=OrderedDict) for line in file_handle.readlines()]
        assert json_objects  # if nothing gets written to the file, this test would trivially pass
        for json_obj in json_objects:
            # want to ensure ordering of keys is correct, so we use list instead of set
            assert list(json_obj.keys()) == [MODEL_INPUT_COLUMN_NAME, CATEGORY_COLUMN_NAME, "scores"]
            assert json_obj[MODEL_INPUT_COLUMN_NAME] in {"hello", "world"}

            if json_obj[MODEL_INPUT_COLUMN_NAME] == "hello":
                assert json_obj[CATEGORY_COLUMN_NAME] == "Age"
                assert json_obj["scores"] == [{"name": "rouge", "value": 0.5}, {"name": "bert_score", "value": 0.42}]

            if json_obj[MODEL_INPUT_COLUMN_NAME] == "world":
                assert json_obj[CATEGORY_COLUMN_NAME] == "Gender"
                assert json_obj["scores"] == [{"name": "rouge", "value": 0.314}, {"name": "bert_score", "value": 0.271}]


def test_save_dataset_many_rows(tmp_path):
    """
    GIVEN a dataset with more rows than EVAL_OUTPUT_RECORDS_BATCH_SIZE
    WHEN save_dataset is called
    THEN all rows in the dataset gets properly written to a file
    """
    # GIVEN
    ds_items = [
        {MODEL_INPUT_COLUMN_NAME: f"input_{i}", CATEGORY_COLUMN_NAME: f"category_{i}", "rouge": 0.5, "bert_score": 0.42}
        for i in range(EVAL_OUTPUT_RECORDS_BATCH_SIZE + 1)
    ]
    dataset = ray.data.from_items(ds_items)
    score_names = ["rouge", "bert_score"]

    # WHEN
    full_path_to_file = str(tmp_path / "test_dataset.jsonl")
    save_dataset(dataset, score_names, full_path_to_file)

    with open(full_path_to_file) as file_handle:
        json_objects = (json.loads(line, object_pairs_hook=OrderedDict) for line in file_handle.readlines())
        for i, json_obj in enumerate(json_objects):
            # want to ensure ordering of keys is correct, so we use list instead of set
            assert list(json_obj.keys()) == [MODEL_INPUT_COLUMN_NAME, CATEGORY_COLUMN_NAME, "scores"]
            assert json_obj[MODEL_INPUT_COLUMN_NAME] == f"input_{i}"
            assert json_obj[CATEGORY_COLUMN_NAME] == f"category_{i}"
