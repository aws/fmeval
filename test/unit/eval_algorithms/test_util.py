import json
import multiprocessing as mp
import os
from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple, Optional
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import ray
from ray.data import Dataset

from fmeval.constants import (
    DatasetColumns,
    EVAL_OUTPUT_RECORDS_BATCH_SIZE,
    PARALLELIZATION_FACTOR,
)
from fmeval.eval_algorithms import EvalAlgorithm, DATASET_CONFIGS, BOOLQ, TRIVIA_QA, NATURAL_QUESTIONS
from fmeval.eval_algorithms.eval_algorithm import EvalScore
from fmeval.eval_algorithms.util import (
    generate_model_predict_response_for_dataset,
    generate_prompt_column_for_dataset,
    category_wise_aggregation,
    dataset_aggregation,
    save_dataset,
    EvalOutputRecord,
    generate_output_dataset_path,
    generate_mean_delta_score,
    verify_model_determinism,
    get_dataset_configs,
)
from fmeval.exceptions import EvalAlgorithmInternalError
from fmeval.util import camel_to_snake, get_num_actors
from fmeval.eval_algorithms.util import get_bert_score

BERTSCORE_DUMMY_VALUE = (
    0.5  # we don't evaluate the real BERTScore inside unit tests because of runtime, so we hardcode a dummy value
)


def test_camel_to_snake():
    assert camel_to_snake("camel2_camel2_case") == "camel2_camel2_case"
    assert camel_to_snake("getHTTPResponseCode") == "get_http_response_code"
    assert camel_to_snake("HTTPResponseCodeXYZ") == "http_response_code_xyz"


def test_get_actors():
    os.environ[PARALLELIZATION_FACTOR] = "2"
    assert get_num_actors() == 2

    os.environ.pop(PARALLELIZATION_FACTOR)
    assert get_num_actors() == mp.cpu_count() - 1


def test_get_actors_invalid_value():
    os.environ[PARALLELIZATION_FACTOR] = "hello"
    assert get_num_actors() == mp.cpu_count() - 1


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
                    DatasetColumns.MODEL_INPUT.value.name: "a",
                },
                {
                    "id": 2,
                    DatasetColumns.MODEL_INPUT.value.name: "b",
                },
                {
                    "id": 3,
                    DatasetColumns.MODEL_INPUT.value.name: "c",
                },
            ],
            num_rows=3,
            expected_dataset=[
                {
                    "id": 1,
                    DatasetColumns.MODEL_INPUT.value.name: "a",
                    "model_output": "output",
                    "model_log_probability": 1.0,
                },
                {
                    "id": 2,
                    DatasetColumns.MODEL_INPUT.value.name: "b",
                    "model_output": "output",
                    "model_log_probability": 1.0,
                },
                {
                    "id": 3,
                    DatasetColumns.MODEL_INPUT.value.name: "c",
                    "model_output": "output",
                    "model_log_probability": 1.0,
                },
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
        model_input_column_name=DatasetColumns.MODEL_INPUT.value.name,
        model_output_column_name=DatasetColumns.MODEL_OUTPUT.value.name,
        model_log_probability_column_name=DatasetColumns.MODEL_LOG_PROBABILITY.value.name,
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
                    DatasetColumns.MODEL_INPUT.value.name: "a",
                },
                {
                    "id": 2,
                    DatasetColumns.MODEL_INPUT.value.name: "b",
                },
                {
                    "id": 3,
                    DatasetColumns.MODEL_INPUT.value.name: "c",
                },
            ],
            num_rows=3,
            prompt_template="Summarise: $model_input",
            expected_dataset=[
                {
                    "id": 1,
                    DatasetColumns.MODEL_INPUT.value.name: "a",
                    DatasetColumns.PROMPT.value.name: "Summarise: a",
                },
                {
                    "id": 2,
                    DatasetColumns.MODEL_INPUT.value.name: "b",
                    DatasetColumns.PROMPT.value.name: "Summarise: b",
                },
                {
                    "id": 3,
                    DatasetColumns.MODEL_INPUT.value.name: "c",
                    DatasetColumns.PROMPT.value.name: "Summarise: c",
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
        model_input_column_name=DatasetColumns.MODEL_INPUT.value.name,
        prompt_column_name=DatasetColumns.PROMPT.value.name,
    )
    assert returned_dataset.count() == test_case.num_rows
    assert sorted(returned_dataset.take(test_case.num_rows), key=lambda x: x["id"]) == test_case.expected_dataset


@patch("ray.data.ActorPoolStrategy")
@patch("ray.data.Dataset")
def test_num_actors_in_generate_prompt_column_for_dataset(dataset, actor_pool_strategy):
    num_actors = mp.cpu_count() - 2
    os.environ[PARALLELIZATION_FACTOR] = str(num_actors)
    mock_model_runner = Mock()
    mock_model_runner.predict.return_value = model_runner_return_value()
    dataset.map.return_value = dataset
    generate_model_predict_response_for_dataset(mock_model_runner, dataset, DatasetColumns.MODEL_INPUT.value.name)
    actor_pool_strategy.assert_called_with(size=num_actors)

    os.environ.pop(PARALLELIZATION_FACTOR)
    generate_model_predict_response_for_dataset(mock_model_runner, dataset, DatasetColumns.MODEL_INPUT.value.name)
    actor_pool_strategy.assert_called_with(size=mp.cpu_count() - 1)


@patch("ray.data.ActorPoolStrategy")
@patch("ray.data.Dataset")
def test_num_actors_in_generate_prompt_column_for_dataset_bad_value(dataset, actor_pool_strategy):
    os.environ[PARALLELIZATION_FACTOR] = str("hello")
    mock_model_runner = Mock()
    mock_model_runner.predict.return_value = model_runner_return_value()
    dataset.map.return_value = dataset
    generate_model_predict_response_for_dataset(mock_model_runner, dataset, DatasetColumns.MODEL_INPUT.value.name)
    actor_pool_strategy.assert_called_with(size=mp.cpu_count() - 1)


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
    pandas_df[DatasetColumns.CATEGORY.value.name] = category
    with pytest.raises(EvalAlgorithmInternalError, match="Aggregation method median is not supported"):
        category_wise_aggregation(dataset=ray.data.from_pandas(pandas_df), score_column_name="a", agg_method="median")


def test_category_wise_aggregate():
    pandas_df = pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])
    category = pd.Series(["A", "A", "C", "B", "B", "C", "C", "A", "B", "C"])
    pandas_df[DatasetColumns.CATEGORY.value.name] = category
    category_aggregates = category_wise_aggregation(
        dataset=ray.data.from_pandas(pandas_df), score_column_name="a", agg_method="mean"
    )
    for row in category_aggregates.iter_rows():
        assert (
            row[f"mean(a)"] == pandas_df.loc[pandas_df.category == row[DatasetColumns.CATEGORY.value.name]]["a"].mean()
        )


def test_eval_output_record_post_init():
    """
    GIVEN a `non_score_column` argument containing column names that don't belong to constants.COLUMN_NAMES
    WHEN an EvalOutputRecord is created
    THEN the EvalOutputRecord's __post_init__ method will raise an exception
    """
    invalid_col = "invalid_col"
    with pytest.raises(
        EvalAlgorithmInternalError,
        match=f"Attempting to initialize an EvalOutputRecord with invalid non-score column {invalid_col}.",
    ):
        EvalOutputRecord(
            scores=[EvalScore(name="score1", value=0.162)],
            dataset_columns={
                DatasetColumns.MODEL_INPUT.value.name: "my input",
                DatasetColumns.MODEL_OUTPUT.value.name: "my output",
                invalid_col: "blah",
            },
        )


def test_eval_output_record_str():
    """
    GIVEN an EvalOutputRecord
    WHEN its __str__ method is called
    THEN the correct string representation of EvalOutputRecord is returned,
        where the order of the non-score columns matches their relative ordering
        in constants.COLUMN_NAMES, rather than their order in the `dataset_columns`
        list, and "scores" comes at the end.
    """
    record = EvalOutputRecord(
        scores=[EvalScore(name="rouge", value=0.5), EvalScore(name="bert", value=0.4)],
        dataset_columns={
            DatasetColumns.MODEL_OUTPUT.value.name: "output",
            DatasetColumns.MODEL_INPUT.value.name: "input",
        },
    )
    expected_record = OrderedDict(
        [
            (DatasetColumns.MODEL_INPUT.value.name, "input"),
            (DatasetColumns.MODEL_OUTPUT.value.name, "output"),
            ("scores", [{"name": "rouge", "value": 0.5}, {"name": "bert", "value": 0.4}]),
        ]
    )
    assert json.loads(str(record), object_pairs_hook=OrderedDict) == expected_record


def test_eval_output_record_from_row():
    """
    GIVEN a row with both valid and invalid column names
        (a column name is valid if it is a member of the set constants.COLUMN_NAMES)
    WHEN EvalOutputRecord.from_row is called
    THEN an EvalOutputRecord with only the valid column names is returned
    """
    row = {
        "rouge": 0.42,
        DatasetColumns.MODEL_OUTPUT.value.name: "output",
        "bert": 0.162,
        "invalid_col_1": "hello",
        DatasetColumns.MODEL_INPUT.value.name: "input",
        "invalid_col_2": "world",
    }
    expected_record = EvalOutputRecord(
        scores=[EvalScore(name="rouge", value=0.42), EvalScore(name="bert", value=0.162)],
        dataset_columns={
            DatasetColumns.MODEL_INPUT.value.name: "input",
            DatasetColumns.MODEL_OUTPUT.value.name: "output",
        },
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
    unused_column_name = "unused"
    # GIVEN
    ds_items = [
        {
            DatasetColumns.MODEL_INPUT.value.name: "hello",
            DatasetColumns.CATEGORY.value.name: "Age",
            unused_column_name: "Arch",
            "rouge": 0.5,
            "bert_score": 0.42,
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "world",
            DatasetColumns.CATEGORY.value.name: "Gender",
            unused_column_name: "btw",
            "rouge": 0.314,
            "bert_score": 0.271,
        },
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
            assert list(json_obj.keys()) == [
                DatasetColumns.MODEL_INPUT.value.name,
                DatasetColumns.CATEGORY.value.name,
                "scores",
            ]
            assert json_obj[DatasetColumns.MODEL_INPUT.value.name] in {"hello", "world"}

            if json_obj[DatasetColumns.MODEL_INPUT.value.name] == "hello":
                assert json_obj[DatasetColumns.CATEGORY.value.name] == "Age"
                assert json_obj["scores"] == [{"name": "rouge", "value": 0.5}, {"name": "bert_score", "value": 0.42}]

            if json_obj[DatasetColumns.MODEL_INPUT.value.name] == "world":
                assert json_obj[DatasetColumns.CATEGORY.value.name] == "Gender"
                assert json_obj["scores"] == [{"name": "rouge", "value": 0.314}, {"name": "bert_score", "value": 0.271}]


def test_save_dataset_many_rows(tmp_path):
    """
    GIVEN a dataset with more rows than EVAL_OUTPUT_RECORDS_BATCH_SIZE
    WHEN save_dataset is called
    THEN all rows in the dataset gets properly written to a file
    """
    # GIVEN
    ds_items = [
        {
            DatasetColumns.MODEL_INPUT.value.name: f"input_{i}",
            DatasetColumns.CATEGORY.value.name: f"category_{i}",
            "rouge": 0.5,
            "bert_score": 0.42,
        }
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
            assert list(json_obj.keys()) == [
                DatasetColumns.MODEL_INPUT.value.name,
                DatasetColumns.CATEGORY.value.name,
                "scores",
            ]
            assert json_obj[DatasetColumns.MODEL_INPUT.value.name] == f"input_{i}"
            assert json_obj[DatasetColumns.CATEGORY.value.name] == f"category_{i}"


class TestCaseGenerateMeanDeltaScore(NamedTuple):
    original_score: EvalScore
    perturbed_input_scores: List[EvalScore]
    expected_response: float


@pytest.mark.parametrize(
    "test_case",
    [
        TestCaseGenerateMeanDeltaScore(
            original_score=EvalScore(name="my_score", value=2.0),
            perturbed_input_scores=[EvalScore(name="my_score", value=0.5), EvalScore(name="my_score", value=1.0)],
            expected_response=1.25,
        ),
        TestCaseGenerateMeanDeltaScore(
            original_score=EvalScore(name="my_score", value=2.0),
            perturbed_input_scores=[
                EvalScore(name="my_score", value=0.5),
                EvalScore(name="my_score", value=1.0),
                EvalScore(name="my_score", value=2.5),
                EvalScore(name="my_score", value=1.5),
            ],
            expected_response=0.875,
        ),
    ],
)
def test_generate_mean_delta_score(test_case):
    """
    GIVEN correct inputs
    WHEN generate_mean_delta_score util method is called
    THEN correct mean delta is returned
    """
    assert test_case.expected_response == generate_mean_delta_score(
        test_case.original_score, test_case.perturbed_input_scores
    )


class TestCaseVerifyModelDeterminism(NamedTuple):
    dataset: Dataset
    prompt_column_name: str
    expect_num_predict_calls: int
    predict_result: List
    expect_response: bool


@pytest.mark.parametrize(
    "test_case",
    [
        # dataset fewer than 5 rows
        TestCaseVerifyModelDeterminism(
            dataset=ray.data.from_items(
                [
                    {
                        DatasetColumns.MODEL_INPUT.value.name: "Summarize: Cake is so delicious, I really like cake. I want to open a bakery when I grow up.",
                        "another prompt column": "Cake is so delicious, I really like cake. I want to open a bakery when I grow up.",
                        DatasetColumns.TARGET_OUTPUT.value.name: "I like cake.",
                    },
                    {
                        DatasetColumns.MODEL_INPUT.value.name: "Summarize: The art metropolis of Berlin inspires locals and visitors with its famous "
                        "museum landscape and numerous UNESCO World Heritage sites."
                        " It is also an international exhibition venue. "
                        "You will find a selection of current and upcoming exhibitions here.",
                        "another prompt column": "Summarize: The art metropolis of Berlin inspires locals and visitors with its famous "
                        "museum landscape and numerous UNESCO World Heritage sites."
                        " It is also an international exhibition venue. "
                        "You will find a selection of current and upcoming exhibitions here.",
                        DatasetColumns.TARGET_OUTPUT.value.name: "Berlin: an art metropolis.",
                    },
                ]
            ),
            predict_result=[("model output 1",), ("model output 1",), ("model output 2",), ("model output 2",)],
            expect_num_predict_calls=4,
            prompt_column_name="another prompt column",
            expect_response=True,
        ),
        # only test first 5 rows
        TestCaseVerifyModelDeterminism(
            dataset=ray.data.from_items(
                [
                    {
                        DatasetColumns.PROMPT.value.name: "Answer: What is the capital of Italy?",
                        DatasetColumns.TARGET_OUTPUT.value.name: "Rome",
                    },
                    {
                        DatasetColumns.PROMPT.value.name: "Answer: When did Argentina win the FIFA World Cup?",
                        DatasetColumns.TARGET_OUTPUT.value.name: "1978<OR>1986<OR>2022.",
                    },
                    {
                        DatasetColumns.PROMPT.value.name: "Answer: What is the capital of England?",
                        DatasetColumns.TARGET_OUTPUT.value.name: "London",
                    },
                    {
                        DatasetColumns.PROMPT.value.name: "Answer: What is the color of blood?",
                        DatasetColumns.TARGET_OUTPUT.value.name: "Red",
                    },
                    {
                        DatasetColumns.PROMPT.value.name: "Answer: Who directed Pulp Fiction?",
                        DatasetColumns.TARGET_OUTPUT.value.name: "Quentin Tarantino",
                    },
                    {
                        DatasetColumns.PROMPT.value.name: "Answer: When did Argentina win the FIFA World Cup?",
                        DatasetColumns.TARGET_OUTPUT.value.name: "1978<OR>1986<OR>2022",
                    },
                ]
            ),
            predict_result=[
                ("model output 1",),
                ("model output 1",),
                ("model output 2",),
                ("model output 2",),
                ("model output 2",),
                ("model output 2",),
                ("model output 4",),
                ("model output 4",),
                ("model output 5",),
                ("model output 5",),
            ],
            expect_num_predict_calls=10,
            prompt_column_name=DatasetColumns.PROMPT.value.name,
            expect_response=True,
        ),
        # dataset fewer than 5 rows
        TestCaseVerifyModelDeterminism(
            dataset=ray.data.from_items(
                [
                    {
                        DatasetColumns.PROMPT.value.name: "Answer: What is the capital of Italy?",
                        DatasetColumns.TARGET_OUTPUT.value.name: "Rome",
                    },
                    {
                        DatasetColumns.PROMPT.value.name: "Answer: When did Argentina win the FIFA World Cup?",
                        DatasetColumns.TARGET_OUTPUT.value.name: "1978<OR>1986<OR>2022.",
                    },
                ]
            ),
            predict_result=[
                ("model output 1",),
                ("different model output 1",),
                ("model output 2",),
                ("different model output 2",),
            ],
            expect_num_predict_calls=2,
            prompt_column_name=DatasetColumns.PROMPT.value.name,
            expect_response=False,
        ),
    ],
)
def test_verify_model_determinism(test_case):
    """
    GIVEN a deterministic model and other inputs
    WHEN verify_model_determinism is called
    THEN no Exception raised
    """
    model = MagicMock()
    model.predict.side_effect = test_case.predict_result
    result = verify_model_determinism(
        model=model, dataset=test_case.dataset, prompt_column_name=test_case.prompt_column_name
    )
    assert model.predict.call_count == test_case.expect_num_predict_calls
    assert result == test_case.expect_response


@pytest.mark.parametrize(
    "target_output,model_output",
    [("I like cake.", "I like cake."), ("Berlin: Art, Heritage, Exhibitions Hub.", "Berlin: an art metropolis.")],
)
@patch("fmeval.eval_algorithms.util.ray.get")
def test_get_bert_score(mock_ray_get, target_output, model_output):
    mock_ray_get.return_value = BERTSCORE_DUMMY_VALUE
    assert BERTSCORE_DUMMY_VALUE == get_bert_score(target_output, model_output, helper_model=MagicMock())


def test_get_dataset_configs():
    custom_config = Mock()
    configs = get_dataset_configs(custom_config, "blah")
    assert configs == [custom_config]

    # WLOG, use QA accuracy as the test case
    configs = get_dataset_configs(None, EvalAlgorithm.QA_ACCURACY.value)
    assert configs == [DATASET_CONFIGS[BOOLQ], DATASET_CONFIGS[TRIVIA_QA], DATASET_CONFIGS[NATURAL_QUESTIONS]]
