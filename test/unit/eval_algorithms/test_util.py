import json
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
import pytest
import ray

from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from unittest.mock import Mock, call, patch
from ray.data import Dataset

from fmeval.constants import (
    DatasetColumns,
    EVAL_OUTPUT_RECORDS_BATCH_SIZE,
    PARALLELIZATION_FACTOR,
    MEAN,
    NUM_ROWS_DETERMINISTIC,
)
from fmeval.eval_algorithms import (
    EvalAlgorithm,
    DATASET_CONFIGS,
    BOOLQ,
    TRIVIA_QA,
    NATURAL_QUESTIONS,
    CROWS_PAIRS,
    TREX,
    GIGAWORD,
    GOV_REPORT,
    BOLD,
    WIKITEXT2,
    WOMENS_CLOTHING_ECOMMERCE_REVIEWS,
    REAL_TOXICITY_PROMPTS,
    REAL_TOXICITY_PROMPTS_CHALLENGING,
    EvalOutput,
    CategoryScore,
)
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
    aggregate_evaluation_scores,
    create_model_invocation_pipeline,
    evaluate_dataset,
)
from fmeval.exceptions import EvalAlgorithmInternalError, EvalAlgorithmClientError
from fmeval.transforms.common import GeneratePrompt, GetModelOutputs
from fmeval.util import camel_to_snake, get_num_actors

BERTSCORE_DUMMY_VALUE = (
    0.5  # we don't evaluate the real BERTScore inside unit tests because of runtime, so we hardcode a dummy value
)

SCORE_1 = "score_1"
SCORE_2 = "score_2"

DATASET_WITH_SCORES = ray.data.from_items(
    [
        {
            DatasetColumns.MODEL_INPUT.value.name: "What is the capital of England?",
            DatasetColumns.CATEGORY.value.name: "dummy_category_1",
            DatasetColumns.PROMPT.value.name: "unused since we mock the model output",
            DatasetColumns.MODEL_OUTPUT.value.name: "Some model output.",
            SCORE_1: 0.162,
            SCORE_2: 0.189,
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "What is the capital of England?",
            DatasetColumns.CATEGORY.value.name: "dummy_category_2",
            DatasetColumns.PROMPT.value.name: "unused since we mock the model output",
            DatasetColumns.MODEL_OUTPUT.value.name: "Some model output.",
            SCORE_1: 0.126,
            SCORE_2: 0.127,
        },
    ]
)

DATASET_WITH_SCORES_NO_CATEGORY = DATASET_WITH_SCORES.drop_columns(DatasetColumns.CATEGORY.value.name)

DATASET_WITHOUT_SCORES = DATASET_WITH_SCORES.drop_columns(
    cols=[
        DatasetColumns.PROMPT.value.name,
        DatasetColumns.MODEL_OUTPUT.value.name,
        SCORE_1,
        SCORE_2,
    ]
)

DATASET_WITH_MODEL_OUTPUT = DATASET_WITHOUT_SCORES.drop_columns(cols=[SCORE_1, SCORE_2])

DATASET = DATASET_WITH_MODEL_OUTPUT.drop_columns(
    cols=[DatasetColumns.MODEL_OUTPUT.value.name, DatasetColumns.PROMPT.value.name]
)

DATASET_NO_CATEGORY = DATASET.drop_columns(cols=DatasetColumns.CATEGORY.value.name)

DATASET_SCORES = [
    EvalScore(name=SCORE_1, value=(0.162 + 0.126) / 2),
    EvalScore(name=SCORE_2, value=(0.189 + 0.127) / 2),
]

CATEGORY_SCORES = [
    CategoryScore(
        name="dummy_category_1",
        scores=[
            EvalScore(name=SCORE_1, value=0.162),
            EvalScore(name=SCORE_2, value=0.189),
        ],
    ),
    CategoryScore(
        name="dummy_category_2",
        scores=[
            EvalScore(name=SCORE_1, value=0.126),
            EvalScore(name=SCORE_2, value=0.127),
        ],
    ),
]


# DATASET_WITH_MODEL_OUTPUT_NO_CATEGORY = DATASET_WITH_MODEL_OUTPUT.drop_columns(cols=DatasetColumns.CATEGORY.value.name)


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
    dataset: List[Dict]
    predict_responses: List[Tuple]
    expected_num_predict_calls: int
    expected_result: bool


@pytest.mark.parametrize(
    "test_case",
    [
        # Dataset contains > NUM_ROWS_DETERMINISTIC rows, model is deterministic
        TestCaseVerifyModelDeterminism(
            dataset=[
                {
                    DatasetColumns.MODEL_INPUT.value.name: "What is the capital of Italy?",
                    DatasetColumns.TARGET_OUTPUT.value.name: "Rome",
                },
                {
                    DatasetColumns.MODEL_INPUT.value.name: "When did Argentina win the FIFA World Cup?",
                    DatasetColumns.TARGET_OUTPUT.value.name: "1978<OR>1986<OR>2022.",
                },
                {
                    DatasetColumns.MODEL_INPUT.value.name: "What is the capital of England?",
                    DatasetColumns.TARGET_OUTPUT.value.name: "London",
                },
                {
                    DatasetColumns.MODEL_INPUT.value.name: "What is the color of blood?",
                    DatasetColumns.TARGET_OUTPUT.value.name: "Red",
                },
                {
                    DatasetColumns.MODEL_INPUT.value.name: "Who directed Pulp Fiction?",
                    DatasetColumns.TARGET_OUTPUT.value.name: "Quentin Tarantino",
                },
                {
                    DatasetColumns.MODEL_INPUT.value.name: "When did Argentina win the FIFA World Cup?",
                    DatasetColumns.TARGET_OUTPUT.value.name: "1978<OR>1986<OR>2022",
                },
            ],
            predict_responses=[
                ("model output 1", None),
                ("model output 1", None),
                ("model output 2", None),
                ("model output 2", None),
                ("model output 2", None),
                ("model output 2", None),
                ("model output 4", None),
                ("model output 4", None),
                ("model output 5", None),
                ("model output 5", None),
            ],
            expected_num_predict_calls=10,
            expected_result=True,
        ),
        # Dataset contains fewer than NUM_ROWS_DETERMINISTIC rows, model is deterministic
        TestCaseVerifyModelDeterminism(
            dataset=[
                {
                    DatasetColumns.MODEL_INPUT.value.name: "What is the capital of Italy?",
                    DatasetColumns.TARGET_OUTPUT.value.name: "Rome",
                },
                {
                    DatasetColumns.MODEL_INPUT.value.name: "When did Argentina win the FIFA World Cup?",
                    DatasetColumns.TARGET_OUTPUT.value.name: "1978<OR>1986<OR>2022.",
                },
            ],
            predict_responses=[
                ("model output 1", None),
                ("model output 1", None),
                ("model output 2", None),
                ("model output 2", None),
            ],
            expected_num_predict_calls=4,
            expected_result=True,
        ),
        # Dataset contains fewer than NUM_ROWS_DETERMINISTIC rows, model is not deterministic
        TestCaseVerifyModelDeterminism(
            dataset=[
                {
                    DatasetColumns.MODEL_INPUT.value.name: "What is the capital of Italy?",
                    DatasetColumns.TARGET_OUTPUT.value.name: "Rome",
                },
                {
                    DatasetColumns.MODEL_INPUT.value.name: "When did Argentina win the FIFA World Cup?",
                    DatasetColumns.TARGET_OUTPUT.value.name: "1978<OR>1986<OR>2022.",
                },
            ],
            predict_responses=[
                ("model output 1",),
                ("different model output 1",),
            ],
            expected_num_predict_calls=2,
            expected_result=False,
        ),
    ],
)
def test_verify_model_determinism(test_case):
    """
    GIVEN a model.
    WHEN verify_model_determinism is called.
    THEN the correct result is returned.
    """
    model = Mock()
    model.predict.side_effect = test_case.predict_responses
    result = verify_model_determinism(
        model=model,
        dataset=ray.data.from_items(test_case.dataset),
        prompt_template="Answer: $model_input",
        model_input_column_name=DatasetColumns.MODEL_INPUT.value.name,
    )
    assert model.predict.call_count == test_case.expected_num_predict_calls
    assert result == test_case.expected_result
    if test_case.expected_result:
        expected_calls = []
        for row in test_case.dataset[:NUM_ROWS_DETERMINISTIC]:
            expected_calls += [call(f"Answer: {row[DatasetColumns.MODEL_INPUT.value.name]}")] * 2
        model.predict.assert_has_calls(expected_calls)


def test_get_dataset_configs_custom_dataset():
    """
    GIVEN a custom dataset config.
    WHEN get_dataset_configs is called.
    THEN a single-element list with said dataset config is returned.
    """
    custom_config = Mock()
    configs = get_dataset_configs(custom_config, "blah")
    assert configs == [custom_config]


class TestCaseBuiltinDatasetConfigs(NamedTuple):
    eval_name: str
    dataset_names: List[str]


@pytest.mark.parametrize(
    "eval_name, dataset_names",
    [
        TestCaseBuiltinDatasetConfigs(
            eval_name=EvalAlgorithm.FACTUAL_KNOWLEDGE.value,
            dataset_names=[TREX],
        ),
        TestCaseBuiltinDatasetConfigs(
            eval_name=EvalAlgorithm.QA_ACCURACY.value,
            dataset_names=[BOOLQ, TRIVIA_QA, NATURAL_QUESTIONS],
        ),
        TestCaseBuiltinDatasetConfigs(
            eval_name=EvalAlgorithm.QA_ACCURACY_SEMANTIC_ROBUSTNESS.value,
            dataset_names=[BOOLQ, TRIVIA_QA, NATURAL_QUESTIONS],
        ),
        TestCaseBuiltinDatasetConfigs(eval_name=EvalAlgorithm.PROMPT_STEREOTYPING.value, dataset_names=[CROWS_PAIRS]),
        TestCaseBuiltinDatasetConfigs(
            eval_name=EvalAlgorithm.SUMMARIZATION_ACCURACY.value,
            dataset_names=[GIGAWORD, GOV_REPORT],
        ),
        TestCaseBuiltinDatasetConfigs(
            eval_name=EvalAlgorithm.GENERAL_SEMANTIC_ROBUSTNESS.value,
            dataset_names=[BOLD, TREX, WIKITEXT2],
        ),
        TestCaseBuiltinDatasetConfigs(
            eval_name=EvalAlgorithm.CLASSIFICATION_ACCURACY.value,
            dataset_names=[WOMENS_CLOTHING_ECOMMERCE_REVIEWS],
        ),
        TestCaseBuiltinDatasetConfigs(
            eval_name=EvalAlgorithm.CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS.value,
            dataset_names=[WOMENS_CLOTHING_ECOMMERCE_REVIEWS],
        ),
        TestCaseBuiltinDatasetConfigs(
            eval_name=EvalAlgorithm.SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS.value,
            dataset_names=[GIGAWORD, GOV_REPORT],
        ),
        TestCaseBuiltinDatasetConfigs(
            eval_name=EvalAlgorithm.TOXICITY.value,
            dataset_names=[BOLD, REAL_TOXICITY_PROMPTS, REAL_TOXICITY_PROMPTS_CHALLENGING],
        ),
        TestCaseBuiltinDatasetConfigs(
            eval_name=EvalAlgorithm.QA_TOXICITY.value,
            dataset_names=[BOOLQ, TRIVIA_QA, NATURAL_QUESTIONS],
        ),
        TestCaseBuiltinDatasetConfigs(
            eval_name=EvalAlgorithm.SUMMARIZATION_TOXICITY.value,
            dataset_names=[GIGAWORD, GOV_REPORT],
        ),
    ],
)
def test_get_dataset_configs_builtin_dataset(eval_name, dataset_names):
    """
    GIVEN an input argument of None and an evaluation algorithm name.
    WHEN get_dataset_configs is called.
    THEN a list with all builtin dataset configs corresponding to the
        evaluation algorithm is returned.
    """
    configs = get_dataset_configs(None, eval_name)
    assert configs == [DATASET_CONFIGS[dataset_name] for dataset_name in dataset_names]


class TestCaseAggregate(NamedTuple):
    dataset: Dataset
    expected_dataset_scores: List[EvalScore]
    expected_category_scores: Optional[List[CategoryScore]]


@pytest.mark.parametrize(
    "dataset, expected_dataset_scores, expected_category_scores",
    [
        TestCaseAggregate(
            dataset=DATASET_WITH_SCORES,
            expected_dataset_scores=DATASET_SCORES,
            expected_category_scores=CATEGORY_SCORES,
        ),
        TestCaseAggregate(
            dataset=DATASET_WITH_SCORES_NO_CATEGORY,
            expected_dataset_scores=DATASET_SCORES,
            expected_category_scores=None,
        ),
    ],
)
def test_aggregate_evaluation_scores(dataset, expected_dataset_scores, expected_category_scores):
    """
    GIVEN a Ray dataset with scores (and potentially category columns).
    WHEN aggregate_evaluation_scores is called.
    THEN the correct outputs are returned.
    """
    dataset_scores, category_scores = aggregate_evaluation_scores(
        dataset,
        [SCORE_1, SCORE_2],
        MEAN,
    )
    assert dataset_scores == expected_dataset_scores
    assert category_scores == expected_category_scores


def test_model_invocation_pipeline():
    """
    GIVEN a ModelRunner and prompt template.
    WHEN create_model_invocation_pipeline is called.
    THEN the correct TransformPipeline is returned.
    """
    model = Mock()
    pipeline = create_model_invocation_pipeline(model, "Do something with $model_input")
    gen_prompt, get_outputs = pipeline.transforms[0], pipeline.transforms[1]
    assert isinstance(gen_prompt, GeneratePrompt)
    assert gen_prompt.input_keys == [DatasetColumns.MODEL_INPUT.value.name]
    assert gen_prompt.output_keys == [DatasetColumns.PROMPT.value.name]
    assert gen_prompt.prompt_template == "Do something with $model_input"
    assert isinstance(get_outputs, GetModelOutputs)
    assert get_outputs.input_keys == [DatasetColumns.PROMPT.value.name]
    assert get_outputs.output_keys == [DatasetColumns.MODEL_OUTPUT.value.name]
    assert get_outputs.model_runner == model


class TestCaseEvaluateDataset(NamedTuple):
    user_provided_prompt_template: Optional[str]
    dataset_prompt_template: Optional[str]
    save: bool
    agg_method: str = MEAN


@pytest.mark.parametrize(
    "test_case",
    [
        TestCaseEvaluateDataset(
            user_provided_prompt_template=None,
            dataset_prompt_template="$model_input",
            save=True,
        ),
        TestCaseEvaluateDataset(
            user_provided_prompt_template="Do something with $model_input",
            dataset_prompt_template="Do something with $model_input",
            save=False,
        ),
    ],
)
@patch("fmeval.eval_algorithms.util.generate_output_dataset_path")
@patch("fmeval.eval_algorithms.util.save_dataset")
@patch("fmeval.eval_algorithms.util.aggregate_evaluation_scores")
@patch("fmeval.eval_algorithms.util.TransformPipeline")
@patch("fmeval.eval_algorithms.util.create_model_invocation_pipeline")
def test_evaluate_dataset_with_model(
    mock_create_invocation_pipeline,
    mock_transform_pipeline_cls,
    mock_aggregate,
    mock_save,
    mock_generate_output_path,
    test_case,
):
    """
    GIVEN valid arguments and a `model` argument that is not None.
    WHEN `evaluate_dataset` is called.
    THEN a model invocation pipeline (created using `model`) is prepended
        to the input pipeline and the correct EvalOutput is returned.
    """
    mock_create_invocation_pipeline.return_value = Mock()
    # The pipeline that has the GeneratePrompt and GetModelOutputs Transforms prepended.
    final_pipeline = Mock()
    mock_transform_pipeline_cls.return_value = final_pipeline
    final_pipeline.execute = Mock()

    mock_aggregate.return_value = DATASET_SCORES, CATEGORY_SCORES
    mock_generate_output_path.return_value = "path/to/output/dataset"

    input_dataset = Mock()
    # So that validate_dataset does not error
    input_dataset.columns = Mock(return_value=[DatasetColumns.MODEL_INPUT.value.name])
    input_pipeline = Mock()
    model_runner = Mock()

    eval_output = evaluate_dataset(
        dataset=input_dataset,
        pipeline=input_pipeline,
        dataset_name="my_dataset",
        eval_name="MyEvalAlgo",
        metric_names=[SCORE_1, SCORE_2],
        eval_results_path="/path/to/eval_results",
        model=model_runner,
        prompt_template=test_case.user_provided_prompt_template,
        save=test_case.save,
    )

    mock_create_invocation_pipeline.assert_called_once_with(model_runner, test_case.dataset_prompt_template)
    mock_transform_pipeline_cls.assert_called_once_with([mock_create_invocation_pipeline.return_value, input_pipeline])
    final_pipeline.execute.assert_called_once_with(input_dataset)
    mock_aggregate.assert_called_once_with(
        final_pipeline.execute.return_value,
        [SCORE_1, SCORE_2],
        agg_method=test_case.agg_method,
    )
    mock_generate_output_path.assert_called_once_with(
        path_to_parent_dir="/path/to/eval_results",
        eval_name="MyEvalAlgo",
        dataset_name="my_dataset",
    )
    assert eval_output == EvalOutput(
        eval_name="MyEvalAlgo",
        dataset_name="my_dataset",
        prompt_template=test_case.dataset_prompt_template,
        dataset_scores=DATASET_SCORES,
        category_scores=CATEGORY_SCORES,
        output_path="path/to/output/dataset",
    )
    if test_case.save:
        mock_save.assert_called_once_with(
            dataset=final_pipeline.execute.return_value,
            score_names=[SCORE_1, SCORE_2],
            path="path/to/output/dataset",
        )
    else:
        mock_save.assert_not_called()


@pytest.mark.parametrize(
    "test_case",
    [
        TestCaseEvaluateDataset(
            user_provided_prompt_template=None,
            dataset_prompt_template=None,
            save=True,
        ),
    ],
)
@patch("fmeval.eval_algorithms.util.generate_output_dataset_path")
@patch("fmeval.eval_algorithms.util.save_dataset")
@patch("fmeval.eval_algorithms.util.aggregate_evaluation_scores")
@patch("fmeval.eval_algorithms.util.TransformPipeline")
@patch("fmeval.eval_algorithms.util.create_model_invocation_pipeline")
def test_evaluate_dataset_no_model(
    mock_create_invocation_pipeline,
    mock_transform_pipeline_cls,
    mock_aggregate,
    mock_save,
    mock_generate_output_path,
    test_case,
):
    """
    GIVEN valid arguments and a `model` argument that is not None.
    WHEN `evaluate_dataset` is called.
    THEN the pipeline that gets executed is the input pipeline (i.e.
        no model invocation transforms are prepended)
        and the correct EvalOutput is returned.
    """
    mock_aggregate.return_value = DATASET_SCORES, CATEGORY_SCORES
    mock_generate_output_path.return_value = "path/to/output/dataset"

    input_dataset = Mock()
    input_dataset.columns = Mock(return_value=[DatasetColumns.MODEL_OUTPUT.value.name])
    input_pipeline = Mock()

    eval_output = evaluate_dataset(
        dataset=input_dataset,
        pipeline=input_pipeline,
        dataset_name="my_dataset",
        eval_name="MyEvalAlgo",
        metric_names=[SCORE_1, SCORE_2],
        eval_results_path="/path/to/eval_results",
        model=None,
        prompt_template=None,
        save=test_case.save,
    )

    mock_create_invocation_pipeline.assert_not_called()
    mock_transform_pipeline_cls.assert_not_called()
    input_pipeline.execute.assert_called_once_with(input_dataset)
    mock_aggregate.assert_called_once_with(
        input_pipeline.execute.return_value,
        [SCORE_1, SCORE_2],
        agg_method=test_case.agg_method,
    )
    mock_generate_output_path.assert_called_once_with(
        path_to_parent_dir="/path/to/eval_results",
        eval_name="MyEvalAlgo",
        dataset_name="my_dataset",
    )
    assert eval_output == EvalOutput(
        eval_name="MyEvalAlgo",
        dataset_name="my_dataset",
        prompt_template=test_case.dataset_prompt_template,
        dataset_scores=DATASET_SCORES,
        category_scores=CATEGORY_SCORES,
        output_path="path/to/output/dataset",
    )
    if test_case.save:
        mock_save.assert_called_once_with(
            dataset=input_pipeline.execute.return_value,
            score_names=[SCORE_1, SCORE_2],
            path="path/to/output/dataset",
        )
    else:
        mock_save.assert_not_called()


def test_evaluate_dataset_with_model_no_model_input_column():
    """
    GIVEN a model and a dataset that does not contain a model input column.
    WHEN the `evaluate_dataset` function is called.
    THEN the correct exception is raised.
    """
    mock_dataset = Mock()
    mock_dataset.columns = Mock(return_value=[])
    err_msg = (
        "evaluate_dataset has been given a ModelRunner to obtain outputs from "
        "but the provided dataset does not contain a model input column."
    )
    with pytest.raises(EvalAlgorithmClientError, match=err_msg):
        evaluate_dataset(
            dataset=mock_dataset,
            pipeline=Mock(),
            dataset_name="my_dataset",
            eval_name="MyEvalAlgo",
            metric_names=[SCORE_1, SCORE_2],
            eval_results_path="/path/to/eval/results",
            model=Mock(),
            prompt_template=None,
        )


@patch("fmeval.eval_algorithms.util.aggregate_evaluation_scores")
@patch("fmeval.eval_algorithms.util.logging.Logger.warning")
def test_evaluate_dataset_with_prompt_template_without_model(
    mock_logger,
    mock_aggregate,
):
    """
    GIVEN invalid arguments: a non-Null prompt template, but no model.
    WHEN the `evaluate_dataset` function is called.
    THEN a warning is logged.
    """
    mock_aggregate.return_value = [EvalScore(name="a", value=1.0)], None
    prompt_template = "Do something with $model_input."
    dataset = Mock()
    dataset.columns = Mock(return_value=[DatasetColumns.MODEL_OUTPUT.value.name])

    evaluate_dataset(
        dataset=dataset,
        pipeline=Mock(),
        dataset_name="my_dataset",
        eval_name="MyEvalAlgo",
        metric_names=[SCORE_1, SCORE_2],
        eval_results_path="/path/to/eval/results",
        model=None,
        prompt_template=prompt_template,
    )

    warning_msg = (
        "A prompt template, but no corresponding model, was provided."
        "Model outputs from the dataset will be used, and this prompt template will be ignored."
    )
    mock_logger.assert_called_once_with(warning_msg)


def test_evaluate_dataset_no_model_no_model_output_column():
    """
    GIVEN invalid arguments: a `model` that is None and a dataset
        without a model output column.
    WHEN the `evaluate_dataset` function is called.
    THEN the correct exception is raised.
    """
    mock_dataset = Mock()
    mock_dataset.columns = Mock(return_value=[])
    err_msg = (
        "evaluate_dataset has been given a dataset with no model output column "
        "and no ModelRunner to obtain outputs from. Please either provide a model "
        "or use a dataset that contains model outputs already."
    )
    with pytest.raises(EvalAlgorithmClientError, match=err_msg):
        evaluate_dataset(
            dataset=mock_dataset,
            pipeline=Mock(),
            dataset_name="my_dataset",
            eval_name="MyEvalAlgo",
            metric_names=[SCORE_1, SCORE_2],
            eval_results_path="/path/to/eval/results",
            model=None,
            prompt_template=None,
        )
