import json
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
import pytest
import ray

from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple, Optional
from unittest.mock import Mock, patch, MagicMock, call
from ray.data import Dataset

from fmeval.constants import (
    DatasetColumns,
    EVAL_OUTPUT_RECORDS_BATCH_SIZE,
    PARALLELIZATION_FACTOR,
    MEAN,
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
    compute_and_aggregate_metrics,
    aggregate_evaluation_scores,
    evaluate,
    create_model_invocation_pipeline,
)
from fmeval.exceptions import EvalAlgorithmInternalError
from fmeval.transforms.common import GeneratePrompt, GetModelResponses
from fmeval.util import camel_to_snake, get_num_actors
from fmeval.eval_algorithms.util import get_bert_score

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
    gen_prompt, get_response = pipeline.transforms[0], pipeline.transforms[1]
    assert isinstance(gen_prompt, GeneratePrompt)
    assert gen_prompt.input_keys == [DatasetColumns.MODEL_INPUT.value.name]
    assert gen_prompt.output_keys == [DatasetColumns.PROMPT.value.name]
    assert gen_prompt.prompt_template == "Do something with $model_input"
    assert isinstance(get_response, GetModelResponses)
    assert get_response.input_keys == [DatasetColumns.PROMPT.value.name]
    assert get_response.output_keys == [DatasetColumns.MODEL_OUTPUT.value.name]
    assert get_response.model_runner == model


@pytest.mark.parametrize("save", [True, False])
@patch("fmeval.eval_algorithms.util.generate_output_dataset_path")
@patch("fmeval.eval_algorithms.util.save_dataset")
@patch("fmeval.eval_algorithms.util.aggregate_evaluation_scores")
def test_compute_and_aggregate_metrics(mock_aggregate, mock_save, mock_generate_output_path, save):
    """
    GIVEN valid arguments.
    WHEN compute_and_aggregate_metrics is called.
    THEN expected function calls are made and the correct EvalOutput is returned.
    """
    mock_aggregate.return_value = DATASET_SCORES, CATEGORY_SCORES
    pipeline = Mock()
    pipeline.execute = Mock(return_value=Mock())
    mock_generate_output_path.return_value = "path/to/output/dataset"

    actual_eval_output = compute_and_aggregate_metrics(
        pipeline=pipeline,
        dataset=Mock(),
        dataset_name="my_dataset",
        dataset_prompt_template="Do something with $model_input",
        eval_name="MyEvalAlgo",
        metric_names=[SCORE_1, SCORE_2],
        eval_results_path="/path/to/eval_results",
        save=save,
    )

    mock_aggregate.assert_called_once_with(
        pipeline.execute.return_value,
        [SCORE_1, SCORE_2],
        agg_method=MEAN,
    )

    mock_generate_output_path.assert_called_once_with(
        path_to_parent_dir="/path/to/eval_results",
        eval_name="MyEvalAlgo",
        dataset_name="my_dataset",
    )

    assert actual_eval_output == EvalOutput(
        eval_name="MyEvalAlgo",
        dataset_name="my_dataset",
        prompt_template="Do something with $model_input",
        dataset_scores=DATASET_SCORES,
        category_scores=CATEGORY_SCORES,
        output_path="path/to/output/dataset",
    )

    if save:
        mock_save.assert_called_once_with(
            dataset=pipeline.execute.return_value,
            score_names=[SCORE_1, SCORE_2],
            path="path/to/output/dataset",
        )
    else:
        mock_save.assert_not_called()


class TestCaseEvaluate(NamedTuple):
    user_provided_prompt_template: Optional[str]
    dataset_prompt_template: str
    save: bool


@pytest.mark.parametrize(
    "test_case",
    [
        TestCaseEvaluate(
            user_provided_prompt_template="Answer: $model_input",
            dataset_prompt_template="Answer: $model_input",
            save=True,
        ),
        TestCaseEvaluate(
            user_provided_prompt_template=None,
            dataset_prompt_template="$model_input",
            save=False,
        ),
    ],
)
@patch("fmeval.eval_algorithms.util.compute_and_aggregate_metrics")
@patch("fmeval.eval_algorithms.util.create_model_invocation_pipeline")
@patch("fmeval.eval_algorithms.util.TransformPipeline")
@patch("fmeval.eval_algorithms.util.validate_dataset")
@patch("fmeval.eval_algorithms.util.get_dataset")
@patch("fmeval.eval_algorithms.util.get_dataset_configs")
def test_evaluate_no_model_output_column(
    mock_get_dataset_configs,
    mock_get_dataset,
    mock_validate_dataset,
    mock_transform_pipeline,
    mock_standard_invocation_pipeline,
    mock_compute_and_aggregate,
    test_case,
):
    """
    GIVEN multiple dataset configs where none of the corresponding datasets
        contain a model output column.
    WHEN the `evaluate` util function is called.
    THEN all expected functions are called with the correct arguments and a list
        of EvalOutputs generated by `compute_and_aggregate_metrics` is returned.
    """
    config_1, config_2 = Mock(), Mock()
    configs = [config_1, config_2]
    config_1.dataset_name = "custom_dataset_1"
    config_2.dataset_name = "custom_dataset_2"
    mock_get_dataset_configs.return_value = configs

    loaded_datasets = [Mock(), Mock()]
    for ds in loaded_datasets:
        ds.columns = Mock(return_value=[])  # So that model outputs will be generated
    mock_get_dataset.side_effect = loaded_datasets

    mock_model = Mock()
    mock_pipeline = Mock()  # the pipeline that is passed to `evaluate`

    model_invocation_pipeline_1, model_invocation_pipeline_2 = Mock(), Mock()
    invocation_pipelines = [model_invocation_pipeline_1, model_invocation_pipeline_2]
    mock_standard_invocation_pipeline.side_effect = invocation_pipelines

    # The pipelines that are passed to `compute_and_aggregate_metrics`
    # (which have the GeneratePrompt and GetModelResponses Transforms prepended)
    pipeline_to_execute_1, pipeline_to_execute_2 = Mock(), Mock()
    pipelines_to_execute = [pipeline_to_execute_1, pipeline_to_execute_2]
    mock_transform_pipeline.side_effect = pipelines_to_execute

    mocked_eval_outputs = [
        EvalOutput(
            eval_name="my_eval",
            dataset_name="my_dataset_1",
            dataset_scores=[EvalScore(name=SCORE_1, value=0.162), EvalScore(name=SCORE_2, value=0.189)],
        ),
        EvalOutput(
            eval_name="my_eval",
            dataset_name="my_dataset_2",
            dataset_scores=[EvalScore(name=SCORE_1, value=0.126), EvalScore(name=SCORE_2, value=0.127)],
        ),
    ]
    mock_compute_and_aggregate.side_effect = mocked_eval_outputs

    eval_outputs = evaluate(
        eval_name="MyEvalAlgo",
        pipeline=mock_pipeline,
        metric_names=[SCORE_1, SCORE_2],
        required_columns=["required_1", "required_2"],
        eval_results_path="/path/to/eval/results",
        model=mock_model,
        prompt_template=test_case.user_provided_prompt_template,
        num_records=200,
        save=test_case.save,
    )

    mock_get_dataset.assert_has_calls([call(config, 200) for config in configs])
    mock_validate_dataset.assert_has_calls([call(dataset, ["required_1", "required_2"]) for dataset in loaded_datasets])
    mock_standard_invocation_pipeline.assert_has_calls(
        [call(mock_model, test_case.dataset_prompt_template) for _ in range(2)]
    )
    mock_transform_pipeline.assert_has_calls(
        [call([invocation_pipeline, mock_pipeline]) for invocation_pipeline in invocation_pipelines]
    )
    mock_compute_and_aggregate.assert_has_calls(
        [
            call(
                pipeline_to_execute,
                dataset,
                config.dataset_name,
                test_case.dataset_prompt_template,
                "MyEvalAlgo",
                [SCORE_1, SCORE_2],
                "/path/to/eval/results",
                agg_method=MEAN,
                save=test_case.save,
            )
            for pipeline_to_execute, dataset, config in zip(pipelines_to_execute, loaded_datasets, configs)
        ]
    )

    assert eval_outputs == mocked_eval_outputs


@pytest.mark.parametrize(
    "test_case",
    [
        TestCaseEvaluate(
            user_provided_prompt_template="Answer: $model_input",
            dataset_prompt_template="Answer: $model_input",
            save=False,
        ),
        TestCaseEvaluate(
            user_provided_prompt_template=None,
            dataset_prompt_template="$model_input",
            save=True,
        ),
    ],
)
@patch("fmeval.eval_algorithms.util.compute_and_aggregate_metrics")
@patch("fmeval.eval_algorithms.util.create_model_invocation_pipeline")
@patch("fmeval.eval_algorithms.util.validate_dataset")
@patch("fmeval.eval_algorithms.util.get_dataset")
@patch("fmeval.eval_algorithms.util.get_dataset_configs")
def test_evaluate_with_model_output_column(
    mock_get_dataset_configs,
    mock_get_dataset,
    mock_validate_dataset,
    mock_standard_invocation_pipeline,
    mock_compute_and_aggregate,
    test_case,
):
    """
    GIVEN a dataset config where its corresponding dataset already contains a model output column.
    WHEN the `evaluate` util function is called.
    THEN all expected functions are called with the correct arguments,
        the TransformPipeline that is passed in does *not* get prepended with
        prompt-generation and model-invocation Transforms, and a list of EvalOutputs
        generated by `compute_and_aggregate_metrics` is returned.
    """
    config = Mock()
    config.dataset_name = "custom_dataset"
    mock_get_dataset_configs.return_value = [config]

    # So that model outputs won't be generated
    input_dataset = Mock()
    input_dataset.columns = Mock(return_value=[DatasetColumns.MODEL_OUTPUT.value.name])
    mock_get_dataset.return_value = input_dataset

    mock_model = Mock()
    mock_pipeline = Mock()  # the pipeline that is passed to `evaluate`

    mocked_eval_outputs = [
        EvalOutput(
            eval_name="my_eval",
            dataset_name="my_dataset_1",
            dataset_scores=[EvalScore(name=SCORE_1, value=0.162), EvalScore(name=SCORE_2, value=0.189)],
        )
    ]
    mock_compute_and_aggregate.side_effect = mocked_eval_outputs

    eval_outputs = evaluate(
        eval_name="MyEvalAlgo",
        pipeline=mock_pipeline,
        metric_names=[SCORE_1, SCORE_2],
        required_columns=["required_1", "required_2"],
        eval_results_path="/path/to/eval/results",
        model=mock_model,
        prompt_template=test_case.user_provided_prompt_template,
        num_records=200,
        save=test_case.save,
    )

    mock_get_dataset.assert_called_once_with(config, 200)
    mock_validate_dataset.assert_called_with(input_dataset, ["required_1", "required_2"])
    mock_standard_invocation_pipeline.assert_not_called()
    mock_compute_and_aggregate.called_once_with(
        mock_pipeline,
        input_dataset,
        config.dataset_name,
        test_case.dataset_prompt_template,
        "MyEvalAlgo",
        [SCORE_1, SCORE_2],
        "/path/to/eval/results",
        agg_method=MEAN,
        save=test_case.save,
    )

    assert eval_outputs == mocked_eval_outputs
