import json
import multiprocessing as mp
import os
import re

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
    PARALLELIZATION_FACTOR,
    MEAN,
    NUM_ROWS_DETERMINISTIC,
)
from fmeval.eval_algorithms import (
    EvalAlgorithm,
    DATASET_CONFIGS,
    EVAL_DATASETS,
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
    get_dataset_configs,
    generate_model_predict_response_for_dataset,
    generate_prompt_column_for_dataset,
    category_wise_aggregation,
    dataset_aggregation,
    EvalOutputRecord,
    generate_output_dataset_path,
    generate_mean_delta_score,
    verify_model_determinism,
    get_dataset_configs,
    aggregate_evaluation_scores,
    create_model_invocation_pipeline,
    validate_prompt_template,
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


def test_validate_prompt_template_success():
    """
    GIVEN a prompt_template and required placeholder keywords
    WHEN validate_prompt_template is called
    THEN no exception is raised
    """
    validate_prompt_template(
        prompt_template='{"Question":$question, "Answer": $answer}', placeholders=["question", "answer"]
    )


def test_validate_prompt_template_raise_error():
    """
    GIVEN placeholder keywords and a prompt_template doesn't contain required placeholder
    WHEN validate_prompt_template is called
    THEN raise EvalAlgorithmClientError with correct error message
    """
    with pytest.raises(EvalAlgorithmClientError, match=re.escape("Unable to find placeholder")):
        validate_prompt_template(
            prompt_template='{"Question":$question, "Answer": $answer}', placeholders=["model_input"]
        )


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
        scores=[EvalScore(name="rouge", value=0.5), EvalScore(name="bert", error="error generating bert score")],
        dataset_columns={
            DatasetColumns.MODEL_OUTPUT.value.name: "output",
            DatasetColumns.MODEL_INPUT.value.name: "input",
        },
    )
    expected_record = OrderedDict(
        [
            (DatasetColumns.MODEL_INPUT.value.name, "input"),
            (DatasetColumns.MODEL_OUTPUT.value.name, "output"),
            ("scores", [{"name": "rouge", "value": 0.5}, {"name": "bert", "error": "error generating bert score"}]),
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
        "bert": None,
        "invalid_col_1": "hello",
        DatasetColumns.MODEL_INPUT.value.name: "input",
        "invalid_col_2": "world",
        DatasetColumns.ERROR.value.name: "error generating bert score",
    }
    expected_record = EvalOutputRecord(
        scores=[EvalScore(name="rouge", value=0.42), EvalScore(name="bert", error="error generating bert score")],
        dataset_columns={
            DatasetColumns.MODEL_INPUT.value.name: "input",
            DatasetColumns.MODEL_OUTPUT.value.name: "output",
            DatasetColumns.ERROR.value.name: "error generating bert score",
        },
    )

    assert EvalOutputRecord.from_row(row, score_names=["rouge", "bert"]) == expected_record


@pytest.mark.parametrize("parent_dir_path", ["path/to/parent", "path/to/parent/"])
def test_generate_output_dataset_path(parent_dir_path):
    eval_name = "eval"
    dataset_name = "dataset"
    actual = generate_output_dataset_path(parent_dir_path, eval_name, dataset_name)
    assert actual == "path/to/parent/eval_dataset.jsonl"


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


class TestGetDatasetConfigs:
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
            TestCaseBuiltinDatasetConfigs(
                eval_name=EvalAlgorithm.PROMPT_STEREOTYPING.value, dataset_names=[CROWS_PAIRS]
            ),
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
    def test_default_datasets(self, eval_name, dataset_names):
        """
        GIVEN an input argument of None and an evaluation algorithm name
        WHEN get_dataset_configs is called to normalize configuration
        THEN a list with all builtin dataset configs corresponding to the evaluation algorithm is
            returned.
        """
        configs = get_dataset_configs(None, eval_name)
        assert configs == [DATASET_CONFIGS[dataset_name] for dataset_name in dataset_names]

    def test_one_dataset(self):
        """
        GIVEN one input data config
        WHEN get_dataset_configs is called to normalize configuration
        THEN a list is returned containing the one provided config
        """
        custom_config = Mock()
        assert get_dataset_configs(custom_config, "blah") == [custom_config]

    def test_list_multiple_datasets(self):
        """
        GIVEN a list of multiple custom data configurations
        WHEN get_dataset_configs is called to normalize configuration
        THEN the returned list matches the input list
        """
        custom_configs = [Mock(), Mock(), Mock()]
        assert get_dataset_configs(custom_configs, "blah") == custom_configs

    def test_tuple_multiple_datasets(self):
        """
        GIVEN a tuple of multiple custom data configurations
        WHEN get_dataset_configs is called to normalize configuration
        THEN the returned list matches the input tuple
        """
        custom_configs = (Mock(), Mock(), Mock())
        normalized = get_dataset_configs(custom_configs, "blah")
        assert isinstance(normalized, list)
        assert normalized == list(custom_configs)


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
    if expected_category_scores is not None:
        assert len(category_scores) == len(expected_category_scores)
        category_scores.sort(key=lambda x: x.name)
        expected_category_scores.sort(key=lambda x: x.name)
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
