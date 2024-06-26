import json
import os
import tempfile
from collections import OrderedDict
from typing import NamedTuple, Optional
from unittest.mock import Mock, patch

import pytest
import ray

from fmeval.constants import DatasetColumns, EVAL_OUTPUT_RECORDS_BATCH_SIZE, MEAN
from fmeval.eval_algorithms import EvalScore, CategoryScore, EvalOutput
from fmeval.eval_algorithms.common import save_dataset, evaluate_dataset
from fmeval.eval_algorithms.save_strategy import FileSaveStrategy
from fmeval.exceptions import EvalAlgorithmClientError

SCORE_1 = "score_1"
SCORE_2 = "score_2"


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
    save_dataset(dataset, score_names, FileSaveStrategy(full_path_to_file))

    # THEN
    assert os.path.isfile(full_path_to_file)
    with open(full_path_to_file) as file_handle:
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
                assert json_obj["scores"] == [
                    {"name": "rouge", "value": 0.5},
                    {"name": "bert_score", "value": 0.42},
                ]

            if json_obj[DatasetColumns.MODEL_INPUT.value.name] == "world":
                assert json_obj[DatasetColumns.CATEGORY.value.name] == "Gender"
                assert json_obj["scores"] == [
                    {"name": "rouge", "value": 0.314},
                    {"name": "bert_score", "value": 0.271},
                ]


@pytest.mark.parametrize("file_name", ["my_dataset.jsonl", "my_dataset"])
def test_save_dataset_with_error_eval_score(tmp_path, file_name):
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
            DatasetColumns.ERROR.value.name: "error generating rouge score",
            unused_column_name: "Arch",
            "rouge": None,
            "bert_score": 0.42,
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "world",
            DatasetColumns.CATEGORY.value.name: "Gender",
            DatasetColumns.ERROR.value.name: None,
            unused_column_name: "btw",
            "rouge": 0.314,
            "bert_score": 0.271,
        },
    ]
    dataset = ray.data.from_items(ds_items)
    score_names = ["rouge", "bert_score"]

    # WHEN
    full_path_to_file = str(tmp_path / file_name)
    save_dataset(dataset, score_names, FileSaveStrategy(full_path_to_file))

    # THEN
    assert os.path.isfile(full_path_to_file)
    with open(full_path_to_file) as file_handle:
        json_objects = [json.loads(line, object_pairs_hook=OrderedDict) for line in file_handle.readlines()]
        assert json_objects  # if nothing gets written to the file, this test would trivially pass
        for json_obj in json_objects:
            # want to ensure ordering of keys is correct, so we use list instead of set
            assert list(json_obj.keys()) == [
                DatasetColumns.MODEL_INPUT.value.name,
                DatasetColumns.CATEGORY.value.name,
                DatasetColumns.ERROR.value.name,
                "scores",
            ]
            assert json_obj[DatasetColumns.MODEL_INPUT.value.name] in {"hello", "world"}

            if json_obj[DatasetColumns.MODEL_INPUT.value.name] == "hello":
                assert json_obj[DatasetColumns.CATEGORY.value.name] == "Age"
                assert json_obj["scores"] == [
                    {"name": "rouge", "error": "error generating rouge score"},
                    {"name": "bert_score", "value": 0.42},
                ]

            if json_obj[DatasetColumns.MODEL_INPUT.value.name] == "world":
                assert json_obj[DatasetColumns.CATEGORY.value.name] == "Gender"
                assert json_obj["scores"] == [
                    {"name": "rouge", "value": 0.314},
                    {"name": "bert_score", "value": 0.271},
                ]


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
    save_dataset(dataset, score_names, FileSaveStrategy(full_path_to_file))

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
@patch("fmeval.eval_algorithms.common.generate_output_dataset_path")
@patch("fmeval.eval_algorithms.common.save_dataset")
@patch("fmeval.eval_algorithms.common.aggregate_evaluation_scores")
@patch("fmeval.eval_algorithms.common.TransformPipeline")
@patch("fmeval.eval_algorithms.common.create_model_invocation_pipeline")
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

    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = os.path.join(tmpdir, "custom", "path")
        save_strategy = FileSaveStrategy(results_path)
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
            save_strategy=save_strategy,
        )

        mock_create_invocation_pipeline.assert_called_once_with(model_runner, test_case.dataset_prompt_template)
        mock_transform_pipeline_cls.assert_called_once_with(
            [mock_create_invocation_pipeline.return_value, input_pipeline]
        )
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
                dataset=final_pipeline.execute.return_value, score_names=[SCORE_1, SCORE_2], save_strategy=save_strategy
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
@patch("fmeval.eval_algorithms.common.generate_output_dataset_path")
@patch("fmeval.eval_algorithms.common.save_dataset")
@patch("fmeval.eval_algorithms.common.aggregate_evaluation_scores")
@patch("fmeval.eval_algorithms.common.TransformPipeline")
@patch("fmeval.eval_algorithms.common.create_model_invocation_pipeline")
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

    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = os.path.join(tmpdir, "custom", "path")
        save_strategy = FileSaveStrategy(results_path)
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
            save_strategy=save_strategy if test_case.save else None,
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
                save_strategy=save_strategy,
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


@patch("fmeval.eval_algorithms.common.aggregate_evaluation_scores")
@patch("fmeval.eval_algorithms.common.logging.Logger.warning")
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
