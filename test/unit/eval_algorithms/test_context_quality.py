from typing import NamedTuple, List, Tuple
from unittest.mock import Mock, patch

import pytest
import ray
from ray.data import Dataset

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSONLINES,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms import EvalScore, EvalOutput

from fmeval.eval_algorithms.context_quality import ContextQuality, ContextQualityConfig, CONTEXT_PRECISION_SCORE


class TestCaseContextQualityEvaluateSample(NamedTuple):
    model_input: str
    target_output: str
    context: str
    mocked_judge_model_responses: List[Tuple]
    expected_response: List[EvalScore]


class TestCaseContextQualityEvaluate(NamedTuple):
    dataset: Dataset
    data_config: DataConfig
    mocked_judge_model_responses: List[Tuple]
    expected_response: List[EvalOutput]


CONTEXT_QUALITY_DATASET = ray.data.from_items(
    [
        {
            DatasetColumns.MODEL_INPUT.value.name: "What is the tallest mountain in the world?",
            DatasetColumns.TARGET_OUTPUT.value.name: "Mount Everest.",
            DatasetColumns.CONTEXT.value.name: ["The Andes is the longest continental mountain range in the world."],
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "Who won the most super bowls?",
            DatasetColumns.TARGET_OUTPUT.value.name: "Pittsburgh Steelers<OR>New England Patriots",
            DatasetColumns.CONTEXT.value.name: [
                "The AFC's Pittsburgh Steelers and New England Patriots have the most Super Bowl titles at six each."
            ],
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "What do the 3 dots mean in math?",
            DatasetColumns.TARGET_OUTPUT.value.name: "therefore sign",
            DatasetColumns.CONTEXT.value.name: [
                "The therefore sign is generally used before a logical consequence. ",
                "The symbol consists of three dots placed in an upright triangle and is read therefore.",
            ],
        },
    ]
)


class TestContextQuality:
    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseContextQualityEvaluateSample(
                model_input="What is the tallest mountain in the world?",
                target_output="Mount Everest.",
                context=[
                    "The Andes is the longest continental mountain range in the world",
                    "located in South America. ",
                    "It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. ",
                    "The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.",
                ],
                mocked_judge_model_responses=[("The score is 0", None), ("0", None), ("1", None), ("unknown", None)],
                expected_response=[EvalScore(name=CONTEXT_PRECISION_SCORE, value=0.3333333333)],
            ),
            TestCaseContextQualityEvaluateSample(
                model_input="What is the tallest mountain in the world?",
                target_output="Mount Everest.",
                context=["The Andes is the longest continental mountain range in the world"],
                mocked_judge_model_responses=[("0", None)],
                expected_response=[EvalScore(name=CONTEXT_PRECISION_SCORE, value=0)],
            ),
            TestCaseContextQualityEvaluateSample(
                model_input="Who won the most super bowls?",
                target_output="Pittsburgh Steelers<OR>New England Patriots",
                context=[
                    "The AFC's Pittsburgh Steelers and New England Patriots have the most Super Bowl titles at six each."
                ],
                mocked_judge_model_responses=[("1", None)],
                expected_response=[EvalScore(name=CONTEXT_PRECISION_SCORE, value=0.9999999999)],
            ),
            TestCaseContextQualityEvaluateSample(
                model_input="Who won the most super bowls?",
                target_output="Pittsburgh Steelers<OR>New England Patriots",
                context=[
                    "The Patriots have the most Super Bowl appearances at 11. ",
                    "The AFC's Pittsburgh Steelers and New England Patriots have the most Super Bowl titles at six each.",
                ],
                mocked_judge_model_responses=[("The score is 0", None), ("0", None), ("1", None)],
                expected_response=[EvalScore(name=CONTEXT_PRECISION_SCORE, value=0.5)],
            ),
        ],
    )
    def test_context_quality_evaluate_sample(self, test_case):
        """
        GIVEN a dataset sample with model_input, target_output and context
        WHEN evaluate_sample is called.
        THEN the correct score is returned.
        """
        mock_runner = Mock()
        mock_runner.predict.side_effect = test_case.mocked_judge_model_responses
        config = ContextQualityConfig()

        eval_algorithm = ContextQuality(config)
        actual_response = eval_algorithm.evaluate_sample(
            model_input=test_case.model_input,
            target_output=test_case.target_output,
            context=test_case.context,
            judge_model=mock_runner,
        )
        assert actual_response == test_case.expected_response

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseContextQualityEvaluate(
                dataset=CONTEXT_QUALITY_DATASET,
                data_config=DataConfig(
                    dataset_name="retrieved_dataset",
                    dataset_uri="path",
                    dataset_mime_type=MIME_TYPE_JSONLINES,
                    model_input_location="model_input",
                    target_output_location="target_output",
                    context_location="context",
                ),
                mocked_judge_model_responses=[
                    ("0", None),
                    ("The score is 1", None),
                    ("1", None),
                    ("It's hard to say", None),
                ],
                expected_response=[
                    EvalOutput(
                        eval_name="context_quality",
                        dataset_name="retrieved_dataset",
                        dataset_scores=[EvalScore(name="context_precision_score", value=0.6666666666)],
                        prompt_template=None,
                        category_scores=None,
                        output_path="/tmp/eval_results/context_quality_retrieved_dataset.jsonl",
                        error=None,
                    )
                ],
            )
        ],
    )
    @patch("fmeval.eval_algorithms.context_quality.save_dataset")
    @patch("fmeval.eval_algorithms.context_quality.get_dataset")
    def test_context_quality_evaluate(self, get_dataset, save_dataset, test_case):
        """
        GIVEN an input dataset with model_input, target_output and context
        WHEN evaluate is called.
        THEN the correct eval output is returned.
        """
        get_dataset.return_value = test_case.dataset
        mock_runner = Mock()
        mock_runner.predict.side_effect = test_case.mocked_judge_model_responses

        config = ContextQualityConfig()
        eval_algorithm = ContextQuality(config)
        actual_response = eval_algorithm.evaluate(
            judge_model=mock_runner, dataset_config=test_case.data_config, save=True
        )
        assert actual_response == test_case.expected_response
        assert save_dataset.called
