import re
from typing import List, NamedTuple
from unittest.mock import patch, Mock

import pytest
import ray
from _pytest.fixtures import fixture
from ray.data import Dataset

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON,
)
from fmeval.eval_algorithms import CategoryScore, EvalAlgorithm, EvalOutput, EvalScore
from fmeval.eval_algorithms.factual_knowledge import FactualKnowledge, FactualKnowledgeConfig
from fmeval.exceptions import EvalAlgorithmClientError


class TestFactualKnowledge:
    @fixture(scope="module")
    def config(self) -> FactualKnowledgeConfig:
        return FactualKnowledgeConfig(target_output_delimiter="<OR>")

    def test_factual_knowledge_invalid_config(self):
        """
        GIVEN empty string target_output_delimiter
        WHEN FactualKnowledgeConfig is initialized
        THEN correct exception with proper message is raised
        """
        expected_error_message = (
            "Empty target_output_delimiter is provided. " "Please either provide a non-empty string, or set it to None"
        )
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(expected_error_message)):
            FactualKnowledgeConfig(target_output_delimiter="")

    class TestCaseFactualKnowledgeEvaluateSample(NamedTuple):
        model_input: str
        model_output: str
        target_output: str
        expected_response: List[EvalScore]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="London is the capital of",
                model_output="England",
                target_output="England<OR>UK",
                expected_response=[EvalScore(name=EvalAlgorithm.FACTUAL_KNOWLEDGE.value, value=1)],
            ),
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="London is the capital of",
                model_output="England or wait Scotland",
                target_output="England<OR>UK",
                expected_response=[EvalScore(name=EvalAlgorithm.FACTUAL_KNOWLEDGE.value, value=1)],
            ),
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="London is the capital of",
                model_output="India or maybe Pakistan",
                target_output="England<OR>UK",
                expected_response=[EvalScore(name=EvalAlgorithm.FACTUAL_KNOWLEDGE.value, value=0)],
            ),
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="Pulp Fiction was directed by",
                model_output="Quentin Tarantino",
                target_output="QUENTIN TARANTINO",
                expected_response=[EvalScore(name=EvalAlgorithm.FACTUAL_KNOWLEDGE.value, value=1)],
            ),
        ],
    )
    def test_factual_knowledge_evaluate_sample(self, test_case, config):
        """
        GIVEN valid inputs
        WHEN FactualKnowledge.evaluate_sample is called
        THEN correct EvalScore is returned
        """
        eval_algorithm = FactualKnowledge(config)
        actual_response = eval_algorithm.evaluate_sample(test_case.target_output, test_case.model_output)
        assert test_case.expected_response == actual_response

    class TestCaseFactualKnowledgeEvaluate(NamedTuple):
        input_dataset: Dataset
        expected_response: List[EvalOutput]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseFactualKnowledgeEvaluate(
                input_dataset=ray.data.from_items(
                    [
                        {
                            DatasetColumns.MODEL_INPUT.value.name: "London is the capital of",
                            DatasetColumns.TARGET_OUTPUT.value.name: "England<OR>UK",
                            DatasetColumns.CATEGORY.value.name: "Capitals",
                            DatasetColumns.MODEL_OUTPUT.value.name: "uk",
                        },
                        {
                            DatasetColumns.MODEL_INPUT.value.name: "Paris is the capital of",
                            DatasetColumns.TARGET_OUTPUT.value.name: "France",
                            DatasetColumns.CATEGORY.value.name: "Capitals",
                            DatasetColumns.MODEL_OUTPUT.value.name: "uk",
                        },
                        {
                            DatasetColumns.MODEL_INPUT.value.name: "Pulp Fiction was directed by",
                            DatasetColumns.TARGET_OUTPUT.value.name: "QUENTIN TARANTINO",
                            DatasetColumns.CATEGORY.value.name: "Movies",
                            DatasetColumns.MODEL_OUTPUT.value.name: "Quentin Tarantino",
                        },
                        {
                            DatasetColumns.MODEL_INPUT.value.name: "Dark knight was directed by",
                            DatasetColumns.TARGET_OUTPUT.value.name: "Christopher Nolan<OR>NOLAN",
                            DatasetColumns.CATEGORY.value.name: "Movies",
                            DatasetColumns.MODEL_OUTPUT.value.name: "nolan",
                        },
                    ]
                ),
                expected_response=[
                    EvalOutput(
                        eval_name="factual_knowledge",
                        dataset_name="my_custom_dataset",
                        prompt_template=None,
                        dataset_scores=[EvalScore(name="factual_knowledge", value=0.75)],
                        category_scores=[
                            CategoryScore(name="Capitals", scores=[EvalScore(name="factual_knowledge", value=0.5)]),
                            CategoryScore(name="Movies", scores=[EvalScore(name="factual_knowledge", value=1.0)]),
                        ],
                        output_path="/tmp/eval_results/factual_knowledge_my_custom_dataset.jsonl",
                    )
                ],
            )
        ],
    )
    @patch("fmeval.eval_algorithms.factual_knowledge.validate_dataset")
    @patch("fmeval.eval_algorithms.factual_knowledge.get_dataset")
    @patch("fmeval.eval_algorithms.factual_knowledge.get_dataset_configs")
    def test_factual_knowledge_evaluate_without_model(
        self, mock_get_dataset_configs, mock_get_dataset, mock_validate_dataset, test_case, config
    ):
        """
        GIVEN a valid dataset and no model.
        WHEN FactualKnowledge.evaluate is called.
        THEN the correct output is returned.
        """
        dataset_config = Mock()
        dataset_config.dataset_name = "my_custom_dataset"
        mock_get_dataset_configs.return_value = [dataset_config]

        mock_get_dataset.return_value = test_case.input_dataset
        eval_algorithm = FactualKnowledge(config)
        output = eval_algorithm.evaluate(model=None, dataset_config=dataset_config)
        mock_validate_dataset.assert_called_once_with(
            mock_get_dataset.return_value, [DatasetColumns.TARGET_OUTPUT.value.name]
        )
        assert output == test_case.expected_response
