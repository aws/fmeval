import re
from typing import List, NamedTuple
from unittest.mock import patch, Mock

import pytest
import ray
from _pytest.fixtures import fixture
from ray.data import Dataset

from fmeval.constants import (
    DatasetColumns,
)
from fmeval.eval_algorithms import CategoryScore, EvalOutput, EvalScore

from fmeval.eval_algorithms.factual_knowledge import (
    FactualKnowledge,
    FactualKnowledgeConfig,
    EXACT_INCLUSION,
    QUASI_EXACT_INCLUSION,
    _exact_inclusion_score,
    _quasi_exact_inclusion_score,
)
from fmeval.exceptions import EvalAlgorithmClientError

INVALID_TARGET_DELIMITER_ERROR_MESSAGE = (
    "Empty target_output_delimiter is provided. Please either provide a non-empty string, or set it to None"
)
INVALID_LOGICAL_OPERATOR_ERROR_MESSAGE = (
    "Invalid logical_operator is provided. The only valid inputs are strings " '"OR" and "AND".'
)


class TestFactualKnowledge:
    @fixture(scope="module")
    def config(self) -> FactualKnowledgeConfig:
        return FactualKnowledgeConfig(target_output_delimiter="<OR>", logical_operator="OR")

    class TestCaseFactualKnowledgeInvalidConfig(NamedTuple):
        error_message: str
        logical_operator: str
        target_delimiter: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseFactualKnowledgeInvalidConfig(
                error_message=INVALID_TARGET_DELIMITER_ERROR_MESSAGE,
                logical_operator="OR",
                target_delimiter="",
            ),
            TestCaseFactualKnowledgeInvalidConfig(
                error_message=INVALID_LOGICAL_OPERATOR_ERROR_MESSAGE,
                logical_operator="",
                target_delimiter="<OR>",
            ),
            TestCaseFactualKnowledgeInvalidConfig(
                error_message=INVALID_LOGICAL_OPERATOR_ERROR_MESSAGE,
                logical_operator="NOT AND",
                target_delimiter="<OR>",
            ),
            TestCaseFactualKnowledgeInvalidConfig(
                error_message=INVALID_LOGICAL_OPERATOR_ERROR_MESSAGE,
                logical_operator="Random",
                target_delimiter="<OR>",
            ),
            TestCaseFactualKnowledgeInvalidConfig(
                error_message=INVALID_LOGICAL_OPERATOR_ERROR_MESSAGE,
                logical_operator="<OR>",
                target_delimiter="<OR>",
            ),
        ],
    )
    def test_factual_knowledge_invalid_config(self, test_case):
        """
        GIVEN invalid inputs for target_output_delimiter and logical_operator
        WHEN FactualKnowledgeConfig is initialized
        THEN correct exception with proper message is raised
        """
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.error_message)):
            FactualKnowledgeConfig(
                target_output_delimiter=test_case.target_delimiter, logical_operator=test_case.logical_operator
            )

    @patch("src.fmeval.eval_algorithms.factual_knowledge.logging.Logger.warning")
    def test_factual_knowledge_warnings(self, mock_logger):
        """
        GIVEN inconsistent inputs for target_output_delimiter and logical_operator
        WHEN FactualKnowledgeConfig is initialized
        THEN correct warning with proper message is generated
        """
        warning_message = "The target_output_delimiter `{0}` and logical_operator `{1}` are not consistent."
        FactualKnowledgeConfig(target_output_delimiter="<AND>", logical_operator="OR")
        mock_logger.assert_called_with(warning_message.format("<AND>", "OR"))

        FactualKnowledgeConfig(target_output_delimiter="<OR>", logical_operator="AND")
        mock_logger.assert_called_with(warning_message.format("<OR>", "AND"))

    class TestCaseFactualKnowledgeEvaluateSample(NamedTuple):
        model_input: str
        model_output: str
        target_output: str
        target_delimiter: str
        logic_operator: str
        expected_response: List[EvalScore]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="London is the capital of",
                model_output="England",
                target_output="England<OR>UK",
                target_delimiter="<OR>",
                logic_operator="OR",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=1.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=1.0),
                ],
            ),
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="London is the capital of",
                model_output="England or wait Scotland",
                target_output="England<OR>UK",
                target_delimiter="<OR>",
                logic_operator="OR",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=1.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=1.0),
                ],
            ),
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="London is the capital of",
                model_output="England",
                target_output="India or maybe Pakistan",
                target_delimiter="<OR>",
                logic_operator="OR",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=0.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=0.0),
                ],
            ),
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="Pulp Fiction was directed by",
                model_output="Quentin Tarantino",
                target_output="QUENTIN TARANTINO",
                target_delimiter="<OR>",
                logic_operator="OR",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=1.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=1.0),
                ],
            ),
            # Adding tests for quasi-exact inclusion
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="Who is Andrew R. Jassy?",
                model_output="Chief Executive Officer of Amazon.com Inc.",
                target_output="Chief Executive Officer of Amazon.com, Inc.",
                target_delimiter="<OR>",
                logic_operator="OR",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=0.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=1.0),
                ],
            ),
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="Pulp Fiction was directed by",
                model_output=" Quentin   Tarantino ",
                target_output="QUENTIN TARANTINO",
                target_delimiter="<OR>",
                logic_operator="OR",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=0.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=1.0),
                ],
            ),
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="Who is Andrew R. Jassy?",
                model_output="Chief Executive Officer of Amazon.com, Inc.",
                target_output="Chief Executive Officer of Amazon.com Inc.",
                target_delimiter="<OR>",
                logic_operator="OR",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=0.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=1.0),
                ],
            ),
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="Who was the first president of the United States",
                model_output="George Washington - an American Founding Father",
                target_output="George Washington: an American Founding Father",
                target_delimiter="<OR>",
                logic_operator="OR",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=0.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=1.0),
                ],
            ),
            # tests that all facts are in the model output
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="The three primary colors are",
                model_output="Red, blue, and yellow",
                target_output="Red<AND>Blue<AND>Yellow",
                target_delimiter="<AND>",
                logic_operator="AND",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=1.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=1.0),
                ],
            ),
            # tests out of order
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="The three primary colors are",
                model_output="The three primary colors are blue, yellow, and red",
                target_output="Red<AND>Blue<AND>Yellow",
                target_delimiter="<AND>",
                logic_operator="AND",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=1.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=1.0),
                ],
            ),
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="The three primary colors are",
                model_output="Red and blue",
                target_output="Red<AND>Blue<AND>Yellow",
                target_delimiter="<AND>",
                logic_operator="AND",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=0.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=0.0),
                ],
            ),
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="How do I change the password on my computer?",
                model_output="According to my documents, change your password by first hitting control alt delete. "
                "Then click change my password. Then restart your compute",
                target_output="Control alt delete<AND>Change my password<AND>restart",
                target_delimiter="<AND>",
                logic_operator="AND",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=1.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=1.0),
                ],
            ),
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="How do I change the password on my computer?",
                model_output="According to my documents, change your password by first hitting control + alt + delete. "
                "Then click change my password. Then, restart your compute",
                target_output="Control alt delete<AND>Change my password<AND>restart",
                target_delimiter="<AND>",
                logic_operator="AND",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=0.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=1.0),
                ],
            ),
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="How many days can employees work from home at company X?",
                model_output="According to my documents, employees can work from home 10 to 20 days per month.",
                target_output="10<AND>20 days per month",
                target_delimiter="<AND>",
                logic_operator="AND",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=1.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=1.0),
                ],
            ),
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="How many days can employees work from home at company X?",
                model_output="According to my documents, employees can work from home 15 to 20 days per month.",
                target_output="10<AND>20 days per month",
                target_delimiter="<AND>",
                logic_operator="AND",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=0.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=0.0),
                ],
            ),
            # none of the target facts are contained in the model output
            TestCaseFactualKnowledgeEvaluateSample(
                model_input="What are the branches of the Federal Government",
                model_output="congress, president, and supreme court",
                target_output="legislative<AND>executive<AND>judicial",
                target_delimiter="<AND>",
                logic_operator="AND",
                expected_response=[
                    EvalScore(name=EXACT_INCLUSION, value=0.0),
                    EvalScore(name=QUASI_EXACT_INCLUSION, value=0.0),
                ],
            ),
        ],
    )
    def test_factual_knowledge_evaluate_sample(self, test_case):
        """
        GIVEN valid inputs
        WHEN FactualKnowledge.evaluate_sample is called
        THEN correct EvalScore is returned
        """
        eval_algorithm = FactualKnowledge(
            FactualKnowledgeConfig(
                target_output_delimiter=test_case.target_delimiter,
                logical_operator=test_case.logic_operator,
            )
        )
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
                        {
                            DatasetColumns.MODEL_INPUT.value.name: "What year did the RMS Titanic Sink?",
                            DatasetColumns.TARGET_OUTPUT.value.name: "1912",
                            DatasetColumns.CATEGORY.value.name: "History",
                            DatasetColumns.MODEL_OUTPUT.value.name: "It was the year of 1912.",
                        },
                        {
                            DatasetColumns.MODEL_INPUT.value.name: "When was the declaration of independence signed?",
                            DatasetColumns.TARGET_OUTPUT.value.name: "July 4, 1776",
                            DatasetColumns.CATEGORY.value.name: "History",
                            DatasetColumns.MODEL_OUTPUT.value.name: "July 4 - 1776",
                        },
                    ]
                ),
                expected_response=[
                    EvalOutput(
                        eval_name="factual_knowledge",
                        dataset_name="my_custom_dataset",
                        prompt_template=None,
                        dataset_scores=[
                            EvalScore(name=EXACT_INCLUSION, value=2 / 3),
                            EvalScore(name=QUASI_EXACT_INCLUSION, value=5 / 6),
                        ],
                        category_scores=[
                            CategoryScore(
                                name="Capitals",
                                scores=[
                                    EvalScore(name=EXACT_INCLUSION, value=0.5),
                                    EvalScore(name=QUASI_EXACT_INCLUSION, value=0.5),
                                ],
                            ),
                            CategoryScore(
                                name="Movies",
                                scores=[
                                    EvalScore(name=EXACT_INCLUSION, value=1.0),
                                    EvalScore(name=QUASI_EXACT_INCLUSION, value=1.0),
                                ],
                            ),
                            CategoryScore(
                                name="History",
                                scores=[
                                    EvalScore(name=EXACT_INCLUSION, value=0.5),
                                    EvalScore(name=QUASI_EXACT_INCLUSION, value=1.0),
                                ],
                            ),
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

    class TestCaseFactualKnowledgeEvalScore(NamedTuple):
        model_output: str
        target_output: str
        expected_score: float

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseFactualKnowledgeEvalScore(
                model_output="I live in New York!",
                target_output="i     live in new york.",
                expected_score=0,
            ),
            TestCaseFactualKnowledgeEvalScore(
                model_output="THis Is A BAD mOvie",
                target_output="This is a bad movie",
                expected_score=1,
            ),
            TestCaseFactualKnowledgeEvalScore(
                model_output="inclusion but not exact",
                target_output="inclusion",
                expected_score=1,
            ),
            TestCaseFactualKnowledgeEvalScore(
                model_output="Testing words in the middle",
                target_output="Testing in the middle",
                expected_score=0,
            ),
            TestCaseFactualKnowledgeEvalScore(
                model_output="random; punctuation",
                target_output="random punctuation",
                expected_score=0,
            ),
            TestCaseFactualKnowledgeEvalScore(
                model_output="completely different phrase",
                target_output="the correct answer",
                expected_score=0,
            ),
            TestCaseFactualKnowledgeEvalScore(
                model_output="Exact answer",
                target_output="Exact answer",
                expected_score=1,
            ),
        ],
    )
    def test_exact_inclusion_score(self, test_case):
        assert (
            _exact_inclusion_score(model_output=test_case.model_output, target_output=test_case.target_output)
            == test_case.expected_score
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseFactualKnowledgeEvalScore(
                model_output="I live in New York!",
                target_output="i     live in new york.",
                expected_score=1,
            ),
            TestCaseFactualKnowledgeEvalScore(
                model_output="THis Is A BAD mOvie",
                target_output="This is a bad movie",
                expected_score=1,
            ),
            TestCaseFactualKnowledgeEvalScore(
                model_output=" stripped text ",
                target_output="stripped text",
                expected_score=1,
            ),
            TestCaseFactualKnowledgeEvalScore(
                model_output="testing a missing article",
                target_output="testing missing article",
                expected_score=1,
            ),
            TestCaseFactualKnowledgeEvalScore(
                model_output="checking for inclusion",
                target_output="inclusion",
                expected_score=1,
            ),
            TestCaseFactualKnowledgeEvalScore(
                model_output="Testing words in the middle",
                target_output="Testing in the middle",
                expected_score=0,
            ),
            TestCaseFactualKnowledgeEvalScore(
                model_output="random; punctuation",
                target_output="random punctuation",
                expected_score=1,
            ),
            TestCaseFactualKnowledgeEvalScore(
                model_output="completely different phrase",
                target_output="the correct answer",
                expected_score=0,
            ),
            TestCaseFactualKnowledgeEvalScore(
                model_output="Exact answer",
                target_output="Exact answer",
                expected_score=1,
            ),
        ],
    )
    def test_quasi_exact_inclusion_score(self, test_case):
        assert (
            _quasi_exact_inclusion_score(model_output=test_case.model_output, target_output=test_case.target_output)
            == test_case.expected_score
        )
