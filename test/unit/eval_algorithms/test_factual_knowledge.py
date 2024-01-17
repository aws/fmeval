import re
from typing import NamedTuple, List, Optional
from unittest.mock import patch

import pytest
import ray
from _pytest.fixtures import fixture
from ray.data import Dataset

from fmeval.constants import (
    ColumnNames,
    MIME_TYPE_JSON,
)
from fmeval.eval_algorithms.eval_algorithm import DataConfig
from fmeval.eval_algorithms import EvalOutput, CategoryScore, EvalScore, EvalAlgorithm, DEFAULT_PROMPT_TEMPLATE
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

    class TestCaseFactualKnowledgeEvaluateSampleInvalid(NamedTuple):
        model_input: Optional[str]
        model_output: str
        target_output: Optional[str]
        expected_error_message: str

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

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseFactualKnowledgeEvaluateSampleInvalid(
                model_input="London is the capital of",
                model_output="England",
                target_output=None,
                expected_error_message="Missing required input: target_output, for FactualKnowledge evaluate_sample",
            ),
            TestCaseFactualKnowledgeEvaluateSampleInvalid(
                model_input="Pulp Fiction was directed by",
                model_output=None,
                target_output="QUENTIN TARANTINO",
                expected_error_message="Missing required input: model_output, for FactualKnowledge evaluate_sample",
            ),
        ],
    )
    def test_factual_knowledge_evaluate_sample_invalid_input(self, test_case, config):
        """
        GIVEN invalid inputs
        WHEN FactualKnowledge.evaluate_sample is called
        THEN correct exception with proper message is raised
        """
        eval_algorithm = FactualKnowledge(config)
        with pytest.raises(EvalAlgorithmClientError, match=test_case.expected_error_message):
            eval_algorithm.evaluate_sample(test_case.target_output, test_case.model_output)

    class TestCaseFactualKnowledgeEvaluate(NamedTuple):
        input_dataset: Dataset
        prompt_template: Optional[str]
        dataset_config: Optional[DataConfig]
        input_dataset_with_generated_model_output: Optional[Dataset]
        expected_response: List[EvalOutput]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseFactualKnowledgeEvaluate(
                input_dataset=ray.data.from_items(
                    [
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "London is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "England<OR>UK",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Paris is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "France",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Pulp Fiction was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "QUENTIN TARANTINO",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Dark knight was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "Christopher Nolan<OR>NOLAN",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies",
                        },
                    ]
                ),
                dataset_config=None,
                prompt_template=None,
                input_dataset_with_generated_model_output=ray.data.from_items(
                    [
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "London is the capital of",
                            ColumnNames.PROMPT_COLUMN_NAME.value: "London is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "England<OR>UK",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "uk",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Paris is the capital of",
                            ColumnNames.PROMPT_COLUMN_NAME.value: "Paris is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "France",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "uk",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Pulp Fiction was directed by",
                            ColumnNames.PROMPT_COLUMN_NAME.value: "Pulp Fiction was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "QUENTIN TARANTINO",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "Quentin Tarantino",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Dark knight was directed by",
                            ColumnNames.PROMPT_COLUMN_NAME.value: "Dark knight was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "Christopher Nolan<OR>NOLAN",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "nolan",
                        },
                    ]
                ),
                expected_response=[
                    EvalOutput(
                        eval_name="factual_knowledge",
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        dataset_name="trex",
                        dataset_scores=[EvalScore(name="factual_knowledge", value=0.75)],
                        category_scores=[
                            CategoryScore(name="Capitals", scores=[EvalScore(name="factual_knowledge", value=0.5)]),
                            CategoryScore(name="Movies", scores=[EvalScore(name="factual_knowledge", value=1.0)]),
                        ],
                        output_path="/tmp/eval_results/factual_knowledge_trex.jsonl",
                    )
                ],
            ),
            TestCaseFactualKnowledgeEvaluate(
                input_dataset=ray.data.from_items(
                    [
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "London is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "England<OR>UK",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Paris is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "France",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Pulp Fiction was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "QUENTIN TARANTINO",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Dark knight was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "Christopher Nolan<OR>NOLAN",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies",
                        },
                    ]
                ),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                prompt_template="$feature",
                input_dataset_with_generated_model_output=ray.data.from_items(
                    [
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "London is the capital of",
                            ColumnNames.PROMPT_COLUMN_NAME.value: "London is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "England<OR>UK",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "uk",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Paris is the capital of",
                            ColumnNames.PROMPT_COLUMN_NAME.value: "Paris is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "France",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "uk",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Pulp Fiction was directed by",
                            ColumnNames.PROMPT_COLUMN_NAME.value: "Pulp Fiction was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "QUENTIN TARANTINO",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "Quentin Tarantino",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Dark knight was directed by",
                            ColumnNames.PROMPT_COLUMN_NAME.value: "Dark knight was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "Christopher Nolan<OR>NOLAN",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "nolan",
                        },
                    ]
                ),
                expected_response=[
                    EvalOutput(
                        eval_name="factual_knowledge",
                        dataset_name="my_custom_dataset",
                        prompt_template="$feature",
                        dataset_scores=[EvalScore(name="factual_knowledge", value=0.75)],
                        category_scores=[
                            CategoryScore(name="Capitals", scores=[EvalScore(name="factual_knowledge", value=0.5)]),
                            CategoryScore(name="Movies", scores=[EvalScore(name="factual_knowledge", value=1.0)]),
                        ],
                        output_path="/tmp/eval_results/factual_knowledge_my_custom_dataset.jsonl",
                    )
                ],
            ),
            TestCaseFactualKnowledgeEvaluate(
                input_dataset=ray.data.from_items(
                    [
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "London is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "England<OR>UK",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Paris is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "France",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Pulp Fiction was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "QUENTIN TARANTINO",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Dark knight was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "Christopher Nolan<OR>NOLAN",
                        },
                    ]
                ),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                prompt_template=None,
                input_dataset_with_generated_model_output=ray.data.from_items(
                    [
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "London is the capital of",
                            ColumnNames.PROMPT_COLUMN_NAME.value: "London is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "England<OR>UK",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "uk",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Paris is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "France",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "uk",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Pulp Fiction was directed by",
                            ColumnNames.PROMPT_COLUMN_NAME.value: "London is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "QUENTIN TARANTINO",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "Quentin Tarantino",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Dark knight was directed by",
                            ColumnNames.PROMPT_COLUMN_NAME.value: "London is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "Christopher Nolan<OR>NOLAN",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "nolan",
                        },
                    ]
                ),
                expected_response=[
                    EvalOutput(
                        eval_name="factual_knowledge",
                        dataset_name="my_custom_dataset",
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        dataset_scores=[EvalScore(name="factual_knowledge", value=0.75)],
                        category_scores=None,
                        output_path="/tmp/eval_results/factual_knowledge_my_custom_dataset.jsonl",
                    )
                ],
            ),
        ],
    )
    @patch("fmeval.model_runners.model_runner.ModelRunner")
    @patch("fmeval.eval_algorithms.factual_knowledge.get_dataset")
    @patch("fmeval.eval_algorithms.factual_knowledge.save_dataset")
    @patch("fmeval.eval_algorithms.factual_knowledge.generate_model_predict_response_for_dataset")
    def test_factual_knowledge_evaluate(
        self, generate_model_predict_response_for_dataset, save_dataset, get_dataset, model, test_case, config
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset without model_outputs, an input ModelRunner
            and request to save records with scores
        WHEN FactualKnowledge.evaluate is called
        THEN correct EvalOutput is returned
        """
        get_dataset.return_value = test_case.input_dataset
        generate_model_predict_response_for_dataset.return_value = test_case.input_dataset_with_generated_model_output
        eval_algorithm = FactualKnowledge(config)
        actual_response = eval_algorithm.evaluate(
            model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template, save=True
        )
        assert actual_response == test_case.expected_response
        assert save_dataset.called

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseFactualKnowledgeEvaluate(
                input_dataset=ray.data.from_items(
                    [
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "London is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "England<OR>UK",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "uk",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Paris is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "France",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "uk",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Pulp Fiction was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "QUENTIN TARANTINO",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "Quentin Tarantino",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Dark knight was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "Christopher Nolan<OR>NOLAN",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies",
                            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "nolan",
                        },
                    ]
                ),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                prompt_template=None,
                input_dataset_with_generated_model_output=None,
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
    @patch("fmeval.eval_algorithms.factual_knowledge.get_dataset")
    @patch("fmeval.eval_algorithms.factual_knowledge.save_dataset")
    @patch("fmeval.eval_algorithms.factual_knowledge.generate_model_predict_response_for_dataset")
    def test_factual_knowledge_evaluate_without_model(
        self, generate_model_predict_response_for_dataset, save_dataset, get_dataset, test_case, config
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset with model_outputs,
            and no request to save records with scores
        WHEN FactualKnowledge.evaluate is called
        THEN correct EvalOutput is returned
        """
        get_dataset.return_value = test_case.input_dataset
        generate_model_predict_response_for_dataset.return_value = test_case.input_dataset_with_generated_model_output
        eval_algorithm = FactualKnowledge(config)
        actual_response = eval_algorithm.evaluate(model=None, dataset_config=test_case.dataset_config)
        assert not generate_model_predict_response_for_dataset.called
        assert not save_dataset.called
        assert actual_response == test_case.expected_response

    class TestCaseFactualKnowledgeEvaluateInvalid(NamedTuple):
        input_dataset: Dataset
        dataset_config: Optional[DataConfig]
        prompt_template: Optional[str]
        model_provided: bool
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseFactualKnowledgeEvaluateInvalid(
                input_dataset=ray.data.from_items(
                    [
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "London is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "England<OR>UK",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Paris is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "France",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Pulp Fiction was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "QUENTIN TARANTINO",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Dark knight was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "Christopher Nolan<OR>NOLAN",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies",
                        },
                    ]
                ),
                dataset_config=None,
                prompt_template=None,
                model_provided=False,
                expected_error_message="No ModelRunner provided. ModelRunner is required for inference on model_inputs",
            ),
            TestCaseFactualKnowledgeEvaluateInvalid(
                input_dataset=ray.data.from_items(
                    [
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "London is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "England<OR>UK",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Paris is the capital of",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "France",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Pulp Fiction was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "QUENTIN TARANTINO",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies",
                        },
                        {
                            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Dark knight was directed by",
                            ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "Christopher Nolan<OR>NOLAN",
                            ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies",
                        },
                    ]
                ),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                model_provided=False,
                prompt_template=None,
                expected_error_message="No ModelRunner provided. ModelRunner is required for inference on model_inputs",
            ),
            TestCaseFactualKnowledgeEvaluateInvalid(
                input_dataset=ray.data.from_items(
                    [
                        {ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "London is the capital of", ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals"},
                        {ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Paris is the capital of", ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals"},
                        {ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Pulp Fiction was directed by", ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies"},
                        {ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "Dark knight was directed by", ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies"},
                    ]
                ),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                prompt_template=None,
                model_provided=True,
                expected_error_message="Missing required column: target_output, for evaluate() method",
            ),
            TestCaseFactualKnowledgeEvaluateInvalid(
                input_dataset=ray.data.from_items(
                    [
                        {ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "England<OR>UK", ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals"},
                        {ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "France", ColumnNames.CATEGORY_COLUMN_NAME.value: "Capitals"},
                        {ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "QUENTIN TARANTINO", ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies"},
                        {ColumnNames.TARGET_OUTPUT_COLUMN_NAME.value: "Christopher Nolan<OR>NOLAN", ColumnNames.CATEGORY_COLUMN_NAME.value: "Movies"},
                    ]
                ),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                prompt_template=None,
                model_provided=True,
                expected_error_message="Missing required column: model_input, for evaluate() method",
            ),
        ],
    )
    @patch("fmeval.model_runners.model_runner.ModelRunner")
    @patch("fmeval.eval_algorithms.factual_knowledge.get_dataset")
    def test_factual_knowledge_evaluate_invalid_input(self, get_dataset, model, test_case, config):
        """
        GIVEN invalid inputs
        WHEN FactualKnowledge.evaluate is called
        THEN correct exception with proper message is raised
        """
        eval_algorithm = FactualKnowledge(config)
        get_dataset.return_value = test_case.input_dataset
        if not test_case.model_provided:
            model = None
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            eval_algorithm.evaluate(
                model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template
            )
