import re
from typing import NamedTuple, List, Optional
from unittest.mock import patch

import pytest
import ray
from _pytest.fixtures import fixture
from ray.data import Dataset

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON,
    DEFAULT_EVAL_RESULTS_PATH,
)
from fmeval.eval_algorithms.eval_algorithm import DataConfig
from fmeval.eval_algorithms import (
    EvalOutput,
    CategoryScore,
    EvalScore,
    BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES,
    TRIVIA_QA,
    BOOLQ,
    NATURAL_QUESTIONS,
    DEFAULT_PROMPT_TEMPLATE,
)
from fmeval.eval_algorithms.qa_accuracy import (
    QAAccuracy,
    QAAccuracyConfig,
    F1_SCORE,
    EXACT_MATCH_SCORE,
    QUASI_EXACT_MATCH_SCORE,
    PRECISION,
    RECALL,
    _f1_score,
    _exact_match_score,
    _precision,
    _recall,
)
from fmeval.exceptions import EvalAlgorithmClientError

QA_DATASET = ray.data.from_items(
    [
        # Exact match so all scores should have perfect values.
        {
            DatasetColumns.MODEL_INPUT.value.name: "What is the capital of England?",
            DatasetColumns.TARGET_OUTPUT.value.name: "London",
            DatasetColumns.MODEL_OUTPUT.value.name: "London",
            DatasetColumns.CATEGORY.value.name: "capitals",
        },
        # Partial match.
        {
            DatasetColumns.MODEL_INPUT.value.name: "Who directed Pulp Fiction?",
            DatasetColumns.TARGET_OUTPUT.value.name: "Quentin Tarantino",
            DatasetColumns.MODEL_OUTPUT.value.name: "tarantino!",
            DatasetColumns.CATEGORY.value.name: "movies",
        },
        # Wrong answer. All scores should be zero.
        {
            DatasetColumns.MODEL_INPUT.value.name: "What is the capital of France?",
            DatasetColumns.TARGET_OUTPUT.value.name: "Paris",
            DatasetColumns.MODEL_OUTPUT.value.name: "London",
            DatasetColumns.CATEGORY.value.name: "capitals",
        },
        # Correct answer but with punctuation added.
        {
            DatasetColumns.MODEL_INPUT.value.name: "What is the capital of Italy?",
            DatasetColumns.TARGET_OUTPUT.value.name: "Rome",
            DatasetColumns.MODEL_OUTPUT.value.name: "rome!",
            DatasetColumns.CATEGORY.value.name: "capitals",
        },
        # Many correct answers.
        {
            DatasetColumns.MODEL_INPUT.value.name: "When did Argentina win the FIFA World Cup?",
            DatasetColumns.TARGET_OUTPUT.value.name: "1978<OR>1986<OR>2022",
            DatasetColumns.MODEL_OUTPUT.value.name: "2022",
            DatasetColumns.CATEGORY.value.name: "sports",
        },
        # Answer is longer than the model output.
        {
            DatasetColumns.MODEL_INPUT.value.name: "Did RMS Titanic sink in 1912?",
            DatasetColumns.TARGET_OUTPUT.value.name: "yes",
            DatasetColumns.MODEL_OUTPUT.value.name: "Yes. That is true.",
            DatasetColumns.CATEGORY.value.name: "history",
        },
    ]
)

QA_DATASET_WITHOUT_MODEL_OUTPUT = QA_DATASET.drop_columns(DatasetColumns.MODEL_OUTPUT.value.name)

QA_DATASET_WITHOUT_MODEL_INPUT = QA_DATASET.drop_columns(DatasetColumns.MODEL_INPUT.value.name)

QA_DATASET_WITHOUT_TARGET_OUTPUT = QA_DATASET.drop_columns(DatasetColumns.TARGET_OUTPUT.value.name)

QA_DATASET_WITHOUT_CATEGORY = QA_DATASET.drop_columns(DatasetColumns.CATEGORY.value.name)

QA_DATASET_WITHOUT_CATEGORY_WITHOUT_MODEL_OUTPUT = QA_DATASET_WITHOUT_CATEGORY.drop_columns(
    DatasetColumns.MODEL_OUTPUT.value.name
)

CATEGORY_SCORES = [
    CategoryScore(
        name="capitals",
        scores=[
            EvalScore(name=F1_SCORE, value=2 / 3),
            EvalScore(name=EXACT_MATCH_SCORE, value=1 / 3),
            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=2 / 3),
            EvalScore(name=PRECISION, value=2 / 3),
            EvalScore(name=RECALL, value=2 / 3),
        ],
    ),
    CategoryScore(
        name="movies",
        scores=[
            EvalScore(name=F1_SCORE, value=2 / 3),
            EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=PRECISION, value=1.0),
            EvalScore(name=RECALL, value=1 / 2),
        ],
    ),
    CategoryScore(
        name="sports",
        scores=[
            EvalScore(name=F1_SCORE, value=1.0),
            EvalScore(name=EXACT_MATCH_SCORE, value=1.0),
            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=1.0),
            EvalScore(name=PRECISION, value=1.0),
            EvalScore(name=RECALL, value=1.0),
        ],
    ),
    CategoryScore(
        name="history",
        scores=[
            EvalScore(name=F1_SCORE, value=2 / 5),
            EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=PRECISION, value=1 / 4),
            EvalScore(name=RECALL, value=1.0),
        ],
    ),
]

DATASET_SCORES = [
    EvalScore(name=F1_SCORE, value=61 / 90),
    EvalScore(name=EXACT_MATCH_SCORE, value=2 / 6),
    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=3 / 6),
    EvalScore(name=PRECISION, value=17 / 24),
    EvalScore(name=RECALL, value=3 / 4),
]

EVAL_RESULTS_PATH = DEFAULT_EVAL_RESULTS_PATH


class TestQAAccuracy:
    @fixture(scope="module")
    def config(self) -> QAAccuracyConfig:
        return QAAccuracyConfig(target_output_delimiter="<OR>")

    def test_qa_accuracy_invalid_config(self):
        """
        GIVEN empty string target_output_delimiter
        WHEN QAAccuracyConfig is initialized
        THEN correct exception with proper message is raised
        """
        expected_error_message = (
            "Empty target_output_delimiter is provided. " "Please either provide a non-empty string, or set it to None"
        )
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(expected_error_message)):
            QAAccuracyConfig(target_output_delimiter="")

    class TestCaseQAAccuracyEvaluateSample(NamedTuple):
        model_input: str
        model_output: str
        target_output: str
        expected_response: List[EvalScore]

    class TestCaseQAAccuracyEvaluateSampleInvalid(NamedTuple):
        model_input: Optional[str]
        model_output: str
        target_output: Optional[str]
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            # Exact match so all scores should have perfect values.
            TestCaseQAAccuracyEvaluateSample(
                model_input="What is the capital of England?",
                model_output="London",
                target_output="London",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=1.0),
                    EvalScore(name=EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=PRECISION, value=1.0),
                    EvalScore(name=RECALL, value=1.0),
                ],
            ),
            # Partial match
            TestCaseQAAccuracyEvaluateSample(
                model_input="Who directed Pulp Fiction",
                model_output="tarantino!",
                target_output="Quentin Tarantino",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=2 / 3),
                    EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=PRECISION, value=1.0),
                    EvalScore(name=RECALL, value=1 / 2),
                ],
            ),
            # Wrong answer. All scores should be zero.
            TestCaseQAAccuracyEvaluateSample(
                model_input="What is the capital of France?",
                model_output="London",
                target_output="Paris",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=0.0),
                    EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=PRECISION, value=0.0),
                    EvalScore(name=RECALL, value=0.0),
                ],
            ),
            # Correct answer but with punctuation added.
            TestCaseQAAccuracyEvaluateSample(
                model_input="What is the capital of Italy?",
                model_output="rome!",
                target_output="Rome",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=1.0),
                    EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=PRECISION, value=1.0),
                    EvalScore(name=RECALL, value=1.0),
                ],
            ),
            # Many correct answers.
            TestCaseQAAccuracyEvaluateSample(
                model_input="When did Argentina win the FIFA World Cup?",
                model_output="2022",
                target_output="1978<OR>1986<OR>2022",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=1.0),
                    EvalScore(name=EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=PRECISION, value=1.0),
                    EvalScore(name=RECALL, value=1.0),
                ],
            ),
            # Answer is longer than the model output.
            TestCaseQAAccuracyEvaluateSample(
                model_input="Did RMS Titanic sink in 1912?",
                model_output="Yes. That is true.",
                target_output="yes",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=0.4),
                    EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=PRECISION, value=0.25),
                    EvalScore(name=RECALL, value=1.0),
                ],
            ),
        ],
    )
    def test_qa_accuracy_evaluate_sample(self, test_case, config):
        """
        GIVEN valid inputs
        WHEN QAAccuracy.evaluate_sample is called
        THEN correct List of EvalScores is returned
        """
        eval_algorithm = QAAccuracy(config)
        actual_response = eval_algorithm.evaluate_sample(test_case.target_output, test_case.model_output)
        assert test_case.expected_response == actual_response

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracyEvaluateSampleInvalid(
                model_input="London is the capital of",
                model_output="England",
                target_output=None,
                expected_error_message="Missing required input: target_output, for QA Accuracy evaluate_sample",
            ),
            TestCaseQAAccuracyEvaluateSampleInvalid(
                model_input="Pulp Fiction was directed by",
                model_output=None,
                target_output="QUENTIN TARANTINO",
                expected_error_message="Missing required input: model_output, for QA Accuracy evaluate_sample",
            ),
        ],
    )
    def test_qa_accuracy_evaluate_sample_invalid_input(self, test_case, config):
        """
        GIVEN invalid inputs
        WHEN QAAccuracy.evaluate_sample is called
        THEN correct exception with proper message is raised
        """
        eval_algorithm = QAAccuracy(config)
        with pytest.raises(EvalAlgorithmClientError, match=test_case.expected_error_message):
            eval_algorithm.evaluate_sample(test_case.target_output, test_case.model_output)

    class TestCaseQAAccuracyEvaluate(NamedTuple):
        input_dataset: Dataset
        prompt_template: Optional[str]
        dataset_config: Optional[DataConfig]
        input_dataset_with_generated_model_output: Optional[Dataset]
        expected_response: List[EvalOutput]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracyEvaluate(
                input_dataset=QA_DATASET_WITHOUT_MODEL_OUTPUT,
                prompt_template=None,
                dataset_config=None,
                input_dataset_with_generated_model_output=QA_DATASET,
                expected_response=[
                    EvalOutput(
                        eval_name="qa_accuracy",
                        dataset_name=BOOLQ,
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[BOOLQ],
                        dataset_scores=DATASET_SCORES,
                        category_scores=CATEGORY_SCORES,
                        output_path="/tmp/eval_results/qa_accuracy_boolq.jsonl",
                    ),
                    EvalOutput(
                        eval_name="qa_accuracy",
                        dataset_name=TRIVIA_QA,
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[TRIVIA_QA],
                        dataset_scores=DATASET_SCORES,
                        category_scores=CATEGORY_SCORES,
                        output_path="/tmp/eval_results/qa_accuracy_trivia_qa.jsonl",
                    ),
                    EvalOutput(
                        eval_name="qa_accuracy",
                        dataset_name=NATURAL_QUESTIONS,
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[NATURAL_QUESTIONS],
                        dataset_scores=DATASET_SCORES,
                        category_scores=CATEGORY_SCORES,
                        output_path="/tmp/eval_results/qa_accuracy_natural_questions.jsonl",
                    ),
                ],
            ),
            TestCaseQAAccuracyEvaluate(
                input_dataset=QA_DATASET_WITHOUT_MODEL_OUTPUT,
                prompt_template="Answer: $feature",
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                input_dataset_with_generated_model_output=QA_DATASET,
                expected_response=[
                    EvalOutput(
                        eval_name="qa_accuracy",
                        dataset_name="my_custom_dataset",
                        prompt_template="Answer: $feature",
                        dataset_scores=DATASET_SCORES,
                        category_scores=CATEGORY_SCORES,
                        output_path="/tmp/eval_results/qa_accuracy_my_custom_dataset.jsonl",
                    )
                ],
            ),
            TestCaseQAAccuracyEvaluate(
                input_dataset=QA_DATASET_WITHOUT_CATEGORY_WITHOUT_MODEL_OUTPUT,
                prompt_template="Answer: $feature",
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                input_dataset_with_generated_model_output=QA_DATASET_WITHOUT_CATEGORY,
                expected_response=[
                    EvalOutput(
                        eval_name="qa_accuracy",
                        dataset_name="my_custom_dataset",
                        prompt_template="Answer: $feature",
                        dataset_scores=DATASET_SCORES,
                        category_scores=None,
                        output_path="/tmp/eval_results/qa_accuracy_my_custom_dataset.jsonl",
                    )
                ],
            ),
            TestCaseQAAccuracyEvaluate(
                input_dataset=QA_DATASET_WITHOUT_CATEGORY_WITHOUT_MODEL_OUTPUT,
                prompt_template=None,
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                input_dataset_with_generated_model_output=QA_DATASET_WITHOUT_CATEGORY,
                expected_response=[
                    EvalOutput(
                        eval_name="qa_accuracy",
                        dataset_name="my_custom_dataset",
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        dataset_scores=DATASET_SCORES,
                        category_scores=None,
                        output_path="/tmp/eval_results/qa_accuracy_my_custom_dataset.jsonl",
                    )
                ],
            ),
        ],
    )
    @patch("fmeval.model_runners.model_runner.ModelRunner")
    @patch("fmeval.eval_algorithms.qa_accuracy.get_dataset")
    @patch("fmeval.eval_algorithms.qa_accuracy.save_dataset")
    @patch("fmeval.eval_algorithms.qa_accuracy.generate_model_predict_response_for_dataset")
    def test_qa_accuracy_evaluate(
        self, generate_model_predict_response_for_dataset, save_dataset, get_dataset, model, test_case, config
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset without model_outputs, an input ModelRunner
            and request to save records with scores
        WHEN QAAccuracy.evaluate is called
        THEN correct EvalOutput is returned
        """
        get_dataset.return_value = test_case.input_dataset
        generate_model_predict_response_for_dataset.return_value = test_case.input_dataset_with_generated_model_output
        eval_algorithm = QAAccuracy(config)
        actual_response = eval_algorithm.evaluate(
            model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template, save=True
        )
        assert actual_response == test_case.expected_response
        assert save_dataset.called

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracyEvaluate(
                input_dataset=QA_DATASET,
                prompt_template=None,
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                input_dataset_with_generated_model_output=None,
                expected_response=[
                    EvalOutput(
                        eval_name="qa_accuracy",
                        dataset_name="my_custom_dataset",
                        prompt_template=None,
                        dataset_scores=DATASET_SCORES,
                        category_scores=CATEGORY_SCORES,
                        output_path="/tmp/eval_results/qa_accuracy_my_custom_dataset.jsonl",
                    )
                ],
            )
        ],
    )
    @patch("fmeval.eval_algorithms.qa_accuracy.get_dataset")
    @patch("fmeval.eval_algorithms.qa_accuracy.save_dataset")
    @patch("fmeval.eval_algorithms.qa_accuracy.generate_model_predict_response_for_dataset")
    def test_qa_accuracy_evaluate_without_model(
        self, generate_model_predict_response_for_dataset, save_dataset, get_dataset, test_case, config
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset with model_outputs,
            and no request to save records with scores
        WHEN QAACCURACY.evaluate() is called
        THEN correct EvalOutput is returned
        """
        get_dataset.return_value = test_case.input_dataset
        generate_model_predict_response_for_dataset.return_value = test_case.input_dataset_with_generated_model_output
        eval_algorithm = QAAccuracy(config)
        actual_response = eval_algorithm.evaluate(model=None, dataset_config=test_case.dataset_config)
        assert not generate_model_predict_response_for_dataset.called
        assert not save_dataset.called
        assert actual_response == test_case.expected_response

    class TestCaseQAAccuracyEvaluateInvalid(NamedTuple):
        input_dataset: Dataset
        dataset_config: Optional[DataConfig]
        prompt_template: Optional[str]
        model_provided: bool
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracyEvaluateInvalid(
                input_dataset=QA_DATASET_WITHOUT_MODEL_OUTPUT,
                dataset_config=None,
                prompt_template=None,
                model_provided=False,
                expected_error_message="No ModelRunner provided. ModelRunner is required for inference on model_inputs",
            ),
            TestCaseQAAccuracyEvaluateInvalid(
                input_dataset=QA_DATASET_WITHOUT_MODEL_OUTPUT,
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
            TestCaseQAAccuracyEvaluateInvalid(
                input_dataset=QA_DATASET_WITHOUT_TARGET_OUTPUT,
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
            TestCaseQAAccuracyEvaluateInvalid(
                input_dataset=QA_DATASET_WITHOUT_MODEL_INPUT,
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
    @patch("fmeval.eval_algorithms.qa_accuracy.get_dataset")
    def test_qa_accuracy_evaluate_invalid_input(self, get_dataset, model, test_case, config):
        """
        GIVEN invalid inputs
        WHEN QAAccuracy.evaluate is called
        THEN correct exception with proper message is raised
        """
        eval_algorithm = QAAccuracy(config)
        get_dataset.return_value = test_case.input_dataset
        if not test_case.model_provided:
            model = None
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            eval_algorithm.evaluate(
                model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template
            )

    class TestCaseQAAccuracyEvalScore(NamedTuple):
        model_output: str
        target_output: str
        expected_score: float
        strip_text: bool = True

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracyEvalScore(
                model_output="I live in New York!",
                target_output="i     live in new york.",
                strip_text=True,
                expected_score=1.0,
            ),
            TestCaseQAAccuracyEvalScore(
                model_output="This is a bad movie",
                target_output="This is a bad movie",
                strip_text=True,
                expected_score=1.0,
            ),
            TestCaseQAAccuracyEvalScore(
                model_output="Love this movie",
                target_output="Hated that film",
                strip_text=True,
                expected_score=0.0,
            ),
            TestCaseQAAccuracyEvalScore(
                model_output="yes.\n",
                target_output="yes",
                strip_text=True,
                expected_score=1.0,
            ),
            TestCaseQAAccuracyEvalScore(
                model_output="yes.\n",
                target_output="yes",
                strip_text=False,
                expected_score=0.0,
            ),
        ],
    )
    def test_f1_score(self, test_case):
        assert (
            _f1_score(
                model_output=test_case.model_output,
                target_output=test_case.target_output,
                normalize_text=True,
                strip_text=test_case.strip_text,
            )
            == test_case.expected_score
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracyEvalScore(
                model_output="I live in New York!",
                target_output="i     live in new york.",
                strip_text=True,
                expected_score=1.0,
            ),
            TestCaseQAAccuracyEvalScore(
                model_output="This is a bad movie",
                target_output="This is a bad movie",
                strip_text=True,
                expected_score=1.0,
            ),
            TestCaseQAAccuracyEvalScore(
                model_output="Love this movie",
                target_output="Hated that film",
                strip_text=True,
                expected_score=0.0,
            ),
            TestCaseQAAccuracyEvalScore(
                model_output="yes.\n",
                target_output="yes",
                strip_text=True,
                expected_score=1.0,
            ),
            TestCaseQAAccuracyEvalScore(
                model_output="yes.\n",
                target_output="yes",
                strip_text=False,
                expected_score=0.0,
            ),
        ],
    )
    def test_precision(self, test_case):
        assert (
            _precision(
                model_output=test_case.model_output,
                target_output=test_case.target_output,
                normalize_text=True,
                strip_text=test_case.strip_text,
            )
            == test_case.expected_score
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracyEvalScore(
                model_output="I live in New York!",
                target_output="i     live in new york.",
                strip_text=True,
                expected_score=1.0,
            ),
            TestCaseQAAccuracyEvalScore(
                model_output="This is a bad movie",
                target_output="This is a bad movie",
                strip_text=True,
                expected_score=1.0,
            ),
            TestCaseQAAccuracyEvalScore(
                model_output="Love this movie",
                target_output="Hated that film",
                strip_text=True,
                expected_score=0.0,
            ),
            TestCaseQAAccuracyEvalScore(
                model_output="yes.\n",
                target_output="yes",
                strip_text=True,
                expected_score=1.0,
            ),
            TestCaseQAAccuracyEvalScore(
                model_output="yes.\n",
                target_output="yes",
                strip_text=False,
                expected_score=0.0,
            ),
        ],
    )
    def test_recall(self, test_case):
        assert (
            _recall(
                model_output=test_case.model_output,
                target_output=test_case.target_output,
                normalize_text=True,
                strip_text=test_case.strip_text,
            )
            == test_case.expected_score
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracyEvalScore(
                model_output="I live in New York!",
                target_output="i     live in new york.",
                expected_score=0,
            ),
            TestCaseQAAccuracyEvalScore(
                model_output="This is a bad movie",
                target_output="This is a bad movie",
                expected_score=1,
            ),
            TestCaseQAAccuracyEvalScore(
                model_output="Love this movie",
                target_output="Hated this movie",
                expected_score=0,
            ),
        ],
    )
    def test_exact_match_score(self, test_case):
        assert (
            _exact_match_score(model_output=test_case.model_output, target_output=test_case.target_output)
            == test_case.expected_score
        )
