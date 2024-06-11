import re
from typing import NamedTuple, List, Optional
from unittest.mock import patch, Mock

import pytest
import ray
from _pytest.fixtures import fixture
from ray.data import Dataset

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON,
    DEFAULT_EVAL_RESULTS_PATH,
    MEAN,
)
from fmeval.data_loaders.data_config import DataConfig
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
    PRECISION_OVER_WORDS,
    RECALL_OVER_WORDS,
    _f1_score,
    _exact_match_score,
    _precision,
    _recall,
    _split,
    SCORE_NAMES,
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
            EvalScore(name=PRECISION_OVER_WORDS, value=2 / 3),
            EvalScore(name=RECALL_OVER_WORDS, value=2 / 3),
        ],
    ),
    CategoryScore(
        name="movies",
        scores=[
            EvalScore(name=F1_SCORE, value=2 / 3),
            EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=PRECISION_OVER_WORDS, value=1.0),
            EvalScore(name=RECALL_OVER_WORDS, value=1 / 2),
        ],
    ),
    CategoryScore(
        name="sports",
        scores=[
            EvalScore(name=F1_SCORE, value=1.0),
            EvalScore(name=EXACT_MATCH_SCORE, value=1.0),
            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=1.0),
            EvalScore(name=PRECISION_OVER_WORDS, value=1.0),
            EvalScore(name=RECALL_OVER_WORDS, value=1.0),
        ],
    ),
    CategoryScore(
        name="history",
        scores=[
            EvalScore(name=F1_SCORE, value=2 / 5),
            EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=PRECISION_OVER_WORDS, value=1 / 4),
            EvalScore(name=RECALL_OVER_WORDS, value=1.0),
        ],
    ),
]

DATASET_SCORES = [
    EvalScore(name=F1_SCORE, value=61 / 90),
    EvalScore(name=EXACT_MATCH_SCORE, value=2 / 6),
    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=3 / 6),
    EvalScore(name=PRECISION_OVER_WORDS, value=17 / 24),
    EvalScore(name=RECALL_OVER_WORDS, value=3 / 4),
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
                    EvalScore(name=PRECISION_OVER_WORDS, value=1.0),
                    EvalScore(name=RECALL_OVER_WORDS, value=1.0),
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
                    EvalScore(name=PRECISION_OVER_WORDS, value=1.0),
                    EvalScore(name=RECALL_OVER_WORDS, value=1 / 2),
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
                    EvalScore(name=PRECISION_OVER_WORDS, value=0.0),
                    EvalScore(name=RECALL_OVER_WORDS, value=0.0),
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
                    EvalScore(name=PRECISION_OVER_WORDS, value=1.0),
                    EvalScore(name=RECALL_OVER_WORDS, value=1.0),
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
                    EvalScore(name=PRECISION_OVER_WORDS, value=1.0),
                    EvalScore(name=RECALL_OVER_WORDS, value=1.0),
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
                    EvalScore(name=PRECISION_OVER_WORDS, value=0.25),
                    EvalScore(name=RECALL_OVER_WORDS, value=1.0),
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

    @patch("fmeval.eval_algorithms.qa_accuracy.get_eval_results_path")
    @patch("fmeval.eval_algorithms.qa_accuracy.evaluate_dataset")
    @patch("fmeval.eval_algorithms.qa_accuracy.TransformPipeline")
    @patch("fmeval.eval_algorithms.qa_accuracy.get_dataset")
    @patch("fmeval.eval_algorithms.qa_accuracy.get_dataset_configs")
    def test_evaluate(
        self,
        mock_get_dataset_configs,
        mock_get_dataset,
        mock_transform_pipeline_cls,
        mock_evaluate_dataset,
        mock_get_results_path,
    ):
        """
        GIVEN a QAAccuracy instance.
        WHEN its evaluate method is called with valid arguments.
        THEN `evaluate_dataset` is called with the correct arguments.
        """
        dataset_config = Mock()
        dataset_config.dataset_name = "my_custom_dataset"
        mock_get_dataset_configs.return_value = [dataset_config]

        mock_dataset = Mock()
        # So that validate_dataset does not error
        mock_dataset.columns = Mock(return_value=[DatasetColumns.TARGET_OUTPUT.value.name])
        mock_get_dataset.return_value = mock_dataset

        mock_get_results_path.return_value = "/path/to/results"
        model_runner = Mock()

        qa_acc = QAAccuracy()
        output = qa_acc.evaluate(
            model=model_runner,
            dataset_config=dataset_config,
            prompt_template="Answer $model_input, please.",
            num_records=162,
            save=True,
        )

        mock_transform_pipeline_cls.assert_called_once_with([qa_acc.transform])

        mock_evaluate_dataset.assert_called_once_with(
            dataset=mock_dataset,
            pipeline=mock_transform_pipeline_cls.return_value,
            dataset_name=dataset_config.dataset_name,
            eval_name=qa_acc.eval_name,
            metric_names=SCORE_NAMES,
            eval_results_path="/path/to/results",
            model=model_runner,
            prompt_template="Answer $model_input, please.",
            agg_method=MEAN,
            save=True,
        )

        assert output == [mock_evaluate_dataset.return_value]

    class TestCaseQAAccuracyEvaluate(NamedTuple):
        input_dataset: Dataset
        dataset_config: Optional[DataConfig]
        expected_response: List[EvalOutput]

    @pytest.mark.parametrize(
        "test_case",
        [
            # Dataset with category
            TestCaseQAAccuracyEvaluate(
                input_dataset=QA_DATASET,
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
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
            ),
            # Dataset without category
            TestCaseQAAccuracyEvaluate(
                input_dataset=QA_DATASET_WITHOUT_CATEGORY,
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                expected_response=[
                    EvalOutput(
                        eval_name="qa_accuracy",
                        dataset_name="my_custom_dataset",
                        prompt_template=None,
                        dataset_scores=DATASET_SCORES,
                        category_scores=None,
                        output_path="/tmp/eval_results/qa_accuracy_my_custom_dataset.jsonl",
                    )
                ],
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.qa_accuracy.get_dataset")
    def test_qa_accuracy_evaluate(self, mock_get_dataset, test_case, config):
        """
        GIVEN dataset(s) with model outputs already present.
        WHEN the QAAccuracy evaluate method is called.
        THEN the correct EvalOutput is returned.

        Note: this is basically an integration test rather than a unit test like `test_evaluate` above.
        It uses a special toy dataset where we are able to compute the exact expected scores by hand.
        The purpose of this test is really to ensure that the correct scores are being generated.
        """
        mock_get_dataset.return_value = test_case.input_dataset
        eval_algorithm = QAAccuracy(config)
        actual_response = eval_algorithm.evaluate(
            dataset_config=test_case.dataset_config,
            save=True,
        )
        assert actual_response == test_case.expected_response

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
                expected_score=1.0,
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
                expected_score=1.0,
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
                model_output="\n\nyes.\n",
                target_output="yes",
                strip_text=False,
                expected_score=1.0,
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

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("True\n\nI do agree.\tTrue", {"True", "I", "do", "agree."}),
            ("\n\n   \n\n", set()),
            ("I\n\n\n     am you\n", {"I", "am", "you"}),
            ("\r\x0bI\x0csaw\t\n\ryou!", {"I", "saw", "you!"}),
        ],
    )  #
    def test_split(self, text, expected):
        """
        GIVEN text as string
        WHEN _split is called
        THEN returns a set of strings as expected (namley split by ' \t\n\r\x0b\x0c')
        """
        ans = _split(text)
        assert ans == expected
