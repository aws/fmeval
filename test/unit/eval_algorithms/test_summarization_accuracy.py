import re
from typing import NamedTuple, List, Optional
from unittest.mock import patch

import pytest
import ray
from _pytest.fixtures import fixture
from ray.data import Dataset

from constants import (
    MIME_TYPE_JSON,
    MODEL_INPUT_COLUMN_NAME,
    TARGET_OUTPUT_COLUMN_NAME,
    CATEGORY_COLUMN_NAME,
    MODEL_OUTPUT_COLUMN_NAME,
)
from data_loaders.data_config import DataConfig
from eval_algorithms import CategoryScore, EvalOutput, EvalScore
from eval_algorithms.summarization_accuracy import (
    SummarizationAccuracyConfig,
    SummarizationAccuracy,
    METEOR_SCORE,
    ROUGE_SCORE,
    ROUGE_2,
    ROUGE_1,
    ROUGE_L,
    PROMPT_COLUMN_NAME,
)
from exceptions import EvalAlgorithmClientError

DATASET = ray.data.from_items(
    [
        {
            MODEL_INPUT_COLUMN_NAME: "Cake is so delicious, I really like cake. I want to open a bakery when I grow up.",
            TARGET_OUTPUT_COLUMN_NAME: "I like cake.",
            PROMPT_COLUMN_NAME: "Summarize: Cake is so delicious, I really like cake. I want to open a bakery when I "
            "grow up.",
            MODEL_OUTPUT_COLUMN_NAME: "I like cake.",
            CATEGORY_COLUMN_NAME: "dummy_category_1",
        },
        {
            MODEL_INPUT_COLUMN_NAME: "The art metropolis of Berlin inspires locals and visitors with its famous "
            "museum landscape and numerous UNESCO World Heritage sites."
            " It is also an international exhibition venue. "
            "You will find a selection of current and upcoming exhibitions here.",
            TARGET_OUTPUT_COLUMN_NAME: "Berlin: an art metropolis.",
            PROMPT_COLUMN_NAME: "Summarise: The art metropolis of Berlin inspires locals and visitors with its "
            "famous museum landscape and numerous UNESCO World Heritage sites."
            " It is also an international exhibition venue. "
            "You will find a selection of current and upcoming exhibitions here.",
            MODEL_OUTPUT_COLUMN_NAME: "Berlin: Art, Heritage, Exhibitions Hub.",
            CATEGORY_COLUMN_NAME: "dummy_category_2",
        },
        {
            MODEL_INPUT_COLUMN_NAME: "The art metropolis of Berlin inspires locals and visitors with its famous "
            "museum landscape and numerous UNESCO World Heritage sites."
            " It is also an international exhibition venue. "
            "You will find a selection of current and upcoming exhibitions here.",
            TARGET_OUTPUT_COLUMN_NAME: "Berlin: an art metropolis.",
            PROMPT_COLUMN_NAME: "Summarise: The art metropolis of Berlin inspires locals and visitors with its "
            "famous museum landscape and numerous UNESCO World Heritage sites."
            " It is also an international exhibition venue. "
            "You will find a selection of current and upcoming exhibitions here.",
            MODEL_OUTPUT_COLUMN_NAME: "Berlin: Art, Heritage, Exhibitions Hub.",
            CATEGORY_COLUMN_NAME: "dummy_category_1",
        },
    ]
)

DATASET_NO_CATEGORY = DATASET.drop_columns(cols=CATEGORY_COLUMN_NAME)

EVAL_RESULTS_PATH = "/tmp/eval_results/"


def assert_eval_output(expected_eval_output: EvalOutput, actual_eval_output: EvalOutput):
    """
    Assert Util for eval output, specifically for cases where eval scores are to be asserted with tolerance
    """
    assert actual_eval_output.eval_name == expected_eval_output.eval_name
    assert actual_eval_output.prompt_template == expected_eval_output.prompt_template
    assert actual_eval_output.dataset_name == expected_eval_output.dataset_name
    assert actual_eval_output.output_path == expected_eval_output.output_path

    def _assert_eval_scores(actual_eval_scores, expected_eval_scores):
        assert len(actual_eval_scores) == len(expected_eval_scores)
        for actual_eval_score, expected_eval_score in zip(actual_eval_scores, expected_eval_scores):
            assert actual_eval_score.name == expected_eval_score.name
            assert actual_eval_score.value == pytest.approx(expected_eval_score.value, rel=1e-5)

    _assert_eval_scores(actual_eval_output.dataset_scores, expected_eval_output.dataset_scores)

    if actual_eval_output.category_scores:
        assert len(actual_eval_output.category_scores) == len(expected_eval_output.category_scores)
        for actual_scores, expected_scores in zip(
            actual_eval_output.category_scores, expected_eval_output.category_scores
        ):
            assert actual_scores.name == expected_scores.name
            _assert_eval_scores(actual_scores.scores, expected_scores.scores)


class TestSummarizationAccuracy:
    @fixture(scope="module")
    def config(self) -> SummarizationAccuracyConfig:
        return SummarizationAccuracyConfig()

    class TestCaseSummarizationAccuracyEvaluateSample(NamedTuple):
        model_output: str
        target_output: str
        expected_response: List[EvalScore]
        rouge_type: str

    class TestCaseSummarizationAccuracyEvaluateSampleInvalid(NamedTuple):
        model_output: str
        target_output: Optional[str]
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSummarizationAccuracyEvaluateSample(
                model_output="I like cake.",
                target_output="I like cake.",
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=0.9921875),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                ],
                rouge_type=ROUGE_2,
            ),
            TestCaseSummarizationAccuracyEvaluateSample(
                model_output="Berlin: Art, Heritage, Exhibitions Hub.",
                target_output="Berlin: an art metropolis.",
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=0.5009920634920636),
                    EvalScore(name=ROUGE_SCORE, value=0.0),
                ],
                rouge_type=ROUGE_2,
            ),
            TestCaseSummarizationAccuracyEvaluateSample(
                model_output="I like cake.",
                target_output="I like cake.",
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=0.9921875),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                ],
                rouge_type=ROUGE_1,
            ),
            TestCaseSummarizationAccuracyEvaluateSample(
                model_output="Berlin: Art, Heritage, Exhibitions Hub.",
                target_output="Berlin: an art metropolis.",
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=0.5009920634920636),
                    EvalScore(name=ROUGE_SCORE, value=0.4444444444444445),
                ],
                rouge_type=ROUGE_1,
            ),
            TestCaseSummarizationAccuracyEvaluateSample(
                model_output="I like cake.",
                target_output="I like cake.",
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=0.9921875),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                ],
                rouge_type=ROUGE_L,
            ),
            TestCaseSummarizationAccuracyEvaluateSample(
                model_output="Berlin: Art, Heritage, Exhibitions Hub.",
                target_output="Berlin: an art metropolis.",
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=0.5009920634920636),
                    EvalScore(name=ROUGE_SCORE, value=0.4444444444444445),
                ],
                rouge_type=ROUGE_L,
            ),
        ],
    )
    def test_summarization_accuracy_evaluate_sample(self, test_case):
        """
        GIVEN valid inputs
        WHEN SummarizationAccuracy.evaluate_sample is called
        THEN correct List of EvalScores is returned
        """
        config = SummarizationAccuracyConfig(rouge_type=test_case.rouge_type)
        eval_algorithm = SummarizationAccuracy(config)
        actual_response = eval_algorithm.evaluate_sample(test_case.target_output, test_case.model_output)
        assert test_case.expected_response == actual_response

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSummarizationAccuracyEvaluateSampleInvalid(
                model_output="I like cake.",
                target_output=None,
                expected_error_message="Missing required input: target_output, for Summarization Accuracy "
                "evaluate_sample",
            ),
            TestCaseSummarizationAccuracyEvaluateSampleInvalid(
                model_output=None,
                target_output="I like cake.",
                expected_error_message="Missing required input: model_output, for Summarization Accuracy "
                "evaluate_sample",
            ),
        ],
    )
    def test_summarization_accuracy_evaluate_sample_invalid_input(self, test_case, config):
        """
        GIVEN invalid inputs
        WHEN SummarizationAccuracy.evaluate_sample is called
        THEN correct exception with proper message is raised
        """
        eval_algorithm = SummarizationAccuracy(config)
        with pytest.raises(EvalAlgorithmClientError, match=test_case.expected_error_message):
            eval_algorithm.evaluate_sample(test_case.target_output, test_case.model_output)

    def test_summarization_accuracy_invalid_config(self):
        expected_error_message = (
            "Invalid rouge_type: rouge3 requested in SummarizationAccuracyConfig, please choose "
            "from acceptable values: ['rouge1', 'rouge2', 'rougeL']"
        )
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(expected_error_message)):
            SummarizationAccuracyConfig(rouge_type="rouge3")

    class TestCaseSummarizationAccuracyEvaluate(NamedTuple):
        input_dataset: Dataset
        prompt_template: Optional[str]
        dataset_config: Optional[DataConfig]
        input_dataset_with_generated_model_output: Optional[Dataset]
        expected_response: List[EvalOutput]

    @pytest.mark.parametrize(
        "test_case",
        [
            # Built-in datasets evaluate for dataset without category
            TestCaseSummarizationAccuracyEvaluate(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(cols=[PROMPT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME]),
                dataset_config=None,
                prompt_template=None,
                input_dataset_with_generated_model_output=DATASET_NO_CATEGORY,
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template="Summarise: $feature",
                        dataset_name="cnn_daily_mail",
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.6647238756613758),
                            EvalScore(name="rouge", value=0.3333333333333333),
                        ],
                        category_scores=None,
                        output_path="/tmp/eval_results/",
                    ),
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template="Summarise: $feature",
                        dataset_name="xsum",
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.6647238756613758),
                            EvalScore(name="rouge", value=0.3333333333333333),
                        ],
                        category_scores=None,
                        output_path=EVAL_RESULTS_PATH,
                    ),
                ],
            ),
            # Built-in datasets evaluate for dataset with category
            TestCaseSummarizationAccuracyEvaluate(
                input_dataset=DATASET.drop_columns(cols=[PROMPT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME]),
                dataset_config=None,
                prompt_template=None,
                input_dataset_with_generated_model_output=DATASET,
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template="Summarise: $feature",
                        dataset_name="cnn_daily_mail",
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.6647238756613758),
                            EvalScore(name="rouge", value=0.3333333333333333),
                        ],
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1",
                                scores=[
                                    EvalScore(name="meteor", value=0.7465897817460319),
                                    EvalScore(name="rouge", value=0.5),
                                ],
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[
                                    EvalScore(name="meteor", value=0.5009920634920636),
                                    EvalScore(name="rouge", value=0.0),
                                ],
                            ),
                        ],
                        output_path="/tmp/eval_results/",
                    ),
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template="Summarise: $feature",
                        dataset_name="xsum",
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.6647238756613758),
                            EvalScore(name="rouge", value=0.3333333333333333),
                        ],
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1",
                                scores=[
                                    EvalScore(name="meteor", value=0.7465897817460319),
                                    EvalScore(name="rouge", value=0.5),
                                ],
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[
                                    EvalScore(name="meteor", value=0.5009920634920636),
                                    EvalScore(name="rouge", value=0.0),
                                ],
                            ),
                        ],
                        output_path=EVAL_RESULTS_PATH,
                    ),
                ],
            ),
            # Custom dataset evaluate
            TestCaseSummarizationAccuracyEvaluate(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(cols=[PROMPT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME]),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                prompt_template="Summarise: $feature",
                input_dataset_with_generated_model_output=DATASET_NO_CATEGORY,
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template="Summarise: $feature",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.6647238756613758),
                            EvalScore(name="rouge", value=0.3333333333333333),
                        ],
                        category_scores=None,
                        output_path="/tmp/eval_results/",
                    )
                ],
            ),
        ],
    )
    @patch("model_runners.model_runner.ModelRunner")
    @patch("eval_algorithms.summarization_accuracy.get_dataset")
    @patch("eval_algorithms.summarization_accuracy.save_dataset")
    @patch("eval_algorithms.summarization_accuracy.generate_model_predict_response_for_dataset")
    def test_summarization_accuracy_evaluate(
        self, generate_model_predict_response_for_dataset, save_dataset, get_dataset, model, test_case, config
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset without model_outputs, an input ModelRunner
            and request to save records with scores
        WHEN SummarizationAccuracy evaluate() method is called
        THEN correct EvalOutput is returned
        """
        get_dataset.return_value = test_case.input_dataset
        generate_model_predict_response_for_dataset.return_value = test_case.input_dataset_with_generated_model_output
        eval_algorithm = SummarizationAccuracy(config)
        actual_response = eval_algorithm.evaluate(
            model=model, dataset_config=test_case.dataset_config, save=True, prompt_template=test_case.prompt_template
        )
        assert save_dataset.called
        assert len(actual_response) == len(test_case.expected_response)
        for i in range(len(actual_response)):
            assert_eval_output(actual_response[i], test_case.expected_response[i])

    @pytest.mark.parametrize(
        "test_case",
        [
            # Built-in datasets evaluate for dataset without category
            TestCaseSummarizationAccuracyEvaluate(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(cols=[PROMPT_COLUMN_NAME]),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                prompt_template="Summarise: $feature",
                input_dataset_with_generated_model_output=DATASET_NO_CATEGORY,
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template="Summarise: $feature",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.6647238756613758),
                            EvalScore(name="rouge", value=0.3333333333333333),
                        ],
                        category_scores=None,
                        output_path="/tmp/eval_results/",
                    )
                ],
            ),
        ],
    )
    @patch("eval_algorithms.summarization_accuracy.get_dataset")
    @patch("eval_algorithms.summarization_accuracy.save_dataset")
    @patch("eval_algorithms.summarization_accuracy.generate_model_predict_response_for_dataset")
    def test_summarization_accuracy_evaluate_without_model(
        self, generate_model_predict_response_for_dataset, save_dataset, get_dataset, test_case, config
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset without model_outputs, an input ModelRunner
            and request to save records with scores
        WHEN SummarizationAccuracy evaluate() method is called
        THEN correct EvalOutput is returned
        """
        get_dataset.return_value = test_case.input_dataset
        eval_algorithm = SummarizationAccuracy(config)
        actual_response = eval_algorithm.evaluate(
            model=None, dataset_config=test_case.dataset_config, save=False, prompt_template=test_case.prompt_template
        )
        assert not save_dataset.called
        assert not generate_model_predict_response_for_dataset.called
        assert len(actual_response) == len(test_case.expected_response)
        for i in range(len(actual_response)):
            assert_eval_output(actual_response[i], test_case.expected_response[i])

    class TestCaseSummarizationAccuracyEvaluateInvalid(NamedTuple):
        input_dataset: Dataset
        dataset_config: Optional[DataConfig]
        prompt_template: Optional[str]
        model_provided: bool
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSummarizationAccuracyEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(cols=[PROMPT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME]),
                dataset_config=None,
                prompt_template=None,
                model_provided=False,
                expected_error_message="No ModelRunner provided. ModelRunner is required for inference on model_inputs",
            ),
            TestCaseSummarizationAccuracyEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(cols=[PROMPT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME]),
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
            TestCaseSummarizationAccuracyEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(
                    cols=[PROMPT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME, TARGET_OUTPUT_COLUMN_NAME]
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
            TestCaseSummarizationAccuracyEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(
                    cols=[PROMPT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME, MODEL_INPUT_COLUMN_NAME]
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
            TestCaseSummarizationAccuracyEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(cols=[PROMPT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME]),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                model_provided=True,
                prompt_template=None,
                expected_error_message="Missing required input: prompt_template for evaluating custom dataset :",
            ),
        ],
    )
    @patch("model_runners.model_runner.ModelRunner")
    @patch("eval_algorithms.summarization_accuracy.get_dataset")
    def test_summarization_accuracy_evaluate_invalid_input(self, get_dataset, model, test_case, config):
        """
        GIVEN invalid inputs
        WHEN SummarizationAccuracy.evaluate_sample is called
        THEN correct exception with proper message is raised
        """
        eval_algorithm = SummarizationAccuracy(config)
        get_dataset.return_value = test_case.input_dataset
        if not test_case.model_provided:
            model = None
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            eval_algorithm.evaluate(
                model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template
            )
