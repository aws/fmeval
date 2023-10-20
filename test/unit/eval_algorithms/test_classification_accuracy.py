import re
from typing import NamedTuple, List, Optional, Union
from unittest.mock import patch

import pytest
import ray
from _pytest.fixtures import fixture

from amazon_fmeval.constants import (
    MIME_TYPE_JSON,
    MODEL_OUTPUT_COLUMN_NAME,
    TARGET_OUTPUT_COLUMN_NAME,
    CATEGORY_COLUMN_NAME,
    MODEL_INPUT_COLUMN_NAME,
    DEFAULT_EVAL_RESULTS_PATH,
)

from amazon_fmeval.data_loaders.util import DataConfig
from amazon_fmeval.eval_algorithms import (
    EvalOutput,
    EvalScore,
    CategoryScore,
    DEFAULT_PROMPT_TEMPLATE,
    BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES,
    IMDB_MOVIE_REVIEWS,
)
from amazon_fmeval.eval_algorithms.classification_accuracy import (
    ClassificationAccuracyConfig,
    ClassificationAccuracy,
    convert_model_output_to_label,
    CLASSIFICATION_ACCURACY_SCORE,
    BALANCED_ACCURACY_SCORE,
    PRECISION_SCORE,
    RECALL_SCORE,
)
from ray.data import Dataset

from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.util import EVAL_RESULTS_PATH

CLASSIFICATION_DATASET = ray.data.from_items(
    [
        {
            MODEL_INPUT_COLUMN_NAME: "Delicious cake! Would buy again.",
            TARGET_OUTPUT_COLUMN_NAME: "4",
            CATEGORY_COLUMN_NAME: "brownie",
            MODEL_OUTPUT_COLUMN_NAME: "4",
        },
        {
            MODEL_INPUT_COLUMN_NAME: "Tasty cake! Must eat.",
            TARGET_OUTPUT_COLUMN_NAME: "4",
            CATEGORY_COLUMN_NAME: "vanilla cake",
            MODEL_OUTPUT_COLUMN_NAME: "4",
        },
        {
            MODEL_INPUT_COLUMN_NAME: "Terrible! Nightmarish cake.",
            TARGET_OUTPUT_COLUMN_NAME: "1",
            CATEGORY_COLUMN_NAME: "vanilla cake",
            MODEL_OUTPUT_COLUMN_NAME: "2",
        },
    ]
)


CLASSIFICATION_DATASET_WITHOUT_MODEL_OUTPUT = CLASSIFICATION_DATASET.drop_columns(MODEL_OUTPUT_COLUMN_NAME)

CLASSIFICATION_DATASET_WITHOUT_MODEL_INPUT = CLASSIFICATION_DATASET.drop_columns(MODEL_INPUT_COLUMN_NAME)

CLASSIFICATION_DATASET_WITHOUT_TARGET_OUTPUT = CLASSIFICATION_DATASET.drop_columns(TARGET_OUTPUT_COLUMN_NAME)

CLASSIFICATION_DATASET_WITHOUT_CATEGORY_WITHOUT_MODEL_OUTPUT = CLASSIFICATION_DATASET_WITHOUT_MODEL_OUTPUT.drop_columns(
    CATEGORY_COLUMN_NAME
)

CLASSIFICATION_DATASET_WITHOUT_CATEGORY = CLASSIFICATION_DATASET.drop_columns(CATEGORY_COLUMN_NAME)

DATASET_SCORES = [
    EvalScore(name=CLASSIFICATION_ACCURACY_SCORE, value=2 / 3),
    EvalScore(name=BALANCED_ACCURACY_SCORE, value=1 / 2),
    EvalScore(name=PRECISION_SCORE, value=2 / 3),
    EvalScore(name=RECALL_SCORE, value=2 / 3),
]

CATEGORY_SCORES = [
    CategoryScore(
        name="brownie",
        scores=[
            EvalScore(name=CLASSIFICATION_ACCURACY_SCORE, value=1.0),
            EvalScore(name=BALANCED_ACCURACY_SCORE, value=1.0),
            EvalScore(name=PRECISION_SCORE, value=1.0),
            EvalScore(name=RECALL_SCORE, value=1.0),
        ],
    ),
    CategoryScore(
        name="vanilla cake",
        scores=[
            EvalScore(name=CLASSIFICATION_ACCURACY_SCORE, value=1 / 2),
            EvalScore(name=BALANCED_ACCURACY_SCORE, value=1 / 2),
            EvalScore(name=PRECISION_SCORE, value=1 / 2),
            EvalScore(name=RECALL_SCORE, value=1 / 2),
        ],
    ),
]


class TestClassificationAccuracy:
    @fixture(scope="module")
    def config(self) -> ClassificationAccuracyConfig:
        return ClassificationAccuracyConfig(valid_labels=["1", "2", "3", "4", "5"])

    def test_classification_accuracy_config_format_with_castable_labels(self):
        """
        GIVEN valid labels are int but can be cast to str
        WHEN ClassificationAccuracyConfig is initialized with castable integer labels
        THEN warning is raised, ClassificationAccuracyConfig is initialized successfully
        """
        with pytest.warns():
            castable_config = ClassificationAccuracyConfig(valid_labels=[1, 2, 3, 4, 5])
            assert castable_config.valid_labels == ["1", "2", "3", "4", "5"]

    class TestCaseClassificationAccuracyEvaluate(NamedTuple):
        input_dataset: Dataset
        prompt_template: Optional[str]
        dataset_config: Optional[DataConfig]
        input_dataset_with_generated_model_output: Optional[Dataset]
        expected_response: List[EvalOutput]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseClassificationAccuracyEvaluate(
                input_dataset=CLASSIFICATION_DATASET,
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
                input_dataset_with_generated_model_output=CLASSIFICATION_DATASET,
                expected_response=[
                    EvalOutput(
                        eval_name="classification_accuracy",
                        dataset_name="my_custom_dataset",
                        prompt_template=None,
                        dataset_scores=DATASET_SCORES,
                        category_scores=CATEGORY_SCORES,
                        output_path=EVAL_RESULTS_PATH,
                    )
                ],
            )
        ],
    )
    @patch("amazon_fmeval.eval_algorithms.classification_accuracy.get_dataset")
    @patch("amazon_fmeval.eval_algorithms.classification_accuracy.save_dataset")
    @patch("amazon_fmeval.eval_algorithms.classification_accuracy.generate_model_predict_response_for_dataset")
    def test_classification_accuracy_evaluate_without_model(
        self, generate_model_predict_response_for_dataset, save_dataset, get_dataset, test_case, config
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset with model_outputs,
            and no request to save records with scores
        WHEN ClassificationAccuracy.evaluate() is called
        THEN correct EvalOutput is returned
        """
        get_dataset.return_value = test_case.input_dataset
        generate_model_predict_response_for_dataset.return_value = test_case.input_dataset_with_generated_model_output
        eval_algorithm = ClassificationAccuracy(config)
        actual_response = eval_algorithm.evaluate(model=None, dataset_config=test_case.dataset_config)
        assert not generate_model_predict_response_for_dataset.called
        assert not save_dataset.called
        assert actual_response == test_case.expected_response

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseClassificationAccuracyEvaluate(
                input_dataset=CLASSIFICATION_DATASET_WITHOUT_MODEL_OUTPUT,
                prompt_template=None,
                dataset_config=None,
                input_dataset_with_generated_model_output=CLASSIFICATION_DATASET,
                expected_response=[
                    EvalOutput(
                        eval_name="classification_accuracy",
                        dataset_name="imdb_movie_reviews",
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[IMDB_MOVIE_REVIEWS],
                        dataset_scores=DATASET_SCORES,
                        category_scores=CATEGORY_SCORES,
                        output_path=DEFAULT_EVAL_RESULTS_PATH,
                    ),
                    # EvalOutput(
                    #     eval_name="classification_accuracy",
                    #     dataset_name="womens_clothing_ecommerce_reviews",
                    #     prompt_template="$feature",
                    #     dataset_scores=DATASET_SCORES,
                    #     category_scores=CATEGORY_SCORES,
                    #     output_path=EVAL_RESULTS_PATH,
                    # ),
                ],
            ),
            TestCaseClassificationAccuracyEvaluate(
                input_dataset=CLASSIFICATION_DATASET_WITHOUT_MODEL_OUTPUT,
                prompt_template="Classify: $feature",
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                input_dataset_with_generated_model_output=CLASSIFICATION_DATASET,
                expected_response=[
                    EvalOutput(
                        eval_name="classification_accuracy",
                        dataset_name="my_custom_dataset",
                        prompt_template="Classify: $feature",
                        dataset_scores=DATASET_SCORES,
                        category_scores=CATEGORY_SCORES,
                        output_path=EVAL_RESULTS_PATH,
                    )
                ],
            ),
            TestCaseClassificationAccuracyEvaluate(
                input_dataset=CLASSIFICATION_DATASET_WITHOUT_CATEGORY_WITHOUT_MODEL_OUTPUT,
                prompt_template="Classify: $feature",
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                input_dataset_with_generated_model_output=CLASSIFICATION_DATASET_WITHOUT_CATEGORY,
                expected_response=[
                    EvalOutput(
                        eval_name="classification_accuracy",
                        dataset_name="my_custom_dataset",
                        prompt_template="Classify: $feature",
                        dataset_scores=DATASET_SCORES,
                        category_scores=None,
                        output_path=EVAL_RESULTS_PATH,
                    )
                ],
            ),
            TestCaseClassificationAccuracyEvaluate(
                input_dataset=CLASSIFICATION_DATASET_WITHOUT_CATEGORY_WITHOUT_MODEL_OUTPUT,
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
                input_dataset_with_generated_model_output=CLASSIFICATION_DATASET_WITHOUT_CATEGORY,
                expected_response=[
                    EvalOutput(
                        eval_name="classification_accuracy",
                        dataset_name="my_custom_dataset",
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        dataset_scores=DATASET_SCORES,
                        category_scores=None,
                        output_path=EVAL_RESULTS_PATH,
                    )
                ],
            ),
        ],
    )
    @patch("amazon_fmeval.model_runners.model_runner.ModelRunner")
    @patch("amazon_fmeval.eval_algorithms.classification_accuracy.get_dataset")
    @patch("amazon_fmeval.eval_algorithms.classification_accuracy.save_dataset")
    @patch("amazon_fmeval.eval_algorithms.classification_accuracy.generate_model_predict_response_for_dataset")
    def test_classification_accuracy_evaluate(
        self, generate_model_predict_response_for_dataset, save_dataset, get_dataset, model, test_case, config
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset without model_outputs, an input ModelRunner
            and request to save records with scores
        WHEN ClassificationAccuracy.evaluate is called
        THEN correct EvalOutput is returned
        """
        get_dataset.return_value = test_case.input_dataset
        generate_model_predict_response_for_dataset.return_value = test_case.input_dataset_with_generated_model_output
        eval_algorithm = ClassificationAccuracy(config)
        actual_response = eval_algorithm.evaluate(
            model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template, save=True
        )
        assert actual_response == test_case.expected_response
        assert save_dataset.called

    class TestCaseClassificationAccuracyEvaluateInvalid(NamedTuple):
        input_dataset: Dataset
        dataset_config: Optional[DataConfig]
        prompt_template: Optional[str]
        model_provided: bool
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseClassificationAccuracyEvaluateInvalid(
                input_dataset=CLASSIFICATION_DATASET_WITHOUT_MODEL_OUTPUT,
                dataset_config=None,
                prompt_template=None,
                model_provided=False,
                expected_error_message="No ModelRunner provided. ModelRunner is required for inference on model_inputs",
            ),
            TestCaseClassificationAccuracyEvaluateInvalid(
                input_dataset=CLASSIFICATION_DATASET_WITHOUT_MODEL_OUTPUT,
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
            TestCaseClassificationAccuracyEvaluateInvalid(
                input_dataset=CLASSIFICATION_DATASET_WITHOUT_TARGET_OUTPUT,
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
            TestCaseClassificationAccuracyEvaluateInvalid(
                input_dataset=CLASSIFICATION_DATASET_WITHOUT_MODEL_INPUT,
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
    @patch("amazon_fmeval.model_runners.model_runner.ModelRunner")
    @patch("amazon_fmeval.eval_algorithms.classification_accuracy.get_dataset")
    def test_classification_accuracy_evaluate_invalid_input(self, get_dataset, model, test_case, config):
        """
        GIVEN invalid inputs
        WHEN ClassificationAccuracy.evaluate is called
        THEN correct exception with proper message is raised
        """
        eval_algorithm = ClassificationAccuracy(config)
        get_dataset.return_value = test_case.input_dataset
        if not test_case.model_provided:
            model = None
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            eval_algorithm.evaluate(
                model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template
            )

    class TestCaseClassificationAccuracyEvaluateSample(NamedTuple):
        model_input: str
        model_output: str
        target_output: str
        expected_response: List[EvalScore]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseClassificationAccuracyEvaluateSample(
                model_input="Delicious cake! Would buy again.",
                model_output="4",
                target_output="4",
                expected_response=[
                    EvalScore(name=CLASSIFICATION_ACCURACY_SCORE, value=1.0),
                ],
            ),
            TestCaseClassificationAccuracyEvaluateSample(
                model_input="Terrible! Nightmarish cake.",
                model_output="1",
                target_output="2",
                expected_response=[
                    EvalScore(name=CLASSIFICATION_ACCURACY_SCORE, value=0),
                ],
            ),
        ],
    )
    def test_classification_accuracy_evaluate_sample(self, test_case, config):
        """
        GIVEN valid inputs
        WHEN ClassificationAccuracy.evaluate_sample is called
        THEN correct List of EvalScores is returned
        """
        eval_algorithm = ClassificationAccuracy(config)
        actual_response = eval_algorithm.evaluate_sample(test_case.target_output, test_case.model_output)
        assert test_case.expected_response == actual_response

    class TestCaseClassificationAccuracyEvaluateSampleInvalid(NamedTuple):
        model_input: Optional[str]
        model_output: str
        target_output: Optional[str]
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseClassificationAccuracyEvaluateSampleInvalid(
                model_input="Delicious cake! Would buy again.",
                model_output="4",
                target_output=None,
                expected_error_message="Missing required input: target_output, for Classification Accuracy evaluate_sample",
            ),
            TestCaseClassificationAccuracyEvaluateSampleInvalid(
                model_input="Tasty cake! Absolutely must eat.",
                model_output=None,
                target_output="4",
                expected_error_message="Missing required input: model_output, for Classification Accuracy evaluate_sample",
            ),
        ],
    )
    def test_classification_accuracy_evaluate_sample_invalid_input(self, test_case, config):
        """
        GIVEN invalid inputs
        WHEN ClassificationAccuracy.evaluate_sample is called
        THEN correct exception with proper message is raised
        """
        eval_algorithm = ClassificationAccuracy(config)
        with pytest.raises(EvalAlgorithmClientError, match=test_case.expected_error_message):
            eval_algorithm.evaluate_sample(test_case.target_output, test_case.model_output)

    class TestCaseLabelConversion(NamedTuple):
        model_output: Union[str, int]
        valid_labels: List[str]
        expected_label: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseLabelConversion(model_output="1", valid_labels=["0", "1"], expected_label="1"),
            TestCaseLabelConversion(model_output="2", valid_labels=["0", "1"], expected_label="unknown"),
            TestCaseLabelConversion(model_output="The answer is 1", valid_labels=["0", "1"], expected_label="1"),
            TestCaseLabelConversion(model_output="The answer is 1 or 0", valid_labels=["0", "1"], expected_label="1"),
            TestCaseLabelConversion(model_output="Bad", valid_labels=["1"], expected_label="unknown"),
            # check that lowercasing & stripping model output works
            TestCaseLabelConversion(model_output="One", valid_labels=["zero", "one"], expected_label="one"),
            # check that lowercasing & stripping label works
            TestCaseLabelConversion(model_output="one", valid_labels=["zero", "One"], expected_label="one"),
        ],
    )
    def test_convert_model_output_to_label(self, test_case):
        """
        GIVEN model output and valid labels
        WHEN convert_model_output_to_label is called
        THEN correct string label is returned
        """
        assert (
            convert_model_output_to_label(model_output=test_case.model_output, valid_labels=test_case.valid_labels)
            == test_case.expected_label
        )
