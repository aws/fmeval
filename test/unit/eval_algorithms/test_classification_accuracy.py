import re
from typing import NamedTuple, List, Optional, Union
from unittest.mock import patch, Mock

import pytest
import ray
from _pytest.fixtures import fixture

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON,
)

from fmeval.data_loaders.util import DataConfig
from fmeval.eval_algorithms import (
    EvalOutput,
    EvalScore,
    CategoryScore,
    DEFAULT_PROMPT_TEMPLATE,
    BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES,
    WOMENS_CLOTHING_ECOMMERCE_REVIEWS,
)
from fmeval.eval_algorithms.classification_accuracy import (
    ClassificationAccuracyConfig,
    ClassificationAccuracy,
    convert_model_output_to_label,
    CLASSIFICATION_ACCURACY_SCORE,
    BALANCED_ACCURACY_SCORE,
    PRECISION_SCORE,
    RECALL_SCORE,
    ClassificationAccuracyScores,
    CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME,
)
from ray.data import Dataset

from fmeval.exceptions import EvalAlgorithmClientError

CLASSIFICATION_DATASET = ray.data.from_items(
    [
        {
            DatasetColumns.MODEL_INPUT.value.name: "Delicious cake! Would buy again.",
            DatasetColumns.TARGET_OUTPUT.value.name: "4",
            DatasetColumns.CATEGORY.value.name: "brownie",
            DatasetColumns.MODEL_OUTPUT.value.name: "4",
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "Tasty cake! Must eat.",
            DatasetColumns.TARGET_OUTPUT.value.name: "4",
            DatasetColumns.CATEGORY.value.name: "vanilla cake",
            DatasetColumns.MODEL_OUTPUT.value.name: "4",
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "Terrible! Nightmarish cake.",
            DatasetColumns.TARGET_OUTPUT.value.name: "1",
            DatasetColumns.CATEGORY.value.name: "vanilla cake",
            DatasetColumns.MODEL_OUTPUT.value.name: "2",
        },
    ]
)


CLASSIFICATION_DATASET_WITHOUT_MODEL_OUTPUT = CLASSIFICATION_DATASET.drop_columns(
    cols=[DatasetColumns.MODEL_OUTPUT.value.name]
)

CLASSIFICATION_DATASET_WITHOUT_MODEL_INPUT = CLASSIFICATION_DATASET.drop_columns(
    cols=[DatasetColumns.MODEL_INPUT.value.name]
)

CLASSIFICATION_DATASET_WITHOUT_MODEL_INPUT_OR_MODEL_OUTPUT = CLASSIFICATION_DATASET_WITHOUT_MODEL_INPUT.drop_columns(
    cols=[DatasetColumns.MODEL_OUTPUT.value.name]
)

CLASSIFICATION_DATASET_WITHOUT_TARGET_OUTPUT = CLASSIFICATION_DATASET.drop_columns(
    cols=[DatasetColumns.TARGET_OUTPUT.value.name]
)

CLASSIFICATION_DATASET_WITHOUT_CATEGORY_WITHOUT_MODEL_OUTPUT = CLASSIFICATION_DATASET_WITHOUT_MODEL_OUTPUT.drop_columns(
    cols=[DatasetColumns.CATEGORY.value.name]
)

CLASSIFICATION_DATASET_WITHOUT_CATEGORY = CLASSIFICATION_DATASET.drop_columns(cols=[DatasetColumns.CATEGORY.value.name])

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
                        output_path="/tmp/eval_results/classification_accuracy_my_custom_dataset.jsonl",
                    )
                ],
            )
        ],
    )
    @patch("fmeval.eval_algorithms.classification_accuracy.create_model_invocation_pipeline")
    @patch("fmeval.eval_algorithms.classification_accuracy.save_dataset")
    @patch("fmeval.eval_algorithms.classification_accuracy.get_dataset")
    def test_classification_accuracy_evaluate_without_model(
        self, get_dataset, save_dataset, create_model_invocation_pipeline, test_case, config
    ):
        """
        GIVEN an input dataset with model outputs, no model,
            and a `save` argument of False.
        WHEN ClassificationAccuracy.evaluate is called.
        THEN the correct EvalOutput is returned, and neither save_dataset
            nor create_model_invocation_pipeline is called.
        """
        get_dataset.return_value = test_case.input_dataset_with_generated_model_output
        eval_algorithm = ClassificationAccuracy(config)
        actual_response = eval_algorithm.evaluate(model=None, dataset_config=test_case.dataset_config)
        create_model_invocation_pipeline.assert_not_called()
        save_dataset.assert_not_called()
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
                        dataset_name=WOMENS_CLOTHING_ECOMMERCE_REVIEWS,
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[WOMENS_CLOTHING_ECOMMERCE_REVIEWS],
                        dataset_scores=DATASET_SCORES,
                        category_scores=CATEGORY_SCORES,
                        output_path="/tmp/eval_results/classification_accuracy_womens_clothing_ecommerce_reviews.jsonl",
                    ),
                ],
            ),
            TestCaseClassificationAccuracyEvaluate(
                input_dataset=CLASSIFICATION_DATASET_WITHOUT_MODEL_OUTPUT,
                prompt_template="Classify: $model_input",
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
                        prompt_template="Classify: $model_input",
                        dataset_scores=DATASET_SCORES,
                        category_scores=CATEGORY_SCORES,
                        output_path="/tmp/eval_results/classification_accuracy_my_custom_dataset.jsonl",
                    )
                ],
            ),
            TestCaseClassificationAccuracyEvaluate(
                input_dataset=CLASSIFICATION_DATASET_WITHOUT_CATEGORY_WITHOUT_MODEL_OUTPUT,
                prompt_template="Classify: $model_input",
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
                        prompt_template="Classify: $model_input",
                        dataset_scores=DATASET_SCORES,
                        category_scores=None,
                        output_path="/tmp/eval_results/classification_accuracy_my_custom_dataset.jsonl",
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
                        output_path="/tmp/eval_results/classification_accuracy_my_custom_dataset.jsonl",
                    )
                ],
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.classification_accuracy.save_dataset")
    @patch("fmeval.eval_algorithms.classification_accuracy.get_dataset")
    def test_classification_accuracy_evaluate(self, get_dataset, save_dataset, test_case):
        """
        GIVEN an input dataset without model outputs, a ModelRunner,
            and `save` argument of True.
        WHEN ClassificationAccuracy.evaluate is called.
        THEN correct EvalOutput is returned.
        """
        model_runner = Mock()
        # We mock the behavior of generating the model output column in
        # CLASSIFICATION_DATASET from CLASSIFICATION_DATASET_WITHOUT_MODEL_OUTPUT.
        model_runner.predict.side_effect = [("4", None), ("4", None), ("2", None)]

        get_dataset.return_value = test_case.input_dataset

        eval_algorithm = ClassificationAccuracy()
        actual_response = eval_algorithm.evaluate(
            model=model_runner,
            dataset_config=test_case.dataset_config,
            prompt_template=test_case.prompt_template,
            save=True,
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
            # Assert that model is provided when built-in datasets are used
            TestCaseClassificationAccuracyEvaluateInvalid(
                input_dataset=CLASSIFICATION_DATASET_WITHOUT_MODEL_OUTPUT,
                dataset_config=None,
                prompt_template=None,
                model_provided=False,
                expected_error_message="No ModelRunner provided. ModelRunner is required for inference on model_inputs",
            ),
            # Assert that model is provided when a dataset without model output col is provided
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
            # Assert that target output column is present in dataset.
            # This applies for both when a model is provided and when it's not.
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
                model_provided=False,
                expected_error_message="Missing required column: target_output, for evaluate() method",
            ),
            # Assert that the dataset contains a model input column, only if
            # the dataset doesn't contain a model output column.
            TestCaseClassificationAccuracyEvaluateInvalid(
                input_dataset=CLASSIFICATION_DATASET_WITHOUT_MODEL_INPUT_OR_MODEL_OUTPUT,
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
    @patch("fmeval.eval_algorithms.classification_accuracy.get_dataset")
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
        actual_response = eval_algorithm.evaluate_sample(
            test_case.target_output,
            test_case.model_output,
        )
        assert test_case.expected_response == actual_response

    def test_classification_accuracy_evaluate_sample_missing_valid_labels(self):
        """
        GIVEN a ClassificationAccuracy where its `valid_labels` attribute is not set.
        WHEN its evaluate_sample method is called.
        THEN the correct exception is raised.
        """
        err_msg = (
            "ClassificationAccuracy evaluate_sample method requires the `valid_labels` "
            "attribute of the ClassificationAccuracy instance to be set."
        )
        eval_algorithm = ClassificationAccuracy()
        with pytest.raises(EvalAlgorithmClientError, match=err_msg):
            eval_algorithm.evaluate_sample("some target output", "some model output")

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

    class TestCaseClassificationAccuracyScores(NamedTuple):
        model_output: str
        target_output: str
        expected_score: float
        expected_classified_output: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseClassificationAccuracyScores(
                model_output="4  ",
                target_output="4",
                expected_score=1.0,
                expected_classified_output="4",
            ),
            TestCaseClassificationAccuracyScores(
                model_output="  1",
                target_output="2",
                expected_score=0.0,
                expected_classified_output="1",
            ),
        ],
    )
    def test_classification_accuracy_transform(self, test_case):
        """
        GIVEN a ClassificationAccuracyScores instance.
        WHEN its __call__ method is invoked.
        THEN the correct output is returned.
        """
        get_scores = ClassificationAccuracyScores(valid_labels=["1", "2", "3", "4", "5"])
        sample = {
            DatasetColumns.TARGET_OUTPUT.value.name: test_case.target_output,
            DatasetColumns.MODEL_OUTPUT.value.name: test_case.model_output,
        }
        result = get_scores(sample)
        assert result[CLASSIFICATION_ACCURACY_SCORE] == test_case.expected_score
        assert result[CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME] == test_case.expected_classified_output
