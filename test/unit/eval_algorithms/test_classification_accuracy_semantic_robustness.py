import re
from typing import NamedTuple, List, Optional, Tuple
from unittest.mock import patch, MagicMock, Mock

import pytest
import ray
from _pytest.fixtures import fixture
from ray.data import Dataset

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON, BUTTER_FINGER, RANDOM_UPPER_CASE, WHITESPACE_ADD_REMOVE, MEAN,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms import (
    EvalScore,
    EvalOutput,
    CategoryScore,
    BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES,
    DEFAULT_PROMPT_TEMPLATE,
    WOMENS_CLOTHING_ECOMMERCE_REVIEWS,
)
from fmeval.eval_algorithms.classification_accuracy_semantic_robustness import (
    ClassificationAccuracySemanticRobustnessConfig,
    ClassificationAccuracySemanticRobustness,
    DELTA_CLASSIFICATION_ACCURACY_SCORE,
)
from fmeval.eval_algorithms.classification_accuracy import CLASSIFICATION_ACCURACY_SCORE
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner

DATASET = ray.data.from_items(
    [
        {
            DatasetColumns.MODEL_INPUT.value.name: "Delicious cake! Would buy again.",
            DatasetColumns.TARGET_OUTPUT.value.name: "4",
            DatasetColumns.MODEL_OUTPUT.value.name: "Some model output.",
            DatasetColumns.CATEGORY.value.name: "brownie",
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "Tasty cake! Must eat.",
            DatasetColumns.TARGET_OUTPUT.value.name: "4",
            DatasetColumns.MODEL_OUTPUT.value.name: "Some model output.",
            DatasetColumns.CATEGORY.value.name: "vanilla cake",
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "Terrible! Nightmarish cake.",
            DatasetColumns.TARGET_OUTPUT.value.name: "1",
            DatasetColumns.MODEL_OUTPUT.value.name: "Some model output.",
            DatasetColumns.CATEGORY.value.name: "vanilla cake",
        },
    ]
)

DATASET_WITHOUT_CATEGORY = DATASET.drop_columns(cols=[DatasetColumns.CATEGORY.value.name])

DATASET_WITHOUT_MODEL_OUTPUT = DATASET.drop_columns(cols=[DatasetColumns.MODEL_OUTPUT.value.name])

DATASET_WITHOUT_MODEL_INPUT = DATASET.drop_columns(cols=[DatasetColumns.MODEL_INPUT.value.name])


CATEGORY_SCORES = [
    CategoryScore(
        name="brownie",
        scores=[
            EvalScore(name=CLASSIFICATION_ACCURACY_SCORE, value=0.0),
            EvalScore(name=DELTA_CLASSIFICATION_ACCURACY_SCORE, value=0.0),
        ],
    ),
    CategoryScore(
        name="vanilla cake",
        scores=[
            EvalScore(name=CLASSIFICATION_ACCURACY_SCORE, value=0.0),
            EvalScore(name=DELTA_CLASSIFICATION_ACCURACY_SCORE, value=0.0),
        ],
    ),
]


class ConstantModel(ModelRunner):
    def __init__(self):
        super().__init__('{"data": $prompt}', output="output")

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        return "Some model output.", None


class TestClassificationAccuracySemanticRobustness:
    @fixture(scope="module")
    def config(self) -> ClassificationAccuracySemanticRobustnessConfig:
        return ClassificationAccuracySemanticRobustnessConfig(
            valid_labels=["1", "2", "3", "4", "5"], num_perturbations=2
        )

    class TestCaseClassificationAccuracySemanticRobustnessInvalidConfig(NamedTuple):
        valid_labels: List[str]
        perturbation_type: str
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseClassificationAccuracySemanticRobustnessInvalidConfig(
                valid_labels=["1", "2"],
                perturbation_type="my_perturb",
                expected_error_message="Invalid perturbation type 'my_perturb requested, please choose from "
                "acceptable values: dict_keys(['butter_finger', 'random_upper_case', 'whitespace_add_remove'])",
            ),
        ],
    )
    def test_classification_accuracy_semantic_robustness_invalid_config(self, test_case):
        """
        GIVEN invalid configs
        WHEN ClassificationAccuracySemanticRobustnessConfig is initialized
        THEN correct exception with proper message is raised
        """
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            ClassificationAccuracySemanticRobustnessConfig(
                valid_labels=test_case.valid_labels,
                perturbation_type=test_case.perturbation_type,
            )

    def test_classification_accuracy_config_format_with_castable_labels(self):
        """
        GIVEN valid labels are int but can be cast to str
        WHEN ClassificationAccuracySemanticRobustnessConfig is initialized with castable integer labels
        THEN warning is raised, ClassificationAccuracySemanticRobustnessConfig is initialized successfully
        """
        with pytest.warns():
            castable_config = ClassificationAccuracySemanticRobustnessConfig(
                valid_labels=[1, 2, 3, 4, 5], perturbation_type="butter_finger"
            )
            assert castable_config.valid_labels == ["1", "2", "3", "4", "5"]

    class TestCaseClassificationAccuracySemanticRobustnessEvaluateSample(NamedTuple):
        model_input: str
        original_model_output: str
        perturbed_model_output_1: str
        perturbed_model_output_2: str
        target_output: str
        expected_response: List[EvalScore]
        config: ClassificationAccuracySemanticRobustnessConfig

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseClassificationAccuracySemanticRobustnessEvaluateSample(
                model_input="Ok brownie.",
                original_model_output="3",
                perturbed_model_output_1="Some model output.",
                perturbed_model_output_2="Some model output.",
                target_output="3",
                expected_response=[
                    EvalScore(name=CLASSIFICATION_ACCURACY_SCORE, value=1.0),
                    EvalScore(name=DELTA_CLASSIFICATION_ACCURACY_SCORE, value=1.0),
                ],
                config=ClassificationAccuracySemanticRobustnessConfig(
                    valid_labels=["1", "2", "3", "4", "5"],
                    num_perturbations=2,
                ),
            ),
            TestCaseClassificationAccuracySemanticRobustnessEvaluateSample(
                model_input="Good cake.",
                original_model_output="4",
                perturbed_model_output_1="Another model output.",
                perturbed_model_output_2="Some model output.",
                target_output="4",
                expected_response=[
                    EvalScore(name=CLASSIFICATION_ACCURACY_SCORE, value=1.0),
                    EvalScore(name=DELTA_CLASSIFICATION_ACCURACY_SCORE, value=1.0),
                ],
                config=ClassificationAccuracySemanticRobustnessConfig(
                    valid_labels=["1", "2", "3", "4", "5"], num_perturbations=2, perturbation_type=BUTTER_FINGER
                ),
            ),
            TestCaseClassificationAccuracySemanticRobustnessEvaluateSample(
                model_input="Delicious! Nightmarish cake.",
                original_model_output="5",
                perturbed_model_output_1="Another model output.",
                perturbed_model_output_2="Some model output.",
                target_output="2",
                expected_response=[
                    EvalScore(name=CLASSIFICATION_ACCURACY_SCORE, value=0.0),
                    EvalScore(name=DELTA_CLASSIFICATION_ACCURACY_SCORE, value=0.0),
                ],
                config=ClassificationAccuracySemanticRobustnessConfig(
                    valid_labels=["1", "2", "3", "4", "5"], num_perturbations=2, perturbation_type=RANDOM_UPPER_CASE
                ),
            ),
            TestCaseClassificationAccuracySemanticRobustnessEvaluateSample(
                model_input="Terrible! Nightmarish cake.",
                original_model_output="1",
                perturbed_model_output_1="Another model output.",
                perturbed_model_output_2="Another model output.",
                target_output="1",
                expected_response=[
                    EvalScore(name=CLASSIFICATION_ACCURACY_SCORE, value=1.0),
                    EvalScore(name=DELTA_CLASSIFICATION_ACCURACY_SCORE, value=1.0),
                ],
                config=ClassificationAccuracySemanticRobustnessConfig(
                    valid_labels=["1", "2", "3", "4", "5"], num_perturbations=2, perturbation_type=WHITESPACE_ADD_REMOVE
                ),
            ),
        ],
    )
    def test_classification_accuracy_semantic_robustness_evaluate_sample(self, test_case):
        """
        GIVEN valid inputs
        WHEN ClassificationAccuracySemanticRobustness.evaluate_sample is called
        THEN correct List of EvalScores is returned
        """
        model = MagicMock()
        model.predict.side_effect = [
            (test_case.original_model_output, None),
            (test_case.perturbed_model_output_1, None),
            (test_case.perturbed_model_output_2, None),
        ]

        eval_algorithm = ClassificationAccuracySemanticRobustness(test_case.config)
        assert (
            eval_algorithm.evaluate_sample(
                model_input=test_case.model_input, model=model, target_output=test_case.target_output
            )
            == test_case.expected_response
        )
        assert model.predict.call_count == 3

    class TestCaseEvaluate(NamedTuple):
        user_provided_prompt_template: Optional[str]
        dataset_prompt_template: str
        valid_labels: Optional[List[str]] = None

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseEvaluate(
                user_provided_prompt_template="Summarize: $model_input",
                dataset_prompt_template="Summarize: $model_input",
                valid_labels=["0", "1"],
            ),
            TestCaseEvaluate(
                user_provided_prompt_template=None,
                dataset_prompt_template="$model_input",
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.classification_accuracy_semantic_robustness.get_eval_results_path")
    @patch("fmeval.eval_algorithms.classification_accuracy_semantic_robustness.evaluate_dataset")
    @patch("fmeval.eval_algorithms.classification_accuracy_semantic_robustness.ClassificationAccuracySemanticRobustness._build_pipeline")
    @patch("fmeval.eval_algorithms.classification_accuracy_semantic_robustness.get_dataset")
    @patch("fmeval.eval_algorithms.classification_accuracy_semantic_robustness.get_dataset_configs")
    def test_evaluate(
            self,
            mock_get_dataset_configs,
            mock_get_dataset,
            mock_build_pipeline,
            mock_evaluate_dataset,
            mock_get_results_path,
            test_case,
    ):
        """
        GIVEN a ClassificationAccuracySemanticRobustness instance.
        WHEN its evaluate method is called with valid arguments.
        THEN `evaluate_dataset` is called with the correct arguments.
        """
        dataset_config = Mock()
        dataset_config.dataset_name = "my_custom_dataset"
        mock_get_dataset_configs.return_value = [dataset_config]

        mock_dataset = Mock()
        # So that validate_dataset does not error
        mock_dataset.columns = Mock(
            return_value=[DatasetColumns.MODEL_INPUT.value.name, DatasetColumns.TARGET_OUTPUT.value.name]
        )
        mock_dataset.unique = Mock(return_value=["0", "1", "2"])
        # So that the uniqueness factor check passes
        mock_dataset.count = Mock(return_value=100)
        mock_get_dataset.return_value = mock_dataset

        mock_build_pipeline.return_value = Mock()
        mock_get_results_path.return_value = "/path/to/results"
        model_runner = Mock()

        eval_algo = ClassificationAccuracySemanticRobustness(
            ClassificationAccuracySemanticRobustnessConfig(valid_labels=test_case.valid_labels)
        )
        output = eval_algo.evaluate(
            model=model_runner,
            dataset_config=dataset_config,
            prompt_template=test_case.user_provided_prompt_template,
            num_records=162,
            save=True,
        )

        mock_evaluate_dataset.assert_called_once_with(
            dataset=mock_dataset,
            pipeline=mock_build_pipeline.return_value,
            dataset_name=dataset_config.dataset_name,
            eval_name=eval_algo.eval_name,
            metric_names=[CLASSIFICATION_ACCURACY_SCORE, DELTA_CLASSIFICATION_ACCURACY_SCORE],
            eval_results_path="/path/to/results",
            model=model_runner,
            prompt_template=test_case.dataset_prompt_template,
            agg_method=MEAN,
            save=True,
        )
        mock_build_pipeline.assert_called_with(
            model_runner,
            test_case.dataset_prompt_template,
            test_case.valid_labels if test_case.valid_labels else mock_dataset.unique.return_value
        )
        assert output == [mock_evaluate_dataset.return_value]
