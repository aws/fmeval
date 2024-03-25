import re
import pytest

from typing import NamedTuple, List, Optional, Tuple
from unittest.mock import patch, MagicMock, Mock
from _pytest.fixtures import fixture

from fmeval.constants import (
    BUTTER_FINGER,
    RANDOM_UPPER_CASE,
    WHITESPACE_ADD_REMOVE,
    DatasetColumns,
    MEAN,
)
from fmeval.eval_algorithms import EvalScore
from fmeval.eval_algorithms.qa_accuracy_semantic_robustness import (
    QAAccuracySemanticRobustnessConfig,
    QAAccuracySemanticRobustness,
    DELTA_F1_SCORE,
    DELTA_QUASI_EXACT_MATCH_SCORE,
    DELTA_EXACT_MATCH_SCORE,
    DELTA_PRECISION_OVER_WORDS,
    DELTA_RECALL_OVER_WORDS,
    ORIGINAL_SCORES,
    DELTA_SCORES,
)
from fmeval.eval_algorithms.qa_accuracy import (
    F1_SCORE,
    QUASI_EXACT_MATCH_SCORE,
    EXACT_MATCH_SCORE,
    PRECISION_OVER_WORDS,
    RECALL_OVER_WORDS,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner


class ConstantModel(ModelRunner):
    def __init__(self):
        super().__init__('{"data": $prompt}', output="output")

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        return "Some model output.", None


class TestQAAccuracySemanticRobustness:
    @fixture(scope="module")
    def config(self) -> QAAccuracySemanticRobustnessConfig:
        return QAAccuracySemanticRobustnessConfig(target_output_delimiter="<OR>", num_perturbations=2)

    class TestCaseQAAccuracySemanticRobustnessInvalidConfig(NamedTuple):
        target_output_delimiter: Optional[str]
        perturbation_type: str
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracySemanticRobustnessInvalidConfig(
                target_output_delimiter="<OR>",
                perturbation_type="my_perturb",
                expected_error_message="Invalid perturbation type 'my_perturb requested, please choose from "
                "acceptable values: dict_keys(['butter_finger', 'random_upper_case', 'whitespace_add_remove'])",
            ),
            TestCaseQAAccuracySemanticRobustnessInvalidConfig(
                target_output_delimiter="",
                perturbation_type="butter_finger",
                expected_error_message="Empty target_output_delimiter is provided. Please either provide a non-empty string, or set it to None",
            ),
        ],
    )
    def test_qa_accuracy_semantic_robustness_invalid_config(self, test_case):
        """
        GIVEN invalid configs
        WHEN QAAccuracySemanticRobustnessConfig is initialized
        THEN correct exception with proper message is raised
        """
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            QAAccuracySemanticRobustnessConfig(
                target_output_delimiter=test_case.target_output_delimiter,
                perturbation_type=test_case.perturbation_type,
            )

    class TestCaseQAAccuracySemanticRobustnessEvaluateSample(NamedTuple):
        model_input: str
        original_model_output: str
        perturbed_model_output_1: str
        perturbed_model_output_2: str
        target_output: str
        expected_response: List[EvalScore]
        config: QAAccuracySemanticRobustnessConfig

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracySemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="london!",
                perturbed_model_output_1="Some model output.",
                perturbed_model_output_2="Some model output.",
                target_output="London",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=1.0),
                    EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=PRECISION_OVER_WORDS, value=1.0),
                    EvalScore(name=RECALL_OVER_WORDS, value=1.0),
                    EvalScore(name=DELTA_F1_SCORE, value=1.0),
                    EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=1.0),
                    EvalScore(name=DELTA_RECALL_OVER_WORDS, value=1.0),
                ],
                config=QAAccuracySemanticRobustnessConfig(target_output_delimiter="<OR>", num_perturbations=2),
            ),
            TestCaseQAAccuracySemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="london!",
                perturbed_model_output_1="london",
                perturbed_model_output_2="paris",
                target_output="London",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=1.0),
                    EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=PRECISION_OVER_WORDS, value=1),
                    EvalScore(name=RECALL_OVER_WORDS, value=1),
                    EvalScore(name=DELTA_F1_SCORE, value=0.5),
                    EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=0.5),
                    EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=0.5),
                    EvalScore(name=DELTA_RECALL_OVER_WORDS, value=0.5),
                ],
                config=QAAccuracySemanticRobustnessConfig(
                    target_output_delimiter="<OR>", num_perturbations=2, perturbation_type=BUTTER_FINGER
                ),
            ),
            TestCaseQAAccuracySemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="London is the capital",
                perturbed_model_output_1="Another model output.",
                perturbed_model_output_2="Some model output.",
                target_output="London",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=0.5),
                    EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=PRECISION_OVER_WORDS, value=1 / 3),
                    EvalScore(name=RECALL_OVER_WORDS, value=1),
                    EvalScore(name=DELTA_F1_SCORE, value=0.5),
                    EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=1 / 3),
                    EvalScore(name=DELTA_RECALL_OVER_WORDS, value=1.0),
                ],
                config=QAAccuracySemanticRobustnessConfig(
                    target_output_delimiter="<OR>", num_perturbations=2, perturbation_type=RANDOM_UPPER_CASE
                ),
            ),
            TestCaseQAAccuracySemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="London",
                perturbed_model_output_1="Another model output.",
                perturbed_model_output_2="Another model output.",
                target_output="London",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=1.0),
                    EvalScore(name=EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=PRECISION_OVER_WORDS, value=1.0),
                    EvalScore(name=RECALL_OVER_WORDS, value=1.0),
                    EvalScore(name=DELTA_F1_SCORE, value=1.0),
                    EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=1.0),
                    EvalScore(name=DELTA_RECALL_OVER_WORDS, value=1.0),
                ],
                config=QAAccuracySemanticRobustnessConfig(
                    target_output_delimiter="<OR>", num_perturbations=2, perturbation_type=WHITESPACE_ADD_REMOVE
                ),
            ),
        ],
    )
    def test_qa_accuracy_semantic_robustness_evaluate_sample(self, test_case):
        """
        GIVEN valid inputs
        WHEN QAAccuracySemanticRobustness.evaluate_sample is called
        THEN correct List of EvalScores is returned
        """
        model = MagicMock()
        model.predict.side_effect = [
            (test_case.original_model_output, None),
            (test_case.perturbed_model_output_1, None),
            (test_case.perturbed_model_output_2, None),
        ]

        eval_algorithm = QAAccuracySemanticRobustness(test_case.config)
        assert (
            eval_algorithm.evaluate_sample(
                model_input=test_case.model_input,
                model=model,
                target_output=test_case.target_output,
            )
            == test_case.expected_response
        )
        assert model.predict.call_count == 3

    class TestCaseEvaluate(NamedTuple):
        user_provided_prompt_template: Optional[str]
        dataset_prompt_template: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseEvaluate(
                user_provided_prompt_template="Summarize: $model_input",
                dataset_prompt_template="Summarize: $model_input",
            ),
            TestCaseEvaluate(
                user_provided_prompt_template=None,
                dataset_prompt_template="$model_input",
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.qa_accuracy_semantic_robustness.get_eval_results_path")
    @patch("fmeval.eval_algorithms.qa_accuracy_semantic_robustness.evaluate_dataset")
    @patch("fmeval.eval_algorithms.qa_accuracy_semantic_robustness." "QAAccuracySemanticRobustness.build_pipeline")
    @patch("fmeval.eval_algorithms.qa_accuracy_semantic_robustness.get_dataset")
    @patch("fmeval.eval_algorithms.qa_accuracy_semantic_robustness.get_dataset_configs")
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
        GIVEN a QAAccuracySemanticRobustness instance.
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
        mock_get_dataset.return_value = mock_dataset

        mock_build_pipeline.return_value = Mock()
        mock_get_results_path.return_value = "/path/to/results"
        model_runner = Mock()

        eval_algo = QAAccuracySemanticRobustness()
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
            metric_names=ORIGINAL_SCORES + DELTA_SCORES,
            eval_results_path="/path/to/results",
            model=model_runner,
            prompt_template=test_case.dataset_prompt_template,
            agg_method=MEAN,
            save=True,
        )
        mock_build_pipeline.assert_called_with(model_runner, test_case.dataset_prompt_template)
        assert output == [mock_evaluate_dataset.return_value]
