import re
from typing import NamedTuple, List, Optional, Tuple
from unittest.mock import Mock, call, patch

import pytest
from _pytest.fixtures import fixture

from fmeval.constants import DatasetColumns, BUTTER_FINGER, WHITESPACE_ADD_REMOVE, MEAN
from fmeval.eval_algorithms import EvalScore
from fmeval.eval_algorithms.semantic_robustness_utils import RANDOM_UPPER_CASE, SEMANTIC_PERTURBATIONS
from fmeval.eval_algorithms.summarization_accuracy import METEOR_SCORE, ROUGE_SCORE, BERT_SCORE
from fmeval.eval_algorithms.summarization_accuracy_semantic_robustness import (
    SummarizationAccuracySemanticRobustnessConfig,
    SummarizationAccuracySemanticRobustness,
    DELTA_METEOR_SCORE,
    DELTA_ROUGE_SCORE,
    DELTA_BERT_SCORE,
    ORIGINAL_SCORES,
    DELTA_SCORES,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner


class MockModelRunner(ModelRunner):
    def __init__(self):
        super().__init__('{"data": $prompt}', output="output")

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        return "Some model output.", None


class TestSummarizationAccuracySemanticRobustness:
    @fixture(scope="module")
    def config(self) -> SummarizationAccuracySemanticRobustnessConfig:
        return SummarizationAccuracySemanticRobustnessConfig(num_perturbations=2)

    class TestCaseInvalidConfig(NamedTuple):
        rouge_type: str
        model_type_for_bertscore: str
        perturbation_type: str
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseInvalidConfig(
                rouge_type="rouge3",
                model_type_for_bertscore=None,
                perturbation_type="butter_finger",
                expected_error_message="Invalid rouge_type: rouge3 requested in SummarizationAccuracySemanticRobustnessConfig. "
                "Please choose from acceptable values: ['rouge1', 'rouge2', 'rougeL'].",
            ),
            TestCaseInvalidConfig(
                rouge_type="rouge1",
                model_type_for_bertscore="distilbert-base-uncased",
                perturbation_type="butter_finger",
                expected_error_message="Invalid model_type_for_bertscore: distilbert-base-uncased requested in "
                "SummarizationAccuracySemanticRobustnessConfig. Please choose from acceptable values: ["
                "'microsoft/deberta-xlarge-mnli', 'roberta-large-mnli'].",
            ),
            TestCaseInvalidConfig(
                rouge_type="rouge1",
                model_type_for_bertscore="distilbert-base-uncased",
                perturbation_type="my_perturb",
                expected_error_message="Invalid perturbation type 'my_perturb requested, please choose from "
                "acceptable values: dict_keys(['butter_finger', 'random_upper_case', 'whitespace_add_remove'])",
            ),
        ],
    )
    def test_semantic_robustness_invalid_config(self, test_case):
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            SummarizationAccuracySemanticRobustnessConfig(
                perturbation_type=test_case.perturbation_type,
                rouge_type=test_case.rouge_type,
                model_type_for_bertscore=test_case.model_type_for_bertscore,
            )

    @pytest.mark.parametrize("perturbation_type", [BUTTER_FINGER, RANDOM_UPPER_CASE, WHITESPACE_ADD_REMOVE])
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.create_shared_resource")
    def test_init(self, mock_created_shared_resource, perturbation_type):
        """
        GIVEN valid arguments.
        WHEN a SummarizationAccuracySemanticRobustness is initialized.
        THEN its instance attributes are of the correct type.
        """
        config = SummarizationAccuracySemanticRobustnessConfig(perturbation_type=perturbation_type)
        sasr = SummarizationAccuracySemanticRobustness(config)
        assert isinstance(sasr.perturbation_transform, SEMANTIC_PERTURBATIONS[perturbation_type])
        mock_created_shared_resource.assert_called_once()

    class TestCaseEvaluateSample(NamedTuple):
        original_meteor_score: float
        original_rouge_score: float
        original_bert_score: float
        perturbed_meteor_scores: List[float]
        perturbed_rouge_scores: List[float]
        perturbed_bert_scores: List[float]
        expected_response: List[EvalScore]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseEvaluateSample(
                original_meteor_score=1.0,
                original_rouge_score=1.0,
                original_bert_score=0.5,
                perturbed_meteor_scores=[0.75, 0.5],
                perturbed_rouge_scores=[0.5, 0.2],
                perturbed_bert_scores=[1.0, 0.5],
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=1.0),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                    EvalScore(name=BERT_SCORE, value=0.5),
                    EvalScore(name=DELTA_METEOR_SCORE, value=0.375),
                    EvalScore(name=DELTA_ROUGE_SCORE, value=0.65),
                    EvalScore(name=DELTA_BERT_SCORE, value=0.25),
                ],
            ),
        ],
    )
    @patch("fmeval.transforms.semantic_perturbations.RandomUppercase.perturb")
    @patch("fmeval.transforms.summarization_accuracy_metrics.BertScore.compute_metric")
    @patch("fmeval.transforms.summarization_accuracy_metrics.RougeScore.compute_metric")
    @patch("fmeval.transforms.summarization_accuracy_metrics.MeteorScore.compute_metric")
    def test_semantic_robustness_evaluate_sample(
        self,
        mock_meteor_metric,
        mock_rouge_metric,
        mock_bert_metric,
        mock_random_uppercase_perturb,
        test_case,
        config,
    ):
        """
        GIVEN a config with perturbation_type = RANDOM_UPPERCASE (wlog).
        WHEN SummarizationAccuracySemanticRobustness.evaluate_sample is called.
        THEN the correct list of EvalScores is returned, RandomUppercase.perturb is called,
            prompts are created from the perturbed model inputs, and the model runner's
            `predict` method is called on the original prompt (constructed from the
             original model input) and perturbed prompts (constructed from the perturbed
             model inputs).
        """
        mock_meteor_metric.side_effect = [test_case.original_meteor_score] + test_case.perturbed_meteor_scores
        mock_rouge_metric.side_effect = [test_case.original_rouge_score] + test_case.perturbed_rouge_scores
        mock_bert_metric.side_effect = [test_case.original_bert_score] + test_case.perturbed_bert_scores

        mock_random_uppercase_perturb.return_value = ["perturbed input 1", "perturbed input 2"]

        model = Mock()
        model.predict.side_effect = [("a", None), ("b", None), ("c", None)]

        eval_algorithm = SummarizationAccuracySemanticRobustness(
            SummarizationAccuracySemanticRobustnessConfig(
                perturbation_type=RANDOM_UPPER_CASE,
                num_perturbations=2,
            ),
        )
        assert (
            eval_algorithm.evaluate_sample(
                model_input="the model input", target_output="unused", model=model, prompt_template="Hi $model_input"
            )
            == test_case.expected_response
        )
        model.predict.assert_has_calls(
            [call("Hi the model input"), call("Hi perturbed input 1"), call("Hi perturbed input 2")]
        )

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
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.get_eval_results_path")
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.evaluate_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.create_shared_resource")
    @patch(
        "fmeval.eval_algorithms.summarization_accuracy_semantic_robustness."
        "SummarizationAccuracySemanticRobustness._build_pipeline"
    )
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.get_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.get_dataset_configs")
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.BertscoreHelperModel")
    def test_evaluate(
        self,
        mock_bertscore_model_cls,
        mock_get_dataset_configs,
        mock_get_dataset,
        mock_build_pipeline,
        mock_create_shared_resource,
        mock_evaluate_dataset,
        mock_get_results_path,
        test_case,
    ):
        """
        GIVEN a SummarizationAccuracySemanticRobustness instance.
        WHEN its evaluate method is called with valid arguments.
        THEN `evaluate_dataset` is called with the correct arguments.
        """
        mock_bertscore_model_cls.return_value = Mock()

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

        sasr = SummarizationAccuracySemanticRobustness()
        output = sasr.evaluate(
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
            eval_name=sasr.eval_name,
            metric_names=ORIGINAL_SCORES + DELTA_SCORES,
            eval_results_path="/path/to/results",
            model=model_runner,
            prompt_template=test_case.dataset_prompt_template,
            agg_method=MEAN,
            save=True,
        )
        mock_build_pipeline.assert_called_once_with(
            model_runner,
            test_case.dataset_prompt_template,
        )
        assert output == [mock_evaluate_dataset.return_value]
