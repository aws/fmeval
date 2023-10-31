import os
from typing import NamedTuple, Dict

import pytest
from pytest import approx
from amazon_fmeval.eval_algorithms.qa_accuracy_semantic_robustness import (
    QAAccuracySemanticRobustness,
    QAAccuracySemanticRobustnessConfig,
    BUTTER_FINGER,
    RANDOM_UPPER_CASE,
    WHITESPACE_ADD_REMOVE,
    DELTA_F1_SCORE,
    DELTA_EXACT_MATCH_SCORE,
    DELTA_QUASI_EXACT_MATCH_SCORE,
)
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.constants import MIME_TYPE_JSONLINES

ABS_TOL = 1e-4
os.environ["PARALLELIZATION_FACTOR"] = "2"


class TestQAAccuracySemanticRobustness:
    class TestCaseEvaluateSample(NamedTuple):
        config: QAAccuracySemanticRobustnessConfig
        expected_scores: Dict[str, float]

    @pytest.mark.parametrize(
        "config, expected_scores",
        [
            TestCaseEvaluateSample(
                config=QAAccuracySemanticRobustnessConfig(
                    perturbation_type=BUTTER_FINGER, num_perturbations=5, butter_finger_perturbation_prob=0.1
                ),
                expected_scores={DELTA_F1_SCORE: 0.8, DELTA_EXACT_MATCH_SCORE: 0.8, DELTA_QUASI_EXACT_MATCH_SCORE: 0.8},
            ),
            TestCaseEvaluateSample(
                config=QAAccuracySemanticRobustnessConfig(
                    perturbation_type=RANDOM_UPPER_CASE,
                    num_perturbations=5,
                    random_uppercase_corrupt_proportion=0.1,
                ),
                expected_scores={DELTA_F1_SCORE: 1.0, DELTA_EXACT_MATCH_SCORE: 1.0, DELTA_QUASI_EXACT_MATCH_SCORE: 1.0},
            ),
            TestCaseEvaluateSample(
                config=QAAccuracySemanticRobustnessConfig(
                    perturbation_type=WHITESPACE_ADD_REMOVE,
                    num_perturbations=5,
                    whitespace_remove_prob=0.1,
                    whitespace_add_prob=0.05,
                ),
                expected_scores={DELTA_F1_SCORE: 0.8, DELTA_EXACT_MATCH_SCORE: 0.8, DELTA_QUASI_EXACT_MATCH_SCORE: 0.8},
            ),
        ],
    )
    def test_evaluate_sample(self, config, expected_scores, sm_model_runner, sm_model_runner_prompt_template):
        eval_algo = QAAccuracySemanticRobustness(config)
        model_input = "London is the capital of"
        eval_scores = eval_algo.evaluate_sample(
            model_input=model_input,
            model=sm_model_runner,
            target_output="UK<OR>England<OR>United Kingdom",
            prompt_template=sm_model_runner_prompt_template,
        )
        for eval_score in eval_scores:
            assert eval_score.value == expected_scores[eval_score.name]

    class TestCaseEvaluate(NamedTuple):
        config: QAAccuracySemanticRobustnessConfig
        expected_scores: Dict[str, float]

    @pytest.mark.parametrize(
        "config, expected_scores",
        [
            TestCaseEvaluate(
                config=QAAccuracySemanticRobustnessConfig(
                    perturbation_type=BUTTER_FINGER, num_perturbations=5, butter_finger_perturbation_prob=0.1
                ),
                expected_scores={
                    DELTA_F1_SCORE: 0.1879,
                    DELTA_EXACT_MATCH_SCORE: 0.0383,
                    DELTA_QUASI_EXACT_MATCH_SCORE: 0.1858,
                },
            ),
            TestCaseEvaluate(
                config=QAAccuracySemanticRobustnessConfig(
                    perturbation_type=RANDOM_UPPER_CASE,
                    num_perturbations=5,
                    random_uppercase_corrupt_proportion=0.1,
                ),
                expected_scores={
                    DELTA_F1_SCORE: 0.0984,
                    DELTA_EXACT_MATCH_SCORE: 0.0323,
                    DELTA_QUASI_EXACT_MATCH_SCORE: 0.1111,
                },
            ),
            TestCaseEvaluate(
                config=QAAccuracySemanticRobustnessConfig(
                    perturbation_type=WHITESPACE_ADD_REMOVE,
                    num_perturbations=5,
                    whitespace_remove_prob=0.1,
                    whitespace_add_prob=0.05,
                ),
                expected_scores={
                    DELTA_F1_SCORE: 0.0976,
                    DELTA_EXACT_MATCH_SCORE: 0.0161,
                    DELTA_QUASI_EXACT_MATCH_SCORE: 0.1010,
                },
            ),
        ],
    )
    def test_evaluate(
        self, integration_tests_dir, config, expected_scores, sm_model_runner, sm_model_runner_prompt_template
    ):
        eval_algo = QAAccuracySemanticRobustness(config)
        dataset_config = DataConfig(
            dataset_name="triviaQA_sample",
            dataset_uri=os.path.join(integration_tests_dir, "datasets", "triviaQA_sample.jsonl"),
            dataset_mime_type=MIME_TYPE_JSONLINES,
            model_input_location="question",
            target_output_location="answer",
        )
        eval_output = eval_algo.evaluate(
            model=sm_model_runner,
            dataset_config=dataset_config,
            prompt_template=sm_model_runner_prompt_template,
            save=True,
        )[0]
        for eval_score in eval_output.dataset_scores:
            assert eval_score.value == approx(expected_scores[eval_score.name], abs=ABS_TOL)
