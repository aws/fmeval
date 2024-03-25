import os
import pytest

from pytest import approx
from typing import NamedTuple, Dict
from fmeval.eval_algorithms.qa_accuracy_semantic_robustness import (
    QAAccuracySemanticRobustness,
    QAAccuracySemanticRobustnessConfig,
    DELTA_F1_SCORE,
    DELTA_EXACT_MATCH_SCORE,
    DELTA_QUASI_EXACT_MATCH_SCORE,
    DELTA_PRECISION_OVER_WORDS,
    DELTA_RECALL_OVER_WORDS,
)
from fmeval.eval_algorithms.qa_accuracy import (
    F1_SCORE,
    QUASI_EXACT_MATCH_SCORE,
    EXACT_MATCH_SCORE,
    PRECISION_OVER_WORDS,
    RECALL_OVER_WORDS,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.constants import MIME_TYPE_JSONLINES, BUTTER_FINGER, RANDOM_UPPER_CASE, WHITESPACE_ADD_REMOVE
from test.integration.models.model_runners import sm_model_runner

ABS_TOL = 1e-4
os.environ["PARALLELIZATION_FACTOR"] = "2"

sm_model_runner_prompt_template = """
    <s>[INST] <<SYS>>Answer the question at the end in as few words as possible.
    Do not repeat the question. Do not answer in complete sentences. <</SYS>>
    Question: $model_input [/INST]
    """


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
                expected_scores={
                    F1_SCORE: 1.0,
                    EXACT_MATCH_SCORE: 1.0,
                    QUASI_EXACT_MATCH_SCORE: 1.0,
                    PRECISION_OVER_WORDS: 1.0,
                    RECALL_OVER_WORDS: 1.0,
                    DELTA_F1_SCORE: 0.6,
                    DELTA_EXACT_MATCH_SCORE: 0.6,
                    DELTA_QUASI_EXACT_MATCH_SCORE: 0.6,
                    DELTA_PRECISION_OVER_WORDS: 0.6,
                    DELTA_RECALL_OVER_WORDS: 0.6,
                },
            ),
            TestCaseEvaluateSample(
                config=QAAccuracySemanticRobustnessConfig(
                    perturbation_type=RANDOM_UPPER_CASE,
                    num_perturbations=5,
                    random_uppercase_corrupt_proportion=0.1,
                ),
                expected_scores={
                    F1_SCORE: 1.0,
                    EXACT_MATCH_SCORE: 1.0,
                    QUASI_EXACT_MATCH_SCORE: 1.0,
                    PRECISION_OVER_WORDS: 1.0,
                    RECALL_OVER_WORDS: 1.0,
                    DELTA_F1_SCORE: 0.2,
                    DELTA_EXACT_MATCH_SCORE: 0.2,
                    DELTA_QUASI_EXACT_MATCH_SCORE: 0.2,
                    DELTA_PRECISION_OVER_WORDS: 0.2,
                    DELTA_RECALL_OVER_WORDS: 0.2,
                },
            ),
            TestCaseEvaluateSample(
                config=QAAccuracySemanticRobustnessConfig(
                    perturbation_type=WHITESPACE_ADD_REMOVE,
                    num_perturbations=5,
                    whitespace_remove_prob=0.1,
                    whitespace_add_prob=0.05,
                ),
                expected_scores={
                    F1_SCORE: 1.0,
                    EXACT_MATCH_SCORE: 1.0,
                    QUASI_EXACT_MATCH_SCORE: 1.0,
                    PRECISION_OVER_WORDS: 1.0,
                    RECALL_OVER_WORDS: 1.0,
                    DELTA_F1_SCORE: 0.6,
                    DELTA_EXACT_MATCH_SCORE: 0.6,
                    DELTA_QUASI_EXACT_MATCH_SCORE: 0.6,
                    DELTA_PRECISION_OVER_WORDS: 0.6,
                    DELTA_RECALL_OVER_WORDS: 0.6,
                },
            ),
        ],
    )
    def test_evaluate_sample(self, config, expected_scores):
        eval_algo = QAAccuracySemanticRobustness(config)
        model_input = "London is the capital of"
        eval_scores = eval_algo.evaluate_sample(
            model_input=model_input,
            model=sm_model_runner,
            target_output="UK<OR>England<OR>United Kingdom",
            prompt_template=sm_model_runner_prompt_template,
        )
        for eval_score in eval_scores:
            print(f"eval score {eval_score.name} is {eval_score.value}")
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
                    F1_SCORE: 0.3606,
                    EXACT_MATCH_SCORE: 0.0606,
                    QUASI_EXACT_MATCH_SCORE: 0.3030,
                    PRECISION_OVER_WORDS: 0.3577,
                    RECALL_OVER_WORDS: 0.3813,
                    DELTA_F1_SCORE: 0.2100,
                    DELTA_EXACT_MATCH_SCORE: 0.0687,
                    DELTA_QUASI_EXACT_MATCH_SCORE: 0.1919,
                    DELTA_PRECISION_OVER_WORDS: 0.2123,
                    DELTA_RECALL_OVER_WORDS: 0.2096,
                },
            ),
            TestCaseEvaluate(
                config=QAAccuracySemanticRobustnessConfig(
                    perturbation_type=RANDOM_UPPER_CASE,
                    num_perturbations=5,
                    random_uppercase_corrupt_proportion=0.1,
                ),
                expected_scores={
                    F1_SCORE: 0.3606,
                    EXACT_MATCH_SCORE: 0.0606,
                    QUASI_EXACT_MATCH_SCORE: 0.3030,
                    PRECISION_OVER_WORDS: 0.3577,
                    RECALL_OVER_WORDS: 0.3813,
                    DELTA_F1_SCORE: 0.1507,
                    DELTA_EXACT_MATCH_SCORE: 0.0586,
                    DELTA_QUASI_EXACT_MATCH_SCORE: 0.1293,
                    DELTA_PRECISION_OVER_WORDS: 0.1573,
                    DELTA_RECALL_OVER_WORDS: 0.1525,
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
                    F1_SCORE: 0.3606,
                    EXACT_MATCH_SCORE: 0.0606,
                    QUASI_EXACT_MATCH_SCORE: 0.3030,
                    PRECISION_OVER_WORDS: 0.3577,
                    RECALL_OVER_WORDS: 0.3813,
                    DELTA_F1_SCORE: 0.1351,
                    DELTA_EXACT_MATCH_SCORE: 0.0404,
                    DELTA_QUASI_EXACT_MATCH_SCORE: 0.1212,
                    DELTA_PRECISION_OVER_WORDS: 0.1355,
                    DELTA_RECALL_OVER_WORDS: 0.1384,
                },
            ),
        ],
    )
    def test_evaluate(self, integration_tests_dir, config, expected_scores):
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
        assert False
