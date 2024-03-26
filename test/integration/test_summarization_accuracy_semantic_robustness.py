import json
import os

import pytest

from typing import NamedTuple, Dict
from pytest import approx

from fmeval.eval_algorithms import DATASET_CONFIGS, GIGAWORD
from fmeval.eval_algorithms.summarization_accuracy_semantic_robustness import (
    SummarizationAccuracySemanticRobustness,
    SummarizationAccuracySemanticRobustnessConfig,
    ROUGE_SCORE,
    METEOR_SCORE,
    BERT_SCORE,
    DELTA_ROUGE_SCORE,
    DELTA_METEOR_SCORE,
    DELTA_BERT_SCORE,
)
from fmeval.eval_algorithms.semantic_robustness_utils import BUTTER_FINGER, RANDOM_UPPER_CASE, WHITESPACE_ADD_REMOVE
from test.integration.models.model_runners import sm_model_runner

ABS_TOL = 1e-6
os.environ["PARALLELIZATION_FACTOR"] = "2"

BUTTER_FINGER_CONFIG = SummarizationAccuracySemanticRobustnessConfig(
    perturbation_type=BUTTER_FINGER, num_perturbations=5, butter_finger_perturbation_prob=0.1
)

RANDOM_UPPER_CASE_CONFIG = SummarizationAccuracySemanticRobustnessConfig(
    perturbation_type=RANDOM_UPPER_CASE,
    num_perturbations=5,
    random_uppercase_corrupt_proportion=0.1,
)

WHITESPACE_CONFIG = SummarizationAccuracySemanticRobustnessConfig(
    perturbation_type=WHITESPACE_ADD_REMOVE,
    num_perturbations=5,
    whitespace_remove_prob=0.1,
    whitespace_add_prob=0.05,
)


class TestCase(NamedTuple):
    config: SummarizationAccuracySemanticRobustnessConfig
    expected_scores: Dict[str, float]


class TestSummarizationAccuracySemanticRobustness:
    @pytest.mark.parametrize(
        "config, expected_scores",
        [
            TestCase(
                config=BUTTER_FINGER_CONFIG,
                expected_scores={
                    ROUGE_SCORE: 0.0,
                    METEOR_SCORE: 0.0,
                    BERT_SCORE: 0.536162,
                    DELTA_ROUGE_SCORE: 0.0,
                    DELTA_METEOR_SCORE: 0.037836,
                    DELTA_BERT_SCORE: 0.024666,
                },
            ),
            TestCase(
                config=RANDOM_UPPER_CASE_CONFIG,
                expected_scores={
                    ROUGE_SCORE: 0.0,
                    METEOR_SCORE: 0.0,
                    BERT_SCORE: 0.536162,
                    DELTA_ROUGE_SCORE: 0.0,
                    DELTA_METEOR_SCORE: 0.064103,
                    DELTA_BERT_SCORE: 0.056435,
                },
            ),
            TestCase(
                config=WHITESPACE_CONFIG,
                expected_scores={
                    ROUGE_SCORE: 0.0,
                    METEOR_SCORE: 0.0,
                    BERT_SCORE: 0.536162,
                    DELTA_ROUGE_SCORE: 0.0,
                    DELTA_METEOR_SCORE: 0.038462,
                    DELTA_BERT_SCORE: 0.039566,
                },
            ),
        ],
    )
    def test_evaluate_sample(self, config, expected_scores, integration_tests_dir):
        eval_algo = SummarizationAccuracySemanticRobustness(config)
        with open(os.path.join(integration_tests_dir, "datasets", "gigaword_sample.jsonl")) as fh:
            json_obj = json.loads(fh.readline())
            model_input = json_obj["document"]
            target_output = json_obj["summary"]
            eval_scores = eval_algo.evaluate_sample(
                model_input=model_input,
                target_output=target_output,
                model=sm_model_runner,
            )
            for eval_score in eval_scores:
                assert eval_score.value == approx(expected_scores[eval_score.name], abs=ABS_TOL)

    @pytest.mark.parametrize(
        "config, expected_scores",
        [
            TestCase(
                config=BUTTER_FINGER_CONFIG,
                expected_scores={
                    ROUGE_SCORE: 0.021908,
                    METEOR_SCORE: 0.105540,
                    BERT_SCORE: 0.559893,
                    DELTA_ROUGE_SCORE: 0.023259,
                    DELTA_METEOR_SCORE: 0.059768,
                    DELTA_BERT_SCORE: 0.031421,
                },
            ),
            TestCase(
                config=RANDOM_UPPER_CASE_CONFIG,
                expected_scores={
                    ROUGE_SCORE: 0.021908,
                    METEOR_SCORE: 0.105540,
                    BERT_SCORE: 0.559893,
                    DELTA_ROUGE_SCORE: 0.032086,
                    DELTA_METEOR_SCORE: 0.057150,
                    DELTA_BERT_SCORE: 0.026943,
                },
            ),
            TestCase(
                config=WHITESPACE_CONFIG,
                expected_scores={
                    ROUGE_SCORE: 0.021908,
                    METEOR_SCORE: 0.105540,
                    BERT_SCORE: 0.559893,
                    DELTA_ROUGE_SCORE: 0.020407,
                    DELTA_METEOR_SCORE: 0.048702,
                    DELTA_BERT_SCORE: 0.026193,
                },
            ),
        ],
    )
    def test_evaluate(self, config, expected_scores):
        eval_algo = SummarizationAccuracySemanticRobustness(config)
        dataset_config = DATASET_CONFIGS[GIGAWORD]
        eval_output = eval_algo.evaluate(
            model=sm_model_runner,
            dataset_config=dataset_config,
            save=True,
            num_records=20,
        )[0]
        for eval_score in eval_output.dataset_scores:
            assert eval_score.value == approx(expected_scores[eval_score.name], abs=ABS_TOL)
