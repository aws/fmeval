import os
import json
import ray
from typing import NamedTuple, Dict

import pytest
from pytest import approx
from amazon_fmeval.eval_algorithms.summarization_accuracy_semantic_robustness import (
    SummarizationAccuracySemanticRobustness,
    SummarizationAccuracySemanticRobustnessConfig,
    ROUGE_SCORE,
    METEOR_SCORE,
    BERT_SCORE,
    DELTA_ROUGE_SCORE,
    DELTA_METEOR_SCORE,
    DELTA_BERT_SCORE,
    BUTTER_FINGER,
    RANDOM_UPPER_CASE,
    WHITESPACE_ADD_REMOVE,
)
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.constants import MIME_TYPE_JSONLINES
from test.integration.models.model_runners import sm_model_runner

ABS_TOL = 1e-6
os.environ["PARALLELIZATION_FACTOR"] = "2"


class TestSummarizationAccuracySemanticRobustness:
    class TestCaseEvaluate(NamedTuple):
        config: SummarizationAccuracySemanticRobustnessConfig
        expected_scores: Dict[str, float]

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

    @pytest.mark.parametrize(
        "config, expected_scores",
        [
            TestCaseEvaluate(
                config=BUTTER_FINGER_CONFIG,
                expected_scores={
                    ROUGE_SCORE: 0.0,
                    METEOR_SCORE: 0.055555,
                    BERT_SCORE: 0.590943,
                    DELTA_ROUGE_SCORE: 0.0,
                    DELTA_METEOR_SCORE: 0.0,
                    DELTA_BERT_SCORE: -0.022468,
                },
            ),
            TestCaseEvaluate(
                config=RANDOM_UPPER_CASE_CONFIG,
                expected_scores={
                    ROUGE_SCORE: 0.0,
                    METEOR_SCORE: 0.055555,
                    BERT_SCORE: 0.590943,
                    DELTA_ROUGE_SCORE: 0.0,
                    DELTA_METEOR_SCORE: 0.011111,
                    DELTA_BERT_SCORE: -0.015139,
                },
            ),
            TestCaseEvaluate(
                config=WHITESPACE_CONFIG,
                expected_scores={
                    ROUGE_SCORE: 0.0,
                    METEOR_SCORE: 0.055555,
                    BERT_SCORE: 0.590943,
                    DELTA_ROUGE_SCORE: 0.0,
                    DELTA_METEOR_SCORE: 0.016667,
                    DELTA_BERT_SCORE: 0.009372,
                },
            ),
        ],
    )
    def test_evaluate_sample(self, config, expected_scores, integration_tests_dir):
        eval_algo = SummarizationAccuracySemanticRobustness(config)
        with open(os.path.join(integration_tests_dir, "datasets", "xsum_sample.jsonl")) as fh:
            json_obj = json.loads(fh.readline())
            model_input = json_obj["document"]
            target_output = json_obj["summary"]
            eval_scores = eval_algo.evaluate_sample(
                model_input=model_input,
                model=sm_model_runner,
                target_output=target_output,
            )
            for eval_score in eval_scores:
                assert eval_score.value == approx(expected_scores[eval_score.name], abs=ABS_TOL)

    @pytest.mark.parametrize(
        "config, expected_scores",
        [
            TestCaseEvaluate(
                config=BUTTER_FINGER_CONFIG,
                expected_scores={
                    ROUGE_SCORE: 0.047974,
                    METEOR_SCORE: 0.085727,
                    BERT_SCORE: 0.571300,
                    DELTA_ROUGE_SCORE: 0.012338,
                    DELTA_METEOR_SCORE: 0.010314,
                    DELTA_BERT_SCORE: 0.007925,
                },
            ),
            TestCaseEvaluate(
                config=RANDOM_UPPER_CASE_CONFIG,
                expected_scores={
                    ROUGE_SCORE: 0.047974,
                    METEOR_SCORE: 0.085727,
                    BERT_SCORE: 0.571300,
                    DELTA_ROUGE_SCORE: 0.013651,
                    DELTA_METEOR_SCORE: 0.012043,
                    DELTA_BERT_SCORE: 0.006258,
                },
            ),
            TestCaseEvaluate(
                config=WHITESPACE_CONFIG,
                expected_scores={
                    ROUGE_SCORE: 0.047974,
                    METEOR_SCORE: 0.085727,
                    BERT_SCORE: 0.571300,
                    DELTA_ROUGE_SCORE: 0.006944,
                    DELTA_METEOR_SCORE: 0.005448,
                    DELTA_BERT_SCORE: -0.001602,
                },
            ),
        ],
    )
    def test_evaluate(self, integration_tests_dir, config, expected_scores):
        eval_algo = SummarizationAccuracySemanticRobustness(config)
        dataset_config = DataConfig(
            dataset_name="xsum_sample",
            dataset_uri=os.path.join(integration_tests_dir, "datasets", "xsum_sample.jsonl"),
            dataset_mime_type=MIME_TYPE_JSONLINES,
            model_input_location="document",
            target_output_location="summary",
        )
        eval_output = eval_algo.evaluate(
            model=sm_model_runner,
            dataset_config=dataset_config,
            save=True,
        )[0]
        for eval_score in eval_output.dataset_scores:
            assert eval_score.value == approx(expected_scores[eval_score.name], abs=ABS_TOL)

    def test_ray_shutdown(self):
        """
        Forcefully shut down the Ray session to ensure that resources
        consumed by this session (most importantly, the BertscoreHelperModel
        Actor, which consumes a lot of memory) get freed.
        """
        ray.shutdown()
