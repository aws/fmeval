import os
import ray
import json
import pytest

from typing import NamedTuple, Dict
from pytest import approx
from fmeval.eval_algorithms.summarization_accuracy_semantic_robustness import (
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
from fmeval.data_loaders.data_config import DataConfig
from fmeval.constants import MIME_TYPE_JSONLINES
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


class TestCaseEvaluate(NamedTuple):
    config: SummarizationAccuracySemanticRobustnessConfig
    expected_evaluate_sample_scores: Dict[str, float]
    expected_evaluate_scores: Dict[str, float]


class TestSummarizationAccuracySemanticRobustness:
    @pytest.mark.parametrize(
        "config, expected_evaluate_sample_scores, expected_evaluate_scores",
        [
            TestCaseEvaluate(
                config=BUTTER_FINGER_CONFIG,
                expected_evaluate_sample_scores={
                    ROUGE_SCORE: 0.0,
                    METEOR_SCORE: 0.055555,
                    BERT_SCORE: 0.578891,
                    DELTA_ROUGE_SCORE: 0.0,
                    DELTA_METEOR_SCORE: 0.0,
                    DELTA_BERT_SCORE: 0.020095,
                },
                expected_evaluate_scores={
                    ROUGE_SCORE: 0.067589,
                    METEOR_SCORE: 0.110135,
                    BERT_SCORE: 0.593341,
                    DELTA_ROUGE_SCORE: 0.039784,
                    DELTA_METEOR_SCORE: 0.045444,
                    DELTA_BERT_SCORE: 0.049985,
                },
            ),
            TestCaseEvaluate(
                config=RANDOM_UPPER_CASE_CONFIG,
                expected_evaluate_sample_scores={
                    ROUGE_SCORE: 0.0,
                    METEOR_SCORE: 0.055555,
                    BERT_SCORE: 0.578891,
                    DELTA_ROUGE_SCORE: 0.0,
                    DELTA_METEOR_SCORE: 0.011111,
                    DELTA_BERT_SCORE: 0.017104,
                },
                expected_evaluate_scores={
                    ROUGE_SCORE: 0.067589,
                    METEOR_SCORE: 0.110135,
                    BERT_SCORE: 0.593341,
                    DELTA_ROUGE_SCORE: 0.038557,
                    DELTA_METEOR_SCORE: 0.046560,
                    DELTA_BERT_SCORE: 0.043057,
                },
            ),
            TestCaseEvaluate(
                config=WHITESPACE_CONFIG,
                expected_evaluate_sample_scores={
                    ROUGE_SCORE: 0.0,
                    METEOR_SCORE: 0.055555,
                    BERT_SCORE: 0.578891,
                    DELTA_ROUGE_SCORE: 0.0,
                    DELTA_METEOR_SCORE: 0.016667,
                    DELTA_BERT_SCORE: 0.009599,
                },
                expected_evaluate_scores={
                    ROUGE_SCORE: 0.067589,
                    METEOR_SCORE: 0.110135,
                    BERT_SCORE: 0.593341,
                    DELTA_ROUGE_SCORE: 0.025756,
                    DELTA_BERT_SCORE: 0.039346,
                    DELTA_METEOR_SCORE: 0.033426,
                },
            ),
        ],
    )
    def test_evaluate_sample_and_evaluate(
        self, config, expected_evaluate_sample_scores, expected_evaluate_scores, integration_tests_dir
    ):
        """
        In order to reuse SummarizationAccuracySemanticRobustness objects
        as much as possible (to minimize creation of BertscoreHelperModels),
        we test evaluate_sample and evaluate back to back using the same eval_algo
        (instead of following the convention of the other tests, where evaluate_sample
        and evaluate are tested in separate methods).
        """
        eval_algo = SummarizationAccuracySemanticRobustness(config)
        # Test evaluate_sample
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
                assert eval_score.value == approx(expected_evaluate_sample_scores[eval_score.name], abs=ABS_TOL)

        # Test evaluate
        dataset_config = DataConfig(
            dataset_name="xsum_sample",
            dataset_uri=os.path.join(integration_tests_dir, "datasets", "xsum_sample_small.jsonl"),
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
            assert eval_score.value == approx(expected_evaluate_scores[eval_score.name], abs=ABS_TOL)
        # Calling ray.shutdown() would be overkill since there are still other test cases.
        # Thus, we kill only the SummarizationAccuracySingleton ray actor used by the
        # current test case, to make sure that resources are cleaned up between test cases.
        ray.kill(eval_algo._summarization_accuracy_eval_algo)
