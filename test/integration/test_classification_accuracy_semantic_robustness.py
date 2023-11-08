import os

from copy import deepcopy
from typing import NamedTuple, Dict


import pytest
from pytest import approx

from amazon_fmeval.eval_algorithms import (
    DATASET_CONFIGS,
    WOMENS_CLOTHING_ECOMMERCE_REVIEWS,
)
from amazon_fmeval.eval_algorithms.classification_accuracy_semantic_robustness import (
    BUTTER_FINGER,
    CLASSIFICATION_ACCURACY_SCORE,
    ClassificationAccuracySemanticRobustness,
    ClassificationAccuracySemanticRobustnessConfig,
    DELTA_CLASSIFICATION_ACCURACY_SCORE,
    RANDOM_UPPER_CASE,
    WHITESPACE_ADD_REMOVE,
)

from test.integration.models.model_runners import (
    hf_model_runner,
)

ABS_TOL = 1e-4
os.environ["PARALLELIZATION_FACTOR"] = "2"

SAMPLE_VALID_LABELS = ["0", "1"]


class CASRTestCase(NamedTuple):
    config: ClassificationAccuracySemanticRobustnessConfig
    expected_scores: Dict[str, float]


class TestClassificationAccuracySemanticRobustness:
    @pytest.mark.parametrize(
        "casr_test_case",
        [
            CASRTestCase(
                config=ClassificationAccuracySemanticRobustnessConfig(
                    valid_labels=SAMPLE_VALID_LABELS,
                    perturbation_type=BUTTER_FINGER,
                    num_perturbations=5,
                    butter_finger_perturbation_prob=0.1,
                ),
                expected_scores={
                    CLASSIFICATION_ACCURACY_SCORE: 1,
                    DELTA_CLASSIFICATION_ACCURACY_SCORE: 0.0,
                },
            ),
            CASRTestCase(
                config=ClassificationAccuracySemanticRobustnessConfig(
                    valid_labels=SAMPLE_VALID_LABELS,
                    perturbation_type=RANDOM_UPPER_CASE,
                    num_perturbations=5,
                    random_uppercase_corrupt_proportion=0.1,
                ),
                expected_scores={
                    CLASSIFICATION_ACCURACY_SCORE: 1,
                    DELTA_CLASSIFICATION_ACCURACY_SCORE: 0,
                },
            ),
            CASRTestCase(
                config=ClassificationAccuracySemanticRobustnessConfig(
                    valid_labels=SAMPLE_VALID_LABELS,
                    perturbation_type=WHITESPACE_ADD_REMOVE,
                    num_perturbations=5,
                    whitespace_remove_prob=0.1,
                    whitespace_add_prob=0.05,
                ),
                expected_scores={
                    CLASSIFICATION_ACCURACY_SCORE: 1,
                    DELTA_CLASSIFICATION_ACCURACY_SCORE: 0,
                },
            ),
        ],
    )
    def test_evaluate_sample(self, casr_test_case):
        ca_semantic_robustness = ClassificationAccuracySemanticRobustness(eval_algorithm_config=casr_test_case.config)
        model_input = "Absolutely wonderful - silky and sexy and comfortable"
        prompt_template = "Classify the sentiment of the following review with 0 (negative sentiment) "
        "or 1 (positive sentiment). Review: $feature. Classification:"
        # model_output = sm_model_runner.predict(model_input)
        # print(model_output)
        eval_scores = ca_semantic_robustness.evaluate_sample(
            model_input=model_input,
            # model_output=model_output,
            model=hf_model_runner,
            target_output="1",
            prompt_template=prompt_template,
        )
        for eval_score in eval_scores:
            assert eval_score.value == casr_test_case.expected_scores[eval_score.name]

    @pytest.mark.parametrize(
        "casr_test_case",
        [
            CASRTestCase(
                config=ClassificationAccuracySemanticRobustnessConfig(
                    valid_labels=SAMPLE_VALID_LABELS,
                    perturbation_type=BUTTER_FINGER,
                    num_perturbations=5,
                    butter_finger_perturbation_prob=0.1,
                ),
                expected_scores={
                    CLASSIFICATION_ACCURACY_SCORE: 1,
                    DELTA_CLASSIFICATION_ACCURACY_SCORE: 0.0,
                },
            ),
            CASRTestCase(
                config=ClassificationAccuracySemanticRobustnessConfig(
                    valid_labels=SAMPLE_VALID_LABELS,
                    perturbation_type=RANDOM_UPPER_CASE,
                    num_perturbations=5,
                    random_uppercase_corrupt_proportion=0.1,
                ),
                expected_scores={
                    CLASSIFICATION_ACCURACY_SCORE: 1,
                    DELTA_CLASSIFICATION_ACCURACY_SCORE: 0,
                },
            ),
            CASRTestCase(
                config=ClassificationAccuracySemanticRobustnessConfig(
                    valid_labels=SAMPLE_VALID_LABELS,
                    perturbation_type=WHITESPACE_ADD_REMOVE,
                    num_perturbations=5,
                    whitespace_remove_prob=0.1,
                    whitespace_add_prob=0.05,
                ),
                expected_scores={
                    CLASSIFICATION_ACCURACY_SCORE: 1,
                    DELTA_CLASSIFICATION_ACCURACY_SCORE: 0,
                },
            ),
        ],
    )
    def test_evaluate_full(self, casr_test_case):
        ca_semantic_robustness = ClassificationAccuracySemanticRobustness(eval_algorithm_config=casr_test_case.config)
        prompt_template = "Classify the sentiment of the following review with 0 (negative sentiment) "
        "or 1 (positive sentiment). Review: $feature. Classification:"
        dataset_config = DATASET_CONFIGS[WOMENS_CLOTHING_ECOMMERCE_REVIEWS]
        eval_output = ca_semantic_robustness.evaluate(
            model=hf_model_runner,
            dataset_config=dataset_config,
            prompt_template=prompt_template,
            save=False,
        )[0]
        print(eval_output)
        assert False
        

