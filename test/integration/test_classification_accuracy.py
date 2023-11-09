import os

from typing import NamedTuple, Dict


import pytest
from pytest import approx

from fmeval.eval_algorithms import (
    DATASET_CONFIGS,
    WOMENS_CLOTHING_ECOMMERCE_REVIEWS,
)
from fmeval.eval_algorithms.classification_accuracy import (
    BALANCED_ACCURACY_SCORE,
    CLASSIFICATION_ACCURACY_SCORE,
    ClassificationAccuracy,
    ClassificationAccuracyConfig,
    PRECISION_SCORE,
    RECALL_SCORE,
)

from test.integration.models.model_runners import (
    hf_model_runner,
)

ABS_TOL = 1e-4
os.environ["PARALLELIZATION_FACTOR"] = "2"

SAMPLE_VALID_LABELS = ["0", "1"]


class CATestCase(NamedTuple):
    config: ClassificationAccuracyConfig
    aggregate_scores: Dict[str, float]
    category_scores: Dict[str, Dict[str, float]]  # expected individual category scores


class TestClassificationAccuracy:
    @pytest.mark.parametrize(
        "ca_test_case",
        [
            CATestCase(
                config=ClassificationAccuracyConfig(
                    valid_labels=SAMPLE_VALID_LABELS,
                ),
                aggregate_scores={
                    BALANCED_ACCURACY_SCORE: 0.5,
                    CLASSIFICATION_ACCURACY_SCORE: 0.83,
                    PRECISION_SCORE: 0.83,
                    RECALL_SCORE: 0.83,
                },
                category_scores={
                    "Blouses": {
                        BALANCED_ACCURACY_SCORE: 0.5,
                        CLASSIFICATION_ACCURACY_SCORE: 0.8,
                        PRECISION_SCORE: 0.8,
                        RECALL_SCORE: 0.8,
                    },
                    "Dresses": {
                        BALANCED_ACCURACY_SCORE: 0.5,
                        CLASSIFICATION_ACCURACY_SCORE: 0.8571428571428571,
                        PRECISION_SCORE: 0.8571428571428571,
                        RECALL_SCORE: 0.8571428571428571,
                    },
                    "Fine gauge": {
                        BALANCED_ACCURACY_SCORE: 0.5,
                        CLASSIFICATION_ACCURACY_SCORE: 0.75,
                        PRECISION_SCORE: 0.75,
                        RECALL_SCORE: 0.75,
                    },
                    "Jackets": {
                        BALANCED_ACCURACY_SCORE: 0.0,
                        CLASSIFICATION_ACCURACY_SCORE: 0.0,
                        PRECISION_SCORE: 0.0,
                        RECALL_SCORE: 0.0,
                    },
                    "Jeans": {
                        BALANCED_ACCURACY_SCORE: 1.0,
                        CLASSIFICATION_ACCURACY_SCORE: 1.0,
                        PRECISION_SCORE: 1.0,
                        RECALL_SCORE: 1.0,
                    },
                    "Knits": {
                        BALANCED_ACCURACY_SCORE: 0.5,
                        CLASSIFICATION_ACCURACY_SCORE: 0.9166666666666666,
                        PRECISION_SCORE: 0.9166666666666666,
                        RECALL_SCORE: 0.9166666666666666,
                    },
                    "Lounge": {
                        BALANCED_ACCURACY_SCORE: 0.5,
                        CLASSIFICATION_ACCURACY_SCORE: 0.75,
                        PRECISION_SCORE: 0.75,
                        RECALL_SCORE: 0.75,
                    },
                    "Outerwear": {
                        BALANCED_ACCURACY_SCORE: 0.0,
                        CLASSIFICATION_ACCURACY_SCORE: 0.0,
                        PRECISION_SCORE: 0.0,
                        RECALL_SCORE: 0.0,
                    },
                    "Pants": {
                        BALANCED_ACCURACY_SCORE: 1.0,
                        CLASSIFICATION_ACCURACY_SCORE: 1.0,
                        PRECISION_SCORE: 1.0,
                        RECALL_SCORE: 1.0,
                    },
                    "Shorts": {
                        BALANCED_ACCURACY_SCORE: 0.5,
                        CLASSIFICATION_ACCURACY_SCORE: 0.6666666666666666,
                        PRECISION_SCORE: 0.6666666666666666,
                        RECALL_SCORE: 0.6666666666666666,
                    },
                    "Skirts": {
                        BALANCED_ACCURACY_SCORE: 0.5,
                        CLASSIFICATION_ACCURACY_SCORE: 0.8,
                        PRECISION_SCORE: 0.8,
                        RECALL_SCORE: 0.8,
                    },
                    "Sweaters": {
                        BALANCED_ACCURACY_SCORE: 0.5,
                        CLASSIFICATION_ACCURACY_SCORE: 0.7142857142857143,
                        PRECISION_SCORE: 0.7142857142857143,
                        RECALL_SCORE: 0.7142857142857143,
                    },
                    "Swim": {
                        BALANCED_ACCURACY_SCORE: 1.0,
                        CLASSIFICATION_ACCURACY_SCORE: 1.0,
                        PRECISION_SCORE: 1.0,
                        RECALL_SCORE: 1.0,
                    },
                    "Trend": {
                        BALANCED_ACCURACY_SCORE: 1.0,
                        CLASSIFICATION_ACCURACY_SCORE: 1.0,
                        PRECISION_SCORE: 1.0,
                        RECALL_SCORE: 1.0,
                    },
                },
            ),
        ],
    )
    def test_evaluate(self, ca_test_case):
        class_acc = ClassificationAccuracy(eval_algorithm_config=ca_test_case.config)
        prompt_template = "Classify the sentiment of the following review with 0 (negative sentiment) "
        "or 1 (positive sentiment). Review: $feature. Classification:"
        dataset_config = DATASET_CONFIGS[WOMENS_CLOTHING_ECOMMERCE_REVIEWS]
        eval_output = class_acc.evaluate(
            model=hf_model_runner,
            dataset_config=dataset_config,
            prompt_template=prompt_template,
            save=False,
        )[0]
        for dataset_score in eval_output.dataset_scores:
            assert dataset_score.value == approx(
                ca_test_case.aggregate_scores[dataset_score.name],
                abs=ABS_TOL,
            )
        for category_score in eval_output.category_scores:
            for individual_score in category_score.scores:
                assert individual_score.value == approx(
                    ca_test_case.category_scores[category_score.name][individual_score.name],
                    abs=ABS_TOL,
                )
