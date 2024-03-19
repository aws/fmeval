import os
import re

from typing import (
    Dict,
    List,
    NamedTuple,
)


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

import pytest
from pytest import approx
from test.integration.models.model_runners import (
    sm_model_runner,
)

ABS_TOL = 1e-4
os.environ["PARALLELIZATION_FACTOR"] = "2"

SAMPLE_VALID_LABELS = ["0", "1", "Negative", "Positive"]

TEST_PROMPT_TEMPLATE = (
    "<s>[INST] <<SYS>>Classify the sentiment of the following review as either "
    "0 (negative sentiment) or 1 (positive sentiment). Be brief. Do not restate "
    "the prompt. <</SYS>> Review: $model_input. [/INST]"
)


class CATestCase(NamedTuple):
    config: ClassificationAccuracyConfig
    aggregate_scores: Dict[str, float]
    category_scores: Dict[str, Dict[str, float]]  # expected individual category scores


def converter_func_wrapper(model_output: str, valid_labels: List[str]) -> str:
    label_output = general_converter_func(model_output, valid_labels)
    if label_output == "positive":
        return "1"
    if label_output == "negative":
        return "0"
    return label_output


def general_converter_func(model_output: str, valid_labels: List[str]) -> str:
    # normalise to lowercase & strip
    valid_labels = [label.lower().strip() for label in valid_labels]

    response_words = model_output.split(" ")
    # process the response words by removing non-alphanumeric characters
    processed_words = [re.sub(r"\W+", "", word.lower().strip()) for word in response_words]
    # determine predicted labels
    predicted_labels = [word for word in processed_words if word.lower().strip() in valid_labels]
    # if there is more than one label in the model output we pick the first
    string_label = predicted_labels[0] if predicted_labels else "unknown"

    return string_label


class TestClassificationAccuracy:
    @pytest.mark.parametrize(
        "ca_test_case",
        [
            CATestCase(
                config=ClassificationAccuracyConfig(
                    valid_labels=SAMPLE_VALID_LABELS,
                    converter_fn=converter_func_wrapper,
                ),
                aggregate_scores={
                    BALANCED_ACCURACY_SCORE: 0.7994330262225372,
                    CLASSIFICATION_ACCURACY_SCORE: 0.9,
                    PRECISION_SCORE: 0.9,
                    RECALL_SCORE: 0.9,
                },
                category_scores={
                    "Blouses": {
                        BALANCED_ACCURACY_SCORE: 1.0,
                        CLASSIFICATION_ACCURACY_SCORE: 1.0,
                        PRECISION_SCORE: 1.0,
                        RECALL_SCORE: 1.0,
                    },
                    "Dresses": {
                        BALANCED_ACCURACY_SCORE: 0.7291666666666667,
                        CLASSIFICATION_ACCURACY_SCORE: 0.8928571428571429,
                        PRECISION_SCORE: 0.8928571428571429,
                        RECALL_SCORE: 0.8928571428571429,
                    },
                    "Fine gauge": {
                        BALANCED_ACCURACY_SCORE: 0.5,
                        CLASSIFICATION_ACCURACY_SCORE: 0.75,
                        PRECISION_SCORE: 0.75,
                        RECALL_SCORE: 0.75,
                    },
                    "Jackets": {
                        BALANCED_ACCURACY_SCORE: 0.5,
                        CLASSIFICATION_ACCURACY_SCORE: 0.5,
                        PRECISION_SCORE: 0.5,
                        RECALL_SCORE: 0.5,
                    },
                    "Jeans": {
                        BALANCED_ACCURACY_SCORE: 1.0,
                        CLASSIFICATION_ACCURACY_SCORE: 1.0,
                        PRECISION_SCORE: 1.0,
                        RECALL_SCORE: 1.0,
                    },
                    "Knits": {
                        BALANCED_ACCURACY_SCORE: 0.7045454545454546,
                        CLASSIFICATION_ACCURACY_SCORE: 0.875,
                        PRECISION_SCORE: 0.875,
                        RECALL_SCORE: 0.875,
                    },
                    "Lounge": {
                        BALANCED_ACCURACY_SCORE: 1.0,
                        CLASSIFICATION_ACCURACY_SCORE: 1.0,
                        PRECISION_SCORE: 1.0,
                        RECALL_SCORE: 1.0,
                    },
                    "Outerwear": {
                        BALANCED_ACCURACY_SCORE: 1.0,
                        CLASSIFICATION_ACCURACY_SCORE: 1.0,
                        PRECISION_SCORE: 1.0,
                        RECALL_SCORE: 1.0,
                    },
                    "Pants": {
                        BALANCED_ACCURACY_SCORE: 0.75,
                        CLASSIFICATION_ACCURACY_SCORE: 0.75,
                        PRECISION_SCORE: 0.75,
                        RECALL_SCORE: 0.75,
                    },
                    "Shorts": {
                        BALANCED_ACCURACY_SCORE: 1.0,
                        CLASSIFICATION_ACCURACY_SCORE: 1.0,
                        PRECISION_SCORE: 1.0,
                        RECALL_SCORE: 1.0,
                    },
                    "Skirts": {
                        BALANCED_ACCURACY_SCORE: 1.0,
                        CLASSIFICATION_ACCURACY_SCORE: 1.0,
                        PRECISION_SCORE: 1.0,
                        RECALL_SCORE: 1.0,
                    },
                    "Sweaters": {
                        BALANCED_ACCURACY_SCORE: 0.75,
                        CLASSIFICATION_ACCURACY_SCORE: 0.8571428571428571,
                        PRECISION_SCORE: 0.8571428571428571,
                        RECALL_SCORE: 0.8571428571428571,
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
        dataset_config = DATASET_CONFIGS[WOMENS_CLOTHING_ECOMMERCE_REVIEWS]
        eval_output = class_acc.evaluate(
            model=sm_model_runner,
            dataset_config=dataset_config,
            prompt_template=TEST_PROMPT_TEMPLATE,
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
