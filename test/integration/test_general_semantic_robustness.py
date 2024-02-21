import os
from copy import deepcopy
from typing import NamedTuple, Dict

import pytest
from pytest import approx

from fmeval.eval_algorithms import (
    DATASET_CONFIGS,
    WIKITEXT2,
)
from fmeval.eval_algorithms.general_semantic_robustness import (
    BUTTER_FINGER,
    GeneralSemanticRobustness,
    GeneralSemanticRobustnessConfig,
    WER_SCORE,
    BERT_SCORE_DISSIMILARITY,
    WHITESPACE_ADD_REMOVE,
)

from test.integration.models.model_runners import (
    sm_model_runner,
)

ABS_TOL = 1e-4
os.environ["PARALLELIZATION_FACTOR"] = "2"


class GSRTestCase(NamedTuple):
    config: GeneralSemanticRobustnessConfig
    expected_scores: Dict[str, float]


class TestGeneralSemanticRobustness:
    @pytest.mark.parametrize(
        "gsr_test_case",
        [
            GSRTestCase(
                config=GeneralSemanticRobustnessConfig(
                    perturbation_type=BUTTER_FINGER,
                    num_perturbations=5,
                    butter_finger_perturbation_prob=0.1,
                ),
                expected_scores={
                    WER_SCORE: 1.1,
                    BERT_SCORE_DISSIMILARITY: 0.4326739013195038,
                },
            )
        ],
    )
    def test_evaluate_sample(self, gsr_test_case):
        gen_semantic_robustness = GeneralSemanticRobustness(gsr_test_case.config)
        model_input = "London is the capital of "
        eval_scores = gen_semantic_robustness.evaluate_sample(
            model_input=model_input,
            model=sm_model_runner,
        )
        for eval_score in eval_scores:
            assert eval_score.value == approx(gsr_test_case.expected_scores[eval_score.name], abs=ABS_TOL)

    @pytest.mark.parametrize(
        "gsr_test_case",
        [
            GSRTestCase(
                config=GeneralSemanticRobustnessConfig(
                    perturbation_type=WHITESPACE_ADD_REMOVE,
                    num_perturbations=5,
                    whitespace_remove_prob=0.1,
                    whitespace_add_prob=0.05,
                ),
                expected_scores={
                    WER_SCORE: 0.6135412698412699,
                    BERT_SCORE_DISSIMILARITY: 0.2815050273537636,
                },
            ),
        ],
    )
    def test_evaluate(self, gsr_test_case):
        gen_semantic_robustness = GeneralSemanticRobustness(gsr_test_case.config)
        dataset_config = deepcopy(DATASET_CONFIGS[WIKITEXT2])
        eval_output = gen_semantic_robustness.evaluate(
            model=sm_model_runner,
            dataset_config=dataset_config,
            save=True,
        )[0]

        for eval_score in eval_output.dataset_scores:
            assert eval_score.value == approx(gsr_test_case.expected_scores[eval_score.name], abs=ABS_TOL)
