import os
import pytest

from pytest import approx
from typing import NamedTuple, Dict

from fmeval.eval_algorithms import (
    DATASET_CONFIGS,
    WIKITEXT2,
)
from fmeval.eval_algorithms.general_semantic_robustness import (
    BUTTER_FINGER,
    RANDOM_UPPERCASE,
    ADD_REMOVE_WHITESPACE,
    GeneralSemanticRobustness,
    GeneralSemanticRobustnessConfig,
    WER_SCORE,
    BERT_SCORE_DISSIMILARITY,
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
                    WER_SCORE: 0.5666666666666667,
                    BERT_SCORE_DISSIMILARITY: 0.19456744194030762,
                },
            ),
            GSRTestCase(
                config=GeneralSemanticRobustnessConfig(
                    perturbation_type=RANDOM_UPPERCASE,
                    num_perturbations=3,
                    random_uppercase_fraction=0.1,
                ),
                expected_scores={
                    WER_SCORE: 0.7777777777777778,
                    BERT_SCORE_DISSIMILARITY: 0.18904542922973633,
                },
            ),
            GSRTestCase(
                config=GeneralSemanticRobustnessConfig(
                    perturbation_type=ADD_REMOVE_WHITESPACE,
                    num_perturbations=2,
                    whitespace_add_prob=0.05,
                    whitespace_remove_prob=0.1,
                ),
                expected_scores={
                    WER_SCORE: 0.3333333333333333,
                    BERT_SCORE_DISSIMILARITY: 0.14486145973205566,
                },
            ),
        ],
    )
    def test_evaluate_sample(self, gsr_test_case):
        gen_semantic_robustness = GeneralSemanticRobustness(gsr_test_case.config, use_ray=False)
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
                    perturbation_type=ADD_REMOVE_WHITESPACE,
                    num_perturbations=5,
                    whitespace_remove_prob=0.1,
                    whitespace_add_prob=0.05,
                ),
                expected_scores={
                    WER_SCORE: 0.4727531746031748,
                    BERT_SCORE_DISSIMILARITY: 0.2137835907936097,
                },
            ),
        ],
    )
    def test_evaluate(self, gsr_test_case):
        gen_semantic_robustness = GeneralSemanticRobustness(gsr_test_case.config)
        dataset_config = DATASET_CONFIGS[WIKITEXT2]
        eval_output = gen_semantic_robustness.evaluate(
            model=sm_model_runner,
            dataset_config=dataset_config,
            save=True,
        )[0]

        for eval_score in eval_output.dataset_scores:
            assert eval_score.value == approx(gsr_test_case.expected_scores[eval_score.name], abs=ABS_TOL)
