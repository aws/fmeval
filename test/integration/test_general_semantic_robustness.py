import os
import ray
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
    RANDOM_UPPER_CASE,
    WER_SCORE,
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
                expected_scores={WER_SCORE: 1.1},
            ),
            GSRTestCase(
                config=GeneralSemanticRobustnessConfig(
                    perturbation_type=RANDOM_UPPER_CASE,
                    num_perturbations=5,
                    random_uppercase_corrupt_proportion=0.1,
                ),
                expected_scores={WER_SCORE: 0.26666666666666666},
            ),
            GSRTestCase(
                config=GeneralSemanticRobustnessConfig(
                    perturbation_type=WHITESPACE_ADD_REMOVE,
                    num_perturbations=5,
                    whitespace_remove_prob=0.1,
                    whitespace_add_prob=0.05,
                ),
                expected_scores={WER_SCORE: 0.5},
            ),
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
            assert eval_score.value == gsr_test_case.expected_scores[eval_score.name]

    @pytest.mark.parametrize(
        "gsr_test_case",
        [
            GSRTestCase(
                config=GeneralSemanticRobustnessConfig(
                    perturbation_type=BUTTER_FINGER,
                    num_perturbations=5,
                    butter_finger_perturbation_prob=0.1,
                ),
                expected_scores={WER_SCORE: 0.7579873015873015},
            ),
            GSRTestCase(
                config=GeneralSemanticRobustnessConfig(
                    perturbation_type=RANDOM_UPPER_CASE,
                    num_perturbations=5,
                    random_uppercase_corrupt_proportion=0.1,
                ),
                expected_scores={WER_SCORE: 0.5560531746031746},
            ),
            GSRTestCase(
                config=GeneralSemanticRobustnessConfig(
                    perturbation_type=WHITESPACE_ADD_REMOVE,
                    num_perturbations=5,
                    whitespace_remove_prob=0.1,
                    whitespace_add_prob=0.05,
                ),
                expected_scores={WER_SCORE: 0.6135412698412699},
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

    def test_ray_shutdown(self):
        """
        Forcefully shut down Ray to ensure that resources
        used by these tests get freed.
        """
        ray.shutdown()
