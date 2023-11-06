import os
from typing import NamedTuple, Dict

import pytest
from pytest import approx
from amazon_fmeval.constants import MIME_TYPE_JSONLINES

from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.eval_algorithms.general_semantic_robustness import (
    BUTTER_FINGER,
    GeneralSemanticRobustness,
    GeneralSemanticRobustnessConfig,
    RANDOM_UPPER_CASE,
    WER_SCORE,
    WHITESPACE_ADD_REMOVE,
)
from test.integration.conftest import integration_tests_dir

from test.integration.models.model_runners import (
    sm_model_runner,  # use this or hf model runner?
    sm_model_runner_prompt_template,
)

ABS_TOL = 1e-4
os.environ["PARALLELIZATION_FACTOR"] = "2"


class GSRTestCase(NamedTuple):
    config: GeneralSemanticRobustnessConfig
    expected_scores : Dict[str, float]

# TODO: deep dive into how WER_SCORE is calculated

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
                    WER_SCORE : 1.0
                },
            ),
            GSRTestCase(
                config=GeneralSemanticRobustnessConfig(
                    perturbation_type=RANDOM_UPPER_CASE,
                    num_perturbations=5,
                    random_uppercase_corrupt_proportion=0.1,
                ),
                expected_scores={
                    WER_SCORE : 2.0

                },
            ),
            GSRTestCase(
                config=GeneralSemanticRobustnessConfig(
                    perturbation_type=WHITESPACE_ADD_REMOVE,
                    num_perturbations=5,
                    whitespace_remove_prob=0.1,
                    whitespace_add_prob=0.05,
                ),
                expected_scores={
                    WER_SCORE : 1.0 
                },
            ),
        ],
    )
    def test_evaluate_sample(self, gsr_test_case):
        gen_semantic_robustness = GeneralSemanticRobustness(gsr_test_case.config)
        model_input = "London is the capital of "
        eval_scores = gen_semantic_robustness.evaluate_sample(
            model_input=model_input,
            model=sm_model_runner,  # use this or hf model runner?
            # no target output? what about model output?
            prompt_template=sm_model_runner_prompt_template,
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
                expected_scores={
                    WER_SCORE : 1.1538023088023088
                },
            ),
            GSRTestCase(
                config=GeneralSemanticRobustnessConfig(
                    perturbation_type=RANDOM_UPPER_CASE,
                    num_perturbations=5,
                    random_uppercase_corrupt_proportion=0.1,
                ),
                expected_scores={
                    WER_SCORE : 0.9224410774410775

                },
            ),
            GSRTestCase(
                config=GeneralSemanticRobustnessConfig(
                    perturbation_type=WHITESPACE_ADD_REMOVE,
                    num_perturbations=5,
                    whitespace_remove_prob=0.1,
                    whitespace_add_prob=0.05,
                ),
                expected_scores={
                    WER_SCORE : 0.8698067981401316 
                },
            ),
        ],
    )        
    def test_evaluate(self, integration_tests_dir, gsr_test_case):
        gen_semantic_robustness = GeneralSemanticRobustness(gsr_test_case.config)
        dataset_config = DataConfig(
            dataset_name="triviaQA_sample",
            dataset_uri=os.path.join(integration_tests_dir, "datasets", "triviaQA_sample.jsonl"),
            dataset_mime_type=MIME_TYPE_JSONLINES,
            model_input_location="question",
            target_output_location="answer",
        )
        eval_output = gen_semantic_robustness.evaluate(
            model=sm_model_runner,
            dataset_config=dataset_config,
            prompt_template=sm_model_runner_prompt_template,
            save=True,
        )[0]
        
        for eval_score in eval_output.dataset_scores:
            assert eval_score.value == approx(gsr_test_case.expected_scores[eval_score.name], abs=ABS_TOL)
        