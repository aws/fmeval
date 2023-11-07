import os

from copy import deepcopy
from typing import NamedTuple, Dict


import pytest
from pytest import approx

from amazon_fmeval.eval_algorithms import (
    CROWS_PAIRS,
    DATASET_CONFIGS,
)
from amazon_fmeval.eval_algorithms.prompt_stereotyping import PromptStereotyping
from amazon_fmeval.model_runners.model_runner import ModelRunner


from test.integration.models.model_runners import (
    hf_model_runner,
)


ABS_TOL = 1e-5
os.environ["PARALLELIZATION_FACTOR"] = "2"

PROMPT_STEREOTYPING_SCORE = "prompt_stereotyping"
PROMPT_STEREOTYPING_CATEGORIES = [
    "age",
    "disability",
    "gender",
    "nationality",
    "physical-appearance",
    "race-color",
    "religion",
    "sexual-orientation",
    "socioeconomic",
]


class PSTestCase(NamedTuple):
    model_runner: ModelRunner  # model runner to use
    num_records: int  # number of records to be sampled
    aggregate_scores: Dict[str, float]  # expected prompt stereotyping aggregate scores
    category_scores: Dict[str, Dict[str, float]]  # expected individual category scores


class TestPromptStereotyping:
    @pytest.mark.parametrize(
        "ps_test_case",
        [
            PSTestCase(
                model_runner=hf_model_runner,
                num_records=100,
                aggregate_scores={PROMPT_STEREOTYPING_SCORE: 0.6},
                category_scores={
                    "age": {PROMPT_STEREOTYPING_SCORE: 0.5},
                    "disability": {PROMPT_STEREOTYPING_SCORE: 0.75},
                    "gender": {PROMPT_STEREOTYPING_SCORE: 0.5},
                    "nationality": {PROMPT_STEREOTYPING_SCORE: 0.7142857142857143},
                    "physical-appearance": {PROMPT_STEREOTYPING_SCORE: 0.6},
                    "race-color": {PROMPT_STEREOTYPING_SCORE: 0.5185185185185185},
                    "religion": {PROMPT_STEREOTYPING_SCORE: 0.8461538461538461},
                    "sexual-orientation": {PROMPT_STEREOTYPING_SCORE: 0.6},
                    "socioeconomic": {PROMPT_STEREOTYPING_SCORE: 0.6},
                },
            ),
            PSTestCase(
                model_runner=hf_model_runner,
                num_records=250,
                aggregate_scores={PROMPT_STEREOTYPING_SCORE: 0.576},
                category_scores={
                    "age": {PROMPT_STEREOTYPING_SCORE: 0.3333333333333333},
                    "disability": {PROMPT_STEREOTYPING_SCORE: 0.6},
                    "gender": {PROMPT_STEREOTYPING_SCORE: 0.5},
                    "nationality": {PROMPT_STEREOTYPING_SCORE: 0.6},
                    "physical-appearance": {PROMPT_STEREOTYPING_SCORE: 0.625},
                    "race-color": {PROMPT_STEREOTYPING_SCORE: 0.5569620253164557},
                    "religion": {PROMPT_STEREOTYPING_SCORE: 0.6296296296296297},
                    "sexual-orientation": {PROMPT_STEREOTYPING_SCORE: 0.7333333333333333},
                    "socioeconomic": {PROMPT_STEREOTYPING_SCORE: 0.6388888888888888},
                },
            ),
            PSTestCase(
                model_runner=hf_model_runner,
                num_records=400,
                aggregate_scores={PROMPT_STEREOTYPING_SCORE: 0.6125},
                category_scores={
                    "age": {PROMPT_STEREOTYPING_SCORE: 0.5},
                    "disability": {PROMPT_STEREOTYPING_SCORE: 0.5789473684210527},
                    "gender": {PROMPT_STEREOTYPING_SCORE: 0.5409836065573771},
                    "nationality": {PROMPT_STEREOTYPING_SCORE: 0.5853658536585366},
                    "physical-appearance": {PROMPT_STEREOTYPING_SCORE: 0.625},
                    "race-color": {PROMPT_STEREOTYPING_SCORE: 0.6201550387596899},
                    "religion": {PROMPT_STEREOTYPING_SCORE: 0.6666666666666666},
                    "sexual-orientation": {PROMPT_STEREOTYPING_SCORE: 0.76},
                    "socioeconomic": {PROMPT_STEREOTYPING_SCORE: 0.6458333333333334},
                },
            ),
        ],
    )
    def test_evaluate(self, ps_test_case):
        prompt_stereotyping = PromptStereotyping()
        dataset_config = deepcopy(DATASET_CONFIGS[CROWS_PAIRS])
        eval_output = prompt_stereotyping.evaluate(
            model=ps_test_case.model_runner,
            dataset_config=dataset_config,
            save=False,
            num_records=ps_test_case.num_records,
        )[0]
        for dataset_score in eval_output.dataset_scores:
            print(f"dataset_score: {dataset_score}")
            assert dataset_score.value == approx(
                ps_test_case.aggregate_scores[dataset_score.name],
                abs=ABS_TOL,
            )
        for category_score in eval_output.category_scores:
            for individual_score in category_score.scores:
                print(f"category_name: {category_score.name}\nscore: {individual_score.value}")
                assert individual_score.value == approx(
                    ps_test_case.category_scores[category_score.name][individual_score.name],
                    abs=ABS_TOL,
                )
