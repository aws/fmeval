from typing import NamedTuple, List, Union

import pytest

from amazon_fmeval.eval_algorithms.semantic_perturbation_utils import (
    ButterFinger,
    ButterFingerConfig,
    RandomUpperCaseConfig,
    WhitespaceAddRemoveConfig,
    RandomUpperCase,
    WhitespaceAddRemove,
)


class TestSemanticPerturbationUtils:
    class TestData(NamedTuple):
        seed: int
        input_text: str
        expected_outputs: List[str]
        config: Union[ButterFingerConfig, RandomUpperCaseConfig, WhitespaceAddRemoveConfig]
        num_perturbations: int

    @pytest.mark.parametrize(
        "test_case",
        [
            TestData(
                seed=3,
                input_text="A quick brown fox jumps over the lazy dog 10 times.",
                expected_outputs=[
                    "A quick bcmwn fox jumps over bhr lwzj dog 10 times.",
                    "A auick brown fox jumps over the lazy dog 10 timef.",
                    "A quick brown flz jujps ovef the lazy dog 10 times.",
                ],
                config=ButterFingerConfig(),
                num_perturbations=3,
            ),
            TestData(
                seed=10,
                input_text="A quick brown fox jumps over the lazy dog 10 times.",
                expected_outputs=[
                    "A qujck brown fov jumps over dhe lavg dog 10 times.",
                    "A qhick brocm fox jukps over tie pazy dog 10 times.",
                ],
                config=ButterFingerConfig(perturbation_prob=0.2),
                num_perturbations=2,
            ),
        ],
    )
    def test_butter_finger(self, test_case):
        assert (
            ButterFinger(seed=test_case.seed).perturb(
                test_case.input_text, test_case.config, test_case.num_perturbations
            )
            == test_case.expected_outputs
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            TestData(
                seed=3,
                input_text="A quick brown fox jumps over the lazy dog 10 times.",
                expected_outputs=[
                    "A quick bRowN fox jumps over the lazy dOG 10 timEs.",
                    "A quicK brown fox jUmps Over tHe lazy dog 10 Times.",
                    "A qUicK brown fox jumps over thE lazy Dog 10 timEs.",
                ],
                config=RandomUpperCaseConfig(),
                num_perturbations=3,
            ),
            TestData(
                seed=10,
                input_text="A quick brown fox jumps over the lazy dog 10 times.",
                expected_outputs=[
                    "A qUick brown fox juMPs oveR thE lazy Dog 10 TimEs.",
                    "A quick brown foX jUMps ovER the lazY dog 10 tiMes.",
                ],
                config=RandomUpperCaseConfig(corrupt_proportion=0.2),
                num_perturbations=2,
            ),
        ],
    )
    def test_random_upper_case(self, test_case):
        assert (
            RandomUpperCase(seed=test_case.seed).perturb(
                test_case.input_text, test_case.config, test_case.num_perturbations
            )
            == test_case.expected_outputs
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            TestData(
                seed=3,
                input_text="A quick brown fox jumps over the lazy dog 10 times.",
                expected_outputs=[
                    "A quick  brown fox jumps ov er the lazydog 10 times.",
                    "A quick brown foxjumps o ve r the lazy  dog10 times.",
                    "A quick brow n f oxjumps o ver the lazy do g 10 times.",
                ],
                config=WhitespaceAddRemoveConfig(),
                num_perturbations=3,
            ),
            TestData(
                seed=10,
                input_text="A quick brown fox jumps over the lazy dog 10 times.",
                expected_outputs=[
                    "A qu ickbr o wnfox  jumps  over t he  la zydog  10  times .",
                    "A  q ui c k bro w nf ox  jump sov e r  th e l azy d og 10 ti mes. ",
                ],
                config=WhitespaceAddRemoveConfig(remove_prob=0.2, add_prob=0.4),
                num_perturbations=2,
            ),
        ],
    )
    def test_whitespace_add_remove(self, test_case):
        assert (
            WhitespaceAddRemove(seed=test_case.seed).perturb(
                test_case.input_text, test_case.config, test_case.num_perturbations
            )
            == test_case.expected_outputs
        )

    @pytest.mark.parametrize(
        "test_case, perturbation_type",
        [
            (
                TestData(
                    seed=5,
                    input_text="A quick brown fox jumps over the lazy dog 10 times.",
                    expected_outputs=[
                        "A quick trowm fox jumpa over tne lazy dog 10 times.",
                        "A quicy brodn fud jumps oveg tke lasj dog 10 times.",
                        "A quick bwown fox jumps ovev the lazy dkg 10 times.",
                    ],
                    config=ButterFingerConfig(),
                    num_perturbations=3,
                ),
                ButterFinger,
            ),
            (
                TestData(
                    seed=5,
                    input_text="A quick brown fox jumps over the lazy dog 10 times.",
                    expected_outputs=[
                        "A quick  brown fox  jumps ove r the lazy dog 10 time s.",
                        "A quick b row n fo x  jumps over the  lazy dog 10 time s.",
                        "A quick b rown fox jumps over the lazy dog 10 times.",
                    ],
                    config=WhitespaceAddRemoveConfig(),
                    num_perturbations=3,
                ),
                WhitespaceAddRemove,
            ),
            (
                TestData(
                    seed=5,
                    input_text="A quick brown fox jumps over the lazy dog 10 times.",
                    expected_outputs=[
                        "A quick brown fox jUmps over The Lazy doG 10 times.",
                        "A quIck brown fox jumps over the lazy dog 10 times.",
                        "A quick brown Fox jumps over the lAzy dog 10 tImes.",
                    ],
                    config=RandomUpperCaseConfig(),
                    num_perturbations=3,
                ),
                RandomUpperCase,
            ),
        ],
    )
    def test_default_seed(self, test_case, perturbation_type):
        assert (
            perturbation_type().perturb(test_case.input_text, test_case.config, test_case.num_perturbations)
            == test_case.expected_outputs
        )
