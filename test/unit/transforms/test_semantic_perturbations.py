import pytest
import re
from typing import NamedTuple, List
from unittest.mock import patch

from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.transforms.semantic_perturbations import (
    ButterFinger,
    SemanticPerturbation,
    RandomUppercase,
    AddRemoveWhitespace,
)


class Dummy(SemanticPerturbation):
    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        num_perturbations: int,
        seed: int,
        pos_arg: str,
        kw_arg: str = "Hi",
    ):
        super().__init__(input_keys, output_keys, num_perturbations, seed, pos_arg, kw_arg=kw_arg)

    def perturb(self, text: str) -> List[str]:
        return [f"dummy_{i}" for i in range(self.num_perturbations)]


def test_semantic_perturbation_init_success():
    """
    GIVEN valid arguments to __init__.
    WHEN a concrete subclass of SemanticPerturbation is initialized.
    THEN the instance's attributes match what is expected
        and set_seed is called with the correct argument.
    """

    with patch("fmeval.transforms.semantic_perturbations.set_seed") as mock_set_seed:
        num_perturbations = 3
        seed = 42
        dummy = Dummy(["input"], ["output_1", "output_2", "output_3"], num_perturbations, seed, "asdf", kw_arg="qwerty")
        assert dummy.num_perturbations == num_perturbations
        mock_set_seed.assert_called_once_with(seed)


class TestCaseInitFailure(NamedTuple):
    input_keys: List[str]
    output_keys: List[str]
    err_msg: str


@pytest.mark.parametrize(
    "input_keys, output_keys, err_msg",
    [
        TestCaseInitFailure(
            input_keys=["input"],
            output_keys=["out_1", "out_2", "out_3"],
            err_msg="len(output_keys) is 3 while num_perturbations is 2. They should match.",
        ),
        TestCaseInitFailure(
            input_keys=["input_1", "input_2"],
            output_keys=["out_1", "out_2"],
            err_msg="Dummy takes a single input key.",
        ),
    ],
)
def test_semantic_perturbation_init_failure(input_keys, output_keys, err_msg):
    """
    GIVEN invalid inputs.
    WHEN a subclass of SemanticPerturbation is initialized.
    THEN an EvalAlgorithmClientError with the correct error message is raised.
    """
    with pytest.raises(EvalAlgorithmClientError, match=re.escape(err_msg)):
        Dummy(
            input_keys,
            output_keys,
            num_perturbations=2,
            seed=42,
            pos_arg="asdf",
            kw_arg="qwerty",
        )


def test_semantic_perturbation_call():
    """
    GIVEN a valid instance of a SemanticPerturbation subclass.
    WHEN its __call__ method is called.
    THEN the mapping between keys and perturbed outputs (generated via
        the `perturb` method) in the output record matches what is expected.
    """
    dummy = Dummy(
        ["input"],
        ["out_1", "out_2", "out_3"],
        num_perturbations=3,
        seed=42,
        pos_arg="asdf",
        kw_arg="qwerty",
    )
    sample = {"input": "Hi"}
    output = dummy(sample)
    assert output["out_1"] == "dummy_0"
    assert output["out_2"] == "dummy_1"
    assert output["out_3"] == "dummy_2"


class TestCaseButterFinger(NamedTuple):
    input_text: str
    expected_outputs: List[str]
    num_perturbations: int
    seed: int
    perturbation_prob: float


@pytest.mark.parametrize(
    "test_case",
    [
        TestCaseButterFinger(
            input_text="A quick brown fox jumps over the lazy dog 10 times.",
            expected_outputs=[
                "A quick bcmwn fox jumps over bhr lwzj dog 10 times.",
                "A auick brown fox jumps over the lazy dog 10 timef.",
                "A quick brown flz jujps ovef the lazy dog 10 times.",
            ],
            num_perturbations=3,
            seed=3,
            perturbation_prob=0.1,
        ),
        TestCaseButterFinger(
            input_text="A quick brown fox jumps over the lazy dog 10 times.",
            expected_outputs=[
                "A qujck brown fov jumps over dhe lavg dog 10 times.",
                "A qhick brocm fox jukps over tie pazy dog 10 times.",
            ],
            num_perturbations=2,
            seed=10,
            perturbation_prob=0.2,
        ),
    ],
)
def test_butter_finger_perturb(test_case):
    """
    GIVEN a valid ButterFinger instance.
    WHEN its `perturb` method is called.
    THEN the correct perturbed outputs are returned.
    """
    bf = ButterFinger(
        ["unused"],
        [f"bf_{i}" for i in range(len(test_case.expected_outputs))],
        perturbation_prob=test_case.perturbation_prob,
        num_perturbations=test_case.num_perturbations,
        seed=test_case.seed,
    )
    perturbed_outputs = bf.perturb(test_case.input_text)
    assert perturbed_outputs == test_case.expected_outputs


class TestCaseRandomUppercase(NamedTuple):
    input_text: str
    expected_outputs: List[str]
    num_perturbations: int
    seed: int
    uppercase_fraction: float


@pytest.mark.parametrize(
    "test_case",
    [
        TestCaseRandomUppercase(
            input_text="A quick brown fox jumps over the lazy dog 10 times.",
            expected_outputs=[
                "A quick bRowN fox jumps over the lazy dOG 10 timEs.",
                "A quicK brown fox jUmps Over tHe lazy dog 10 Times.",
                "A qUicK brown fox jumps over thE lazy Dog 10 timEs.",
            ],
            num_perturbations=3,
            seed=3,
            uppercase_fraction=0.1,
        ),
        TestCaseRandomUppercase(
            input_text="A quick brown fox jumps over the lazy dog 10 times.",
            expected_outputs=[
                "A qUick brown fox juMPs oveR thE lazy Dog 10 TimEs.",
                "A quick brown foX jUMps ovER the lazY dog 10 tiMes.",
            ],
            num_perturbations=2,
            seed=10,
            uppercase_fraction=0.2,
        ),
    ],
)
def test_random_uppercase_perturb(test_case):
    """
    GIVEN a valid RandomUppercase instance.
    WHEN its `perturb` method is called.
    THEN the correct perturbed outputs are returned.
    """
    ru = RandomUppercase(
        ["unused"],
        [f"ru_{i}" for i in range(len(test_case.expected_outputs))],
        num_perturbations=test_case.num_perturbations,
        seed=test_case.seed,
        uppercase_fraction=test_case.uppercase_fraction,
    )
    perturbed_outputs = ru.perturb(test_case.input_text)
    assert perturbed_outputs == test_case.expected_outputs


class TestCaseWhitespace(NamedTuple):
    input_text: str
    expected_outputs: List[str]
    num_perturbations: int
    seed: int
    add_prob: float
    remove_prob: float


@pytest.mark.parametrize(
    "test_case",
    [
        TestCaseWhitespace(
            input_text="A quick brown fox jumps over the lazy dog 10 times.",
            expected_outputs=[
                "A quick  brown fox jumps ov er the lazydog 10 times.",
                "A quick brown foxjumps o ve r the lazy  dog10 times.",
                "A quick brow n f oxjumps o ver the lazy do g 10 times.",
            ],
            num_perturbations=3,
            seed=3,
            add_prob=0.05,
            remove_prob=0.1,
        ),
        TestCaseWhitespace(
            input_text="A quick brown fox jumps over the lazy dog 10 times.",
            expected_outputs=[
                "A qu ickbr o wnfox  jumps  over t he  la zydog  10  times .",
                "A  q ui c k bro w nf ox  jump sov e r  th e l azy d og 10 ti mes. ",
            ],
            num_perturbations=2,
            seed=10,
            add_prob=0.4,
            remove_prob=0.2,
        ),
    ],
)
def test_whitespace_add_remove_perturb(test_case):
    """
    GIVEN a valid WhitespaceAddRemove instance.
    WHEN its `perturb` method is called.
    THEN the correct perturbed outputs are returned.
    """
    arw = AddRemoveWhitespace(
        ["unused"],
        [f"arw_{i}" for i in range(len(test_case.expected_outputs))],
        num_perturbations=test_case.num_perturbations,
        seed=test_case.seed,
        add_prob=test_case.add_prob,
        remove_prob=test_case.remove_prob,
    )
    perturbed_outputs = arw.perturb(test_case.input_text)
    assert perturbed_outputs == test_case.expected_outputs
