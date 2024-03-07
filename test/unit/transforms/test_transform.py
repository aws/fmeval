import pytest
from unittest.mock import patch
from typing import List, Dict, Any, NamedTuple

from fmeval.exceptions import EvalAlgorithmInternalError
from fmeval.transforms.transform import Transform


class DummyTransform(Transform):
    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        pos_arg_a: List[int],
        pos_arg_b: Dict[str, Any],
        kw_arg_a: int = 42,
        kw_arg_b: str = "Hello",
    ):
        super().__init__(input_keys, output_keys, pos_arg_a, pos_arg_b, kw_arg_a=kw_arg_a, kw_arg_b=kw_arg_b)

    def __call__(self, record: Dict[str, Any]):
        return record


def test_transform_init_success():
    """
    GIVEN valid initializer arguments.
    WHEN a subclass of Transform is initialized.
    THEN the input_keys, output_keys, args, and kwargs attributes of the transform object
        match what is expected, and deep copies of the relevant data are made.
    """
    with patch("fmeval.transforms.transform.deepcopy") as mock_deepcopy:
        input_keys = ["input"]
        output_keys = ["output"]
        pos_arg_a = [162, 189]
        pos_arg_b = {"k1": ["v1"], "k2": ["v2"]}
        kw_arg_a = 123
        kw_arg_b = "Hi"

        mock_deepcopy.side_effect = [
            ["input_copy"],  # deepcopy called on input_keys
            ["output_copy"],  # deepcopy called on output_keys
            ([163, 190], {"k1_copy": ["v1_copy"], "k2_copy": ["v2_copy"]}),  # deepcopy called on (pos_arg_a, pos_arg_b)
            {124, "Hi_copy"},  # deepcopy called on {"kw_arg_a": kw_arg_a, "kw_arg_b": kw_arg_b}
        ]

        dummy = DummyTransform(input_keys, output_keys, pos_arg_a, pos_arg_b, kw_arg_a=kw_arg_a, kw_arg_b=kw_arg_b)

        assert dummy.input_keys == ["input_copy"]
        assert dummy.output_keys == ["output_copy"]
        assert dummy.args == (
            ["input_copy"],
            ["output_copy"],
            [163, 190],
            {"k1_copy": ["v1_copy"], "k2_copy": ["v2_copy"]},
        )
        assert dummy.kwargs == {124, "Hi_copy"}


class TestCaseTransformInitFailure(NamedTuple):
    input_keys: Any
    output_keys: Any
    err_msg: str


@pytest.mark.parametrize(
    "input_keys, output_keys, err_msg",
    [
        TestCaseTransformInitFailure(
            input_keys="input",
            output_keys=["output"],
            err_msg="input_keys should be a list.",
        ),
        TestCaseTransformInitFailure(
            input_keys=["input", 123],
            output_keys=["output"],
            err_msg="All keys in input_keys should be strings.",
        ),
        TestCaseTransformInitFailure(
            input_keys=["input_1", "input_2", "input_1", "input_1"],
            output_keys=["output"],
            err_msg="Duplicate keys found: ",
        ),
        TestCaseTransformInitFailure(
            input_keys=["input"],
            output_keys="output",
            err_msg="output_keys should be a list.",
        ),
        TestCaseTransformInitFailure(
            input_keys=["input"],
            output_keys=[],
            err_msg="output_keys should be a non-empty list.",
        ),
        TestCaseTransformInitFailure(
            input_keys=["input"],
            output_keys=["output", 123],
            err_msg="All keys in output_keys should be strings.",
        ),
        TestCaseTransformInitFailure(
            input_keys=["input_1", "input_2"],
            output_keys=["output_1", "output_2", "output_1", "output_1"],
            err_msg="Duplicate keys found: ",
        ),
    ],
)
def test_transform_init_failure(input_keys, output_keys, err_msg):
    """
    GIVEN invalid initializer arguments.
    WHEN a Transform is initialized.
    THEN an EvalAlgorithmInternalError is raised.
    """
    with pytest.raises(EvalAlgorithmInternalError, match=err_msg):
        DummyTransform(input_keys, output_keys, [123], {"k": "v"})


def test_repr():
    input_keys = ["input"]
    output_keys = ["output"]
    pos_arg_a = [162, 189]
    pos_arg_b = {"k1": ["v1"], "k2": ["v2"]}
    kw_arg_a = 123
    kw_arg_b = "Hi"
    dummy = DummyTransform(input_keys, output_keys, pos_arg_a, pos_arg_b, kw_arg_a=kw_arg_a, kw_arg_b=kw_arg_b)
    expected = (
        "DummyTransform(input_keys=['input'], output_keys=['output'], "
        "args=[[162, 189], {'k1': ['v1'], 'k2': ['v2']}], "
        "kwargs={'kw_arg_a': 123, 'kw_arg_b': 'Hi'})"
    )
    assert str(dummy) == expected
