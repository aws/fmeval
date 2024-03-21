import pytest
from typing import List, Dict, Any, NamedTuple

from fmeval.exceptions import EvalAlgorithmInternalError
from fmeval.transforms.transform import Transform


class DummyTransform(Transform):
    def __init__(
        self,
        pos_arg_a: List[int],
        pos_arg_b: Dict[str, Any],
        kw_arg_a: int = 42,
        kw_arg_b: str = "Hello",
    ):
        super().__init__(pos_arg_a, pos_arg_b, kw_arg_a=kw_arg_a, kw_arg_b=kw_arg_b)

    def __call__(self, record: Dict[str, Any]):
        return record


def test_transform_init_success():
    """
    GIVEN valid initializer arguments.
    WHEN a subclass of Transform is initialized.
    THEN the input_keys, output_keys, args, and kwargs attributes
        of the transform object match expected values.
    """
    pos_arg_a = [162, 189]
    pos_arg_b = {"k1": ["v1"], "k2": ["v2"]}
    kw_arg_a = 123
    kw_arg_b = "Hi"

    dummy = DummyTransform(pos_arg_a, pos_arg_b, kw_arg_a=kw_arg_a, kw_arg_b=kw_arg_b)

    assert dummy.input_keys is None
    assert dummy.output_keys is None
    assert dummy.args == (
        pos_arg_a,
        pos_arg_b,
    )
    assert dummy.kwargs == {"kw_arg_a": kw_arg_a, "kw_arg_b": kw_arg_b}


class TestCaseRegisterKeysFailure(NamedTuple):
    input_keys: Any
    output_keys: Any
    err_msg: str


@pytest.mark.parametrize(
    "input_keys, output_keys, err_msg",
    [
        TestCaseRegisterKeysFailure(
            input_keys="input",
            output_keys=["output"],
            err_msg="input_keys should be a list.",
        ),
        TestCaseRegisterKeysFailure(
            input_keys=["input", 123],
            output_keys=["output"],
            err_msg="All keys in input_keys should be strings.",
        ),
        TestCaseRegisterKeysFailure(
            input_keys=["input_1", "input_2", "input_1", "input_1"],
            output_keys=["output"],
            err_msg="Duplicate keys found: ",
        ),
        TestCaseRegisterKeysFailure(
            input_keys=["input"],
            output_keys="output",
            err_msg="output_keys should be a list.",
        ),
        TestCaseRegisterKeysFailure(
            input_keys=["input"],
            output_keys=[],
            err_msg="output_keys should be a non-empty list.",
        ),
        TestCaseRegisterKeysFailure(
            input_keys=["input"],
            output_keys=["output", 123],
            err_msg="All keys in output_keys should be strings.",
        ),
        TestCaseRegisterKeysFailure(
            input_keys=["input_1", "input_2"],
            output_keys=["output_1", "output_2", "output_1", "output_1"],
            err_msg="Duplicate keys found: ",
        ),
    ],
)
def test_register_input_output_keys_failure(input_keys, output_keys, err_msg):
    """
    GIVEN invalid arguments.
    WHEN `register_input_output_keys` is called.
    THEN an EvalAlgorithmInternalError is raised.
    """
    with pytest.raises(EvalAlgorithmInternalError, match=err_msg):
        d = DummyTransform([123], {"k": "v"})
        d.register_input_output_keys(input_keys, output_keys)


def test_register_input_output_keys_duplicate_keys_allowed():
    """
    GIVEN a list of input keys with duplicate values.
    WHEN register_input_output_keys is called with `allow_duplicates` = True.
    THEN no exceptions are raised due to duplicate keys being found.
    """
    d = DummyTransform([123], {"k": "v"})
    d.register_input_output_keys(["a", "a"], ["b"], allow_duplicates=True)


def test_repr():
    """
    GIVEN a valid Transform instance.
    WHEN its `__repr__` method is called.
    THEN the correct string is returned.
    """
    input_keys = ["input"]
    output_keys = ["output"]
    pos_arg_a = [162, 189]
    pos_arg_b = {"k1": ["v1"], "k2": ["v2"]}
    kw_arg_a = 123
    kw_arg_b = "Hi"
    dummy = DummyTransform(pos_arg_a, pos_arg_b, kw_arg_a=kw_arg_a, kw_arg_b=kw_arg_b)
    expected = (
        "DummyTransform(input_keys=None, output_keys=None, "
        "args=[[162, 189], {'k1': ['v1'], 'k2': ['v2']}], "
        "kwargs={'kw_arg_a': 123, 'kw_arg_b': 'Hi'})"
    )
    assert str(dummy) == expected

    dummy.register_input_output_keys(input_keys, output_keys)
    expected = (
        "DummyTransform(input_keys=['input'], output_keys=['output'], "
        "args=[[162, 189], {'k1': ['v1'], 'k2': ['v2']}], "
        "kwargs={'kw_arg_a': 123, 'kw_arg_b': 'Hi'})"
    )
    assert str(dummy) == expected
