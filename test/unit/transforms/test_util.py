import re
from unittest.mock import Mock

import pytest
from typing import NamedTuple, Set, Dict, Any, Optional, List
from fmeval.transforms.util import validate_added_keys, validate_existing_keys, validate_key_uniqueness, validate_call
from fmeval.util import EvalAlgorithmInternalError


def test_validate_key_uniqueness_success():
    """
    GIVEN a list of unique keys.
    WHEN validate_key_uniqueness is called.
    THEN no error is raised.
    """
    keys = ["a", "b", "c"]
    validate_key_uniqueness(keys)


def test_validate_key_uniqueness_failure():
    """
    GIVEN a list of non-unique keys.
    WHEN validate_key_uniqueness is called.
    THEN an EvalAlgorithmInternalError with the correct message is raised.
    """
    keys = ["a", "b", "c", "c", "b", "b"]
    duplicates = ["c", "b", "b"]
    with pytest.raises(EvalAlgorithmInternalError, match=re.escape(f"Duplicate keys found: {duplicates}.")):
        validate_key_uniqueness(keys)


def test_validate_existing_keys_success():
    """
    GIVEN a record containing all expected keys.
    WHEN validate_existing_keys is called.
    THEN no exception is raised.
    """
    record = {"a": 1, "b": 2, "c": 3, "d": 4}
    keys = ["a", "b", "c"]
    validate_existing_keys(record, keys)


def test_validate_existing_keys_failure():
    """
    GIVEN a record that is missing an expected key.
    WHEN validate_existing_keys is called.
    THEN an EvalAlgorithmInternalError with the correct message is raised.
    """
    record = {"a": 1, "b": 2, "e": 4}
    keys = ["a", "b", "c", "d"]
    missing_keys = ["c", "d"]
    with pytest.raises(
        EvalAlgorithmInternalError,
        match=re.escape(
            f"Record {record} is expected to contain the following keys, " f"but they are missing: {missing_keys}."
        ),
    ):
        validate_existing_keys(record, keys)


def test_validate_added_keys_success():
    """
    GIVEN arguments that should not raise an exception.
    WHEN validate_added_keys is called.
    THEN no exception is raised.
    """
    current_keys = {"a", "b", "c", "d"}
    original_keys = {"a", "c"}
    keys_to_add = {"b", "d"}
    validate_added_keys(current_keys, original_keys, keys_to_add)


class TestCaseValidateAddedKeys(NamedTuple):
    current_keys: Set[str]
    original_keys: Set[str]
    keys_to_add: Set[str]


@pytest.mark.parametrize(
    "current_keys, original_keys, keys_to_add",
    [
        TestCaseValidateAddedKeys(
            current_keys={"a", "b", "c", "d"},
            original_keys={"a", "c"},
            keys_to_add={"b"},
        ),
        TestCaseValidateAddedKeys(
            current_keys={"a", "b", "c"},
            original_keys={"a", "c"},
            keys_to_add={"b", "d"},
        ),
    ],
)
def test_validate_added_keys_failure(current_keys, original_keys, keys_to_add):
    """
    GIVEN arguments that should raise an exception.
    WHEN validate_added_keys is called.
    THEN an EvalAlgorithmInternalError is raised.
    """
    with pytest.raises(EvalAlgorithmInternalError, match="The set difference"):
        validate_added_keys(current_keys, original_keys, keys_to_add)


def test_validate_call_success():
    """
    GIVEN a function with the signature (self, record: Dict[str, Any]) -> Dict[str, Any]
        (i.e. the __call__ method of some Transform).
    WHEN validate_call is called with this function as its argument, and the
        resulting wrapper function is called with valid arguments.
    THEN no exceptions are raised, and the output of the wrapper function matches
        what is expected (i.e. the output of the original __call__ method).
    """

    def _call(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record["new_key"] = 17
        return record

    validated_call_method = validate_call(_call)
    _self = Mock()
    _self.input_keys = ["input_key"]
    _self.output_keys = ["new_key"]

    original_output = _call(_self, {"input_key": "input"})
    wrapper_output = validated_call_method(_self, {"input_key": "input"})
    assert wrapper_output == original_output == {"input_key": "input", "new_key": 17}


class TestCaseValidateCall(NamedTuple):
    input_keys: Optional[List[str]]
    output_keys: Optional[List[str]]
    err_msg: str


@pytest.mark.parametrize(
    "input_keys, output_keys, err_msg",
    [
        TestCaseValidateCall(
            input_keys=None,
            output_keys=["new_key"],
            err_msg="self.input_keys has not been set. You should set this attribute using "
            "the register_input_output_keys method.",
        ),
        TestCaseValidateCall(
            input_keys=["input_key"],
            output_keys=None,
            err_msg="self.output_keys has not been set. You should set this attribute using "
            "the register_input_output_keys method.",
        ),
    ],
)
def test_validate_call_failure(input_keys, output_keys, err_msg):
    """
    GIVEN a function with the signature (self, record: Dict[str, Any]) -> Dict[str, Any]
        (i.e. the __call__ method of some Transform).
    WHEN validate_call is called with this function as its argument, and the
        resulting wrapper function is called with invalid arguments.
    THEN an exception with the correct error message is raised.
    """

    def _call(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record["new_key"] = 17
        return record

    validated_call_method = validate_call(_call)
    _self = Mock()
    _self.input_keys = input_keys
    _self.output_keys = output_keys

    with pytest.raises(EvalAlgorithmInternalError, match=err_msg):
        validated_call_method(_self, {"input_key": "input"})
