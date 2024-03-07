import re
import pytest
from typing import NamedTuple, Set
from fmeval.transforms.util import validate_added_keys, validate_existing_keys, validate_key_uniqueness
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
