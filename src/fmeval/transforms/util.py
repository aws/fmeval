from typing import Any, Dict, List, Set
from fmeval.util import assert_condition


def validate_key_uniqueness(keys: List[str]) -> None:
    """Validate that a list of keys contains unique values.

    This function exists to capture the full list of duplicate keys
    in the error message that is raised, for a better debugging experience.

    :param keys: The keys to be validated.
    :raises: EvalAlgorithmInternalError if the values in `keys` are not unique.
    """
    seen = set()
    duplicates = []
    for key in keys:
        if key in seen:
            duplicates.append(key)
        else:
            seen.add(key)
    assert_condition(len(duplicates) == 0, f"Duplicate keys found: {duplicates}.")


def validate_existing_keys(record: Dict[str, Any], keys: List[str]) -> None:
    """Validate that all expected keys are present in a record.

    :param record: The record to be validated.
    :param keys: The keys that are expected to be present in the record.
    :raises: EvalAlgorithmInternalError if any validation fails.
    """
    missing_keys = []
    for key in keys:
        if key not in record:
            missing_keys.append(key)
    assert_condition(
        len(missing_keys) == 0,
        f"Record {record} is expected to contain the following keys, " f"but they are missing: {missing_keys}.",
    )


def validate_added_keys(current_keys: Set[str], original_keys: Set[str], keys_to_add: Set[str]) -> None:
    """Validate that the set difference between `current_keys` and `original_keys` is `keys_to_add`.

    Note: this function should only be used when by Transforms that mutate their input record.
    It doesn't make sense to call this function if the Transform constructs a new record to
    be used as output, since said output record need not contain all the keys in the input record.

    :param current_keys: The keys that are currently present in a record.
    :param original_keys: The keys that were originally present in a record
        (prior to performing transform-specific logic, which adds new keys).
    :param keys_to_add: The keys that a transform should have added to its input record.
        When this function is called from within a Transform's __call__ method (which should
        be the primary use case for this function), this parameter should be the transform's
        `output_keys` attribute.
    :raises: EvalAlgorithmInternalError if any validation fails.
    """
    assert_condition(
        current_keys - original_keys == keys_to_add,
        f"The set difference between the current keys: {current_keys} "
        f"and the original keys: {original_keys} does not match "
        f"the expected keys to be added: {keys_to_add}.",
    )
