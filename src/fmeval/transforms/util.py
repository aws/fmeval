from typing import Any, Callable, Dict, List, Tuple
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


def validate_call(call_method: Callable) -> Callable:
    """Decorator for the __call__ method of Transforms used for validating input and output.

    This decorator validates that all keys in a Transform's `input_keys` attribute are
    present in the input record that is passed to `__call__` and that the keys that
    are added to the record by the Transform's internal `__call__` logic are limited
    to the keys specified by the Transform's `output_keys` attribute.

    Note that this decorator should only be used by Transforms that mutate their input record,
    as the output key validation may not make sense in the case where a new record object
    (which may not keep all the same keys as the original record) is returned as the output.

    Additionally, this decorator should be used in conjunction with the
    `register_input_output_keys` method, as the `input_keys` and `output_keys` are initialized
    to None in `Transform.__init__`.

    :param call_method: The `__call__` method of a Transform.
    :returns: A wrapper function that performs pre- and post-validation on top of `__call__`.
    """

    def wrapper(self, record: Dict[str, Any]) -> Dict[str, Any]:
        assert_condition(
            self.input_keys is not None,
            "self.input_keys has not been set. You should set this attribute using "
            "the register_input_output_keys method.",
        )
        assert_condition(
            self.output_keys is not None,
            "self.output_keys has not been set. You should set this attribute using "
            "the register_input_output_keys method.",
        )
        validate_existing_keys(record, self.input_keys)
        call_output = call_method(self, record)
        validate_existing_keys(call_output, self.output_keys)
        return call_output

    return wrapper


def create_output_key(transform_name: str, *args, **kwargs) -> str:
    """Create an output key to be used by a Transform instance.

    This method is used to create unique, easily-identifiable output keys
    for Transform instances. *args and **kwargs are used purely for
    ensuring key uniqueness, and need not be arguments to the Transform's
    initializer, though they generally will be, for ease of interpretability.

    :param transform_name: The name of the Transform class.
        This argument is generally passed via the __name__ attribute of
        a class. Note that we do not simply pass the class itself (which
        would be the more intuitive approach), as Ray wraps actor classes
        in its own wrapper class, which will cause the __name__ attribute
        to return an unexpected value.
    :param *args: Variable length argument list.
    :param **kwargs: Arbitrary keyword arguments.
    """

    def args_to_str(positional_args: Tuple[str]) -> str:
        return ", ".join(str(arg) for arg in positional_args)

    def kwargs_to_str(keyword_args: Dict[str, Any]) -> str:
        return ", ".join(f"{k}={str(v)}" for k, v in keyword_args.items())

    args_string = args_to_str(args)
    kwargs_string = kwargs_to_str(kwargs)
    output_key = (
        f"{transform_name}"
        f"({args_string if args_string else ''}"
        f"{', ' if args_string and kwargs_string else ''}"
        f"{kwargs_string if kwargs_string else ''})"
    )
    return output_key
