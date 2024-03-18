from abc import ABC, abstractmethod
from typing import Any, Dict, List

from fmeval.transforms.util import validate_key_uniqueness
from fmeval.util import assert_condition


class Transform(ABC):
    """A Transform represents a single operation that consumes a record and outputs another.

    Typically, the output record is the same object as the input; the Transform simply
    mutates its input (usually by augmenting it with new data). However, the output
    record can also be a new object, independent of the input record.

    The logic for creating the output record is implemented in the Transform's __call__ method,
    which takes a record as its sole argument. Any additional data besides this record
    that is required to perform the transformation logic should be stored as instance
    attributes in the Transform.
    """

    def __init__(self, *args, **kwargs):
        """Transform initializer.

        Concrete subclasses of Transform should always call super().__init__
        with every argument passed to their own __init__ method.
        Transform.__init__ stores all positional arguments in the `args` instance
        attribute and all keyword arguments in the `kwargs` instance attribute.
        This data is passed to Ray when Ray creates copies of this Transform instance
        to perform parallel execution.

        Note: The `input_keys` and `output_keys` attributes are initialized to None
        and only assigned a meaningful value if the `register_input_output_keys` method
        is called. This method is used in conjunction with the `validate_call` decorator
        to perform validations of the __call__ inputs and outputs at runtime.
        While it is not strictly necessary to utilize `register_input_output_keys` and
        `validate_call` when implementing your own transforms, these methods are used in
        all built-in transforms.

        :param *args: Variable length argument list.
        :param **kwargs: Arbitrary keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs
        self.input_keys = None
        self.output_keys = None

    @abstractmethod
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Return a record containing data that gets computed in this method.

        :param record: The input record to be transformed.
        :returns: A record containing data that gets computed in this method.
            This record can be the same object as the input record. In this case,
            the logic in this method should mutate the input record directly.
        """

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(input_keys={self.input_keys}, output_keys={self.output_keys}, "
            f"args={list(self.args)}, kwargs={self.kwargs})"
        )

    def register_input_output_keys(self, input_keys: List[str], output_keys: List[str], allow_duplicates: bool = False):
        """Assign self.input_keys and self.output_keys attributes.

        Concrete subclasses of Transform should call this method in their __init__
        if their __call__ method is decorated with `validate_call`.

        :param input_keys: The record keys corresponding to data that this Transform
            requires as inputs.
        :param output_keys: The keys introduced by this Transform's __call__ logic
            that will be present in the output record. If this Transform mutates its
            input, then these keys should be added by __call__ to the input record.
        :param allow_duplicates: Whether to allow duplicate values in `input_keys`.
        """
        assert_condition(isinstance(input_keys, List), "input_keys should be a list.")
        assert_condition(
            all(isinstance(input_key, str) for input_key in input_keys),
            "All keys in input_keys should be strings.",
        )
        if not allow_duplicates:
            validate_key_uniqueness(input_keys)
        assert_condition(isinstance(output_keys, List), "output_keys should be a list.")
        assert_condition(len(output_keys) > 0, "output_keys should be a non-empty list.")
        assert_condition(
            all(isinstance(output_key, str) for output_key in output_keys),
            "All keys in output_keys should be strings.",
        )
        validate_key_uniqueness(output_keys)
        self.input_keys = input_keys
        self.output_keys = output_keys
