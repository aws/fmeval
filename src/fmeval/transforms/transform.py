from abc import ABC, abstractmethod
from copy import deepcopy
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

    def __init__(self, input_keys: List[str], output_keys: List[str], *args, **kwargs):
        """Transform initializer.

        Note: concrete subclasses of Transform should always call super().__init__
        with every argument passed to their own __init__ method. This is because
        this method will store a copy of all positional arguments in the `args`
        instance attribute and all keyword arguments in the `kwargs` instance attribute,
        so that Ray can create copies of this Transform instance when performing
        parallel execution.

        :param input_keys: The record keys corresponding to data that this Transform
            requires as inputs.
        :param output_keys: The keys introduced by this Transform's __call__ logic
            that will be present in the output record. If this Transform mutates its
            input, then these keys should be added to the input record.
            It is the responsibility of the implementer of a Transform to validate
            that their __call__ method adds only these keys, and no others.
            See fmeval.transforms.util.validate_added_keys and some built-in
            Transforms for examples on how to perform such validations.
        :param *args: Variable length argument list.
        :param **kwargs: Arbitrary keyword arguments.
        """
        assert_condition(
            isinstance(input_keys, List), "The input_keys argument for Transform.__init__ should be a list."
        )
        assert_condition(
            all(isinstance(input_key, str) for input_key in input_keys),
            "All keys in the input_keys argument for Transform.__init__ should be strings.",
        )
        validate_key_uniqueness(input_keys)
        assert_condition(
            isinstance(output_keys, List), "The output_keys argument for Transform.__init__ should be a list."
        )
        assert_condition(
            all(isinstance(output_key, str) for output_key in output_keys),
            "All keys in the output_keys argument for Transform.__init__ should be strings.",
        )
        validate_key_uniqueness(output_keys)
        self.input_keys = deepcopy(input_keys)
        self.output_keys = deepcopy(output_keys)
        self.args = (self.input_keys, self.output_keys) + deepcopy(args)
        self.kwargs = deepcopy(kwargs)

    @abstractmethod
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Return a record containing data that gets computed in this method.

        :param record: The input record to be transformed.
        :returns: A record containing data that gets computed in this method.
            This record can be the same object as the input record. In this case,
            the logic in this method should mutate the input record directly.
        """
