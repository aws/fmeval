from copy import deepcopy
from typing import Any, List, Set, Optional


def get_nested_list_levels(data: Any) -> int:
    """
    Get the nested level of a nested list. For example, [[0.1]] => 2, [[[0.1]]] => 3.
    By default, level is 0 and empty list is not counted as a level.

    :param data: nested list
    :return: The list's nested level
    """
    level = 0
    next_level_list = deepcopy(data)
    while isinstance(next_level_list, list) and len(next_level_list):
        level += 1
        next_level_list = next_level_list[0]

    return level


def has_only_valid_types_in_nested_list(two_level_list: List, valid_types: Set) -> bool:
    """
    Check if a two-level nested list contains only valid types.
    e.g. Check if [[0.8, 0.9]] contains only floats.

    :param two_level_list: two-level list
    :param valid_types: valid value types
    :return: whether the nested list contains only valid types
    """
    return all([has_only_valid_types(l, valid_types) for l in two_level_list])


def validate_list_with_same_size_item(l: List[List[Any]], expected_size: Optional[int] = None) -> bool:
    """
    Validate if all the items in the list are of the same size.

    :param l: two-level nested list to be validated
    :param expected_size: Optional, the expected size of the item in the list
    :return: whether all the items in the list have the same size
    """
    if expected_size:
        return all([len(i) == expected_size for i in l])
    else:
        return len(set([len(i) for i in l])) == 1


def is_nested_list(data: Any) -> bool:
    """
    Check if the data is a two-level nested list
    :param data: data to be checked
    :return: whether data is a two-level nested list
    """
    return isinstance(data, list) and len(data) > 0 and all(isinstance(sub_lst, list) for sub_lst in data)


def has_only_valid_types(l: List, valid_types: Set) -> bool:
    """
    Check if a list contains only valid types.

    :param l: a list
    :param valid_types: valid value types
    :return: whether the list contains only valid types
    """
    existing_types = set(type(i) for i in l)
    invalid_types = existing_types.difference(valid_types)
    return not invalid_types
