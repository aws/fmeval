from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Set, Optional
from ..exceptions import UserError
from data_sources import DataSource, S3DataFile, LocalDataFile
import os
import urllib
from s3fs import S3FileSystem

s3 = S3FileSystem()

@dataclass
class DataConfig:
    dataset_name: str
    dataset_uri: str
    model_input_jmespaths: List[str]
    model_output_jmespaths: List[str]
    target_output_jmespath: Optional[str]

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

@staticmethod
def get_data_source(dataset_uri: str) -> DataSource:
    """
    validate a dataset uri and return the corresponding DataResource object
    :param dataset_uri: local dataset path or s3 dataset uri
    :return: DataResource object
    """
    if _is_valid_local_path(dataset_uri):
        return _get_local_data_source(dataset_uri)
    elif _is_valid_s3_uri(dataset_uri):
        return _get_s3_data_source(dataset_uri)
    else:
        raise UserError(f"Invalid dataset path: {dataset_uri}")

    @staticmethod
def _get_local_data_source(dataset_uri) -> LocalDataFile:
    """
    :param dataset_uri: local dataset path
    :return: LocalDataFile object with dataset uri
    """
    absolute_local_path = os.path.abspath(urllib.parse.urlparse(dataset_uri).path)
    if os.path.isfile(absolute_local_path):
        return LocalDataFile(absolute_local_path)
    if os.path.isdir(absolute_local_path):
        raise UserError("Please provide a local file path instead of a directory path.")
    raise UserError(f"Invalid local path: {dataset_uri}")

@staticmethod
def _get_s3_data_source(dataset_uri) -> S3DataFile:
    """
    :param dataset_uri: s3 dataset uri
    :return: S3DataFile object with dataset uri
    """
    s3_info = s3.info(dataset_uri)
    if s3_info["type"] == "file":
        return S3DataFile(s3, dataset_uri)
    if s3_info["type"] == "directory":
        raise UserError("Please provide a s3 file path instead of a directory path.")
    raise UserError(f"Invalid s3 path: {dataset_uri}")

@staticmethod
def _is_valid_s3_uri(uri: str) -> bool:
    """
    :param uri: s3 file path
    :return: True if uri is a valid s3 path, False otherwise
    """
    parsed_url = urllib.parse.urlparse(uri)
    return parsed_url.scheme.lower() in ["s3", "s3n", "s3a"] and s3.exists(uri)

@staticmethod
def _is_valid_local_path(path: str) -> bool:
    """
    :param path: local file path
    :return: True if path is a valid local path, False otherwise
    """
    parsed_url = urllib.parse.urlparse(path)
    return parsed_url.scheme in ["", "file"] and os.path.exists(parsed_url.path)
