import logging
import os
import ray.data
import urllib.parse
from typing import Type
from s3fs import S3FileSystem
from amazon_fmeval.constants import MIME_TYPE_JSON, MIME_TYPE_JSONLINES
from amazon_fmeval.data_loaders.data_sources import DataSource, LocalDataFile, S3DataFile, DataFile
from amazon_fmeval.data_loaders.json_data_loader import JsonDataLoaderConfig, JsonDataLoader
from amazon_fmeval.data_loaders.json_parser import JsonParser
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.exceptions import EvalAlgorithmClientError, EvalAlgorithmInternalError
from amazon_fmeval.perf_util import timed_block

s3 = S3FileSystem()
logger = logging.getLogger(__name__)


def get_dataset(config: DataConfig) -> ray.data.Dataset:
    """
    Util method to load Ray datasets using an input DataConfig.

    :param config: Input DataConfig
    """
    with timed_block(f"Loading dataset {config.dataset_name}", logger):
        data_source = get_data_source(config.dataset_uri)
        data_loader_config = _get_data_loader_config(data_source, config)
        data_loader = _get_data_loader(config.dataset_mime_type)
        data = data_loader.load_dataset(data_loader_config)
    return data


def _get_data_loader_config(data_source: DataSource, config: DataConfig) -> JsonDataLoaderConfig:
    """
    Returns a dataloader config based on the dataset MIME type specified in `config`.

    :param data_source: The dataset's DataSource object.
    :param config: Configures the returned dataloader config.
    :returns: A dataloader config object, created from `data_source` and `config`.
    """
    if config.dataset_mime_type == MIME_TYPE_JSON:
        if not isinstance(data_source, DataFile):
            raise EvalAlgorithmInternalError(
                f"JSON datasets must be stored in a single file. " f"Provided dataset has type {type(data_source)}."
            )
        return JsonDataLoaderConfig(
            parser=JsonParser(config),
            data_file=data_source,
            dataset_mime_type=MIME_TYPE_JSON,
            dataset_name=config.dataset_name,
        )
    elif config.dataset_mime_type == MIME_TYPE_JSONLINES:
        if not isinstance(data_source, DataFile):
            raise EvalAlgorithmInternalError(
                f"JSONLines datasets must be stored in a single file. "
                f"Provided dataset has type {type(data_source)}."
            )
        return JsonDataLoaderConfig(
            parser=JsonParser(config),
            data_file=data_source,
            dataset_mime_type=MIME_TYPE_JSONLINES,
            dataset_name=config.dataset_name,
        )
    else:  # pragma: no cover
        raise EvalAlgorithmInternalError(
            "Dataset MIME types other than JSON and JSON Lines are not supported. "
            f"MIME type detected from config is {config.dataset_mime_type}."
        )


def _get_data_loader(dataset_mime_type: str) -> Type[JsonDataLoader]:
    """
    Returns the dataloader class corresponding to the given dataset MIME type.

    :param dataset_mime_type: Determines which dataloader class to return.
    :returns: A dataloader class.
    """
    if dataset_mime_type == MIME_TYPE_JSON:
        return JsonDataLoader
    elif dataset_mime_type == MIME_TYPE_JSONLINES:
        return JsonDataLoader
    else:  # pragma: no cover
        raise EvalAlgorithmInternalError(
            "Dataset MIME types other than JSON and JSON Lines are not supported. "
            f"MIME type detected from config is {dataset_mime_type}."
        )


def get_data_source(dataset_uri: str) -> DataSource:
    """
    Validates a dataset URI and returns the corresponding DataSource object
    :param dataset_uri: local dataset path or s3 dataset uri
    :return: DataSource object
    """
    if _is_valid_local_path(dataset_uri):
        return _get_local_data_source(dataset_uri)
    elif _is_valid_s3_uri(dataset_uri):
        return _get_s3_data_source(dataset_uri)
    else:
        raise EvalAlgorithmClientError(f"Invalid dataset path: {dataset_uri}")


def _get_local_data_source(dataset_uri) -> LocalDataFile:
    """
    :param dataset_uri: local dataset path
    :return: LocalDataFile object with dataset uri
    """
    absolute_local_path = os.path.abspath(urllib.parse.urlparse(dataset_uri).path)
    if os.path.isfile(absolute_local_path):
        return LocalDataFile(absolute_local_path)
    if os.path.isdir(absolute_local_path):
        # TODO: extend support to directories
        raise EvalAlgorithmClientError("Please provide a local file path instead of a directory path.")
    raise EvalAlgorithmClientError(f"Invalid local path: {dataset_uri}")


def _get_s3_data_source(dataset_uri) -> S3DataFile:
    """
    :param dataset_uri: s3 dataset uri
    :return: S3DataFile object with dataset uri
    """
    s3_info = s3.info(dataset_uri)
    if s3_info["type"] == "file":
        return S3DataFile(s3, dataset_uri)
    if s3_info["type"] == "directory":
        # TODO: extend support to directories
        raise EvalAlgorithmClientError("Please provide a s3 file path instead of a directory path.")
    raise EvalAlgorithmClientError(f"Invalid s3 path: {dataset_uri}")


def _is_valid_s3_uri(uri: str) -> bool:
    """
    :param uri: s3 file path
    :return: True if uri is a valid s3 path, False otherwise
    """
    parsed_url = urllib.parse.urlparse(uri)
    return parsed_url.scheme.lower() in ["s3", "s3n", "s3a"] and s3.exists(uri)


def _is_valid_local_path(path: str) -> bool:
    """
    :param path: local file path
    :return: True if path is a valid local path, False otherwise
    """
    parsed_url = urllib.parse.urlparse(path)
    return parsed_url.scheme in ["", "file"] and os.path.exists(parsed_url.path)
