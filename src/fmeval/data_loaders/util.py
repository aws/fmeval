import logging
import os
import boto3
import botocore.errorfactory
import ray.data
import urllib.parse

from typing import Type, Optional
from fmeval import util
from fmeval.constants import (
    MIME_TYPE_JSON,
    MIME_TYPE_JSONLINES,
    PARTITION_MULTIPLIER,
    SEED,
    MAX_ROWS_TO_TAKE,
)
from fmeval.data_loaders.data_sources import DataSource, LocalDataFile, S3DataFile, DataFile, S3Uri, get_s3_client
from fmeval.data_loaders.json_data_loader import JsonDataLoaderConfig, JsonDataLoader
from fmeval.data_loaders.json_parser import JsonParser
from fmeval.data_loaders.data_config import DataConfig
from fmeval.util import get_num_actors
from fmeval.exceptions import EvalAlgorithmClientError, EvalAlgorithmInternalError
from fmeval.perf_util import timed_block

client = boto3.client("s3")
logger = logging.getLogger(__name__)


def get_dataset(config: DataConfig, num_records: Optional[int] = None) -> ray.data.Dataset:
    """
    Util method to load Ray datasets using an input DataConfig.

    :param config: Input DataConfig
    :param num_records: the number of records to sample from the dataset
    """
    # The following setup is necessary to instruct Ray to preserve the
    # order of records in the datasets
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = True
    with timed_block(f"Loading dataset {config.dataset_name}", logger):
        data_source = get_data_source(config.dataset_uri)
        data_loader_config = _get_data_loader_config(data_source, config)
        data_loader = _get_data_loader(config.dataset_mime_type)
        data = data_loader.load_dataset(data_loader_config)
        count = data.count()
        util.require(count > 0, "Data has to have at least one record")
        if num_records and num_records > 0:  # pragma: no branch
            # TODO update sampling logic - current logic is biased towards first MAX_ROWS_TO_TAKE rows
            num_records = min(num_records, count)
            # We are using to_pandas, sampling with Pandas dataframe, and then converting back to Ray Dataset to use
            # Pandas DataFrame's ability to sample deterministically. This is temporary workaround till Ray solves this
            # issue: https://github.com/ray-project/ray/issues/40406
            if count > MAX_ROWS_TO_TAKE:
                # If count is larger than 100000, we take the first 100000 row, and then sample from that to
                # maintain deterministic behaviour. We are using take_batch to get a pandas dataframe of size
                # MAX_ROWS_TO_TAKE when the size of original dataset is greater than MAX_ROWS_TO_TAKE. This is to avoid
                # failures in driver node by pulling too much data.
                pandas_df = data.take_batch(batch_size=MAX_ROWS_TO_TAKE, batch_format="pandas")
            else:
                pandas_df = data.to_pandas()
            sampled_df = pandas_df.sample(num_records, random_state=SEED)
            data = ray.data.from_pandas(sampled_df)
        data = data.repartition(get_num_actors() * PARTITION_MULTIPLIER).materialize()
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
    s3_client = get_s3_client(dataset_uri)
    s3_uri = S3Uri(dataset_uri)
    s3_obj = s3_client.get_object(Bucket=s3_uri.bucket, Key=s3_uri.key)
    if "application/x-directory" in s3_obj["ContentType"]:
        # TODO: extend support to directories
        raise EvalAlgorithmClientError("Please provide a s3 file path instead of a directory path.")
    else:
        # There isn't a good way to check if s3_obj corresponds specifically to a file,
        # so we treat anything that is not a directory as a file.
        return S3DataFile(dataset_uri)


def _is_valid_s3_uri(uri: str) -> bool:
    """
    :param uri: s3 file path
    :return: True if uri is a valid s3 path, False otherwise
    """
    parsed_url = urllib.parse.urlparse(uri)
    if parsed_url.scheme.lower() not in ["s3", "s3n", "s3a"]:
        return False
    try:
        s3_client = get_s3_client(uri)
        s3_uri = S3Uri(uri)
        s3_client.get_object(Bucket=s3_uri.bucket, Key=s3_uri.key)
        return True
    except botocore.errorfactory.ClientError:
        return False


def _is_valid_local_path(path: str) -> bool:
    """
    :param path: local file path
    :return: True if path is a valid local path, False otherwise
    """
    parsed_url = urllib.parse.urlparse(path)
    return parsed_url.scheme in ["", "file"] and os.path.exists(parsed_url.path)
