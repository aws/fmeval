import random

import botocore.errorfactory
import pytest
import ray.data

from typing import NamedTuple
from unittest.mock import call, Mock, patch

from fmeval.constants import MIME_TYPE_JSON, MIME_TYPE_JSONLINES
from fmeval.data_loaders.data_sources import LocalDataFile, S3DataFile, DataSource
from fmeval.data_loaders.json_data_loader import JsonDataLoaderConfig, JsonDataLoader
from fmeval.data_loaders.util import (
    _get_data_loader,
    _get_data_loader_config,
    _is_valid_s3_uri,
    _is_valid_local_path,
    get_data_source,
    get_dataset,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.exceptions import EvalAlgorithmClientError, EvalAlgorithmInternalError

S3_PREFIX = "s3://"
LOCAL_PREFIX = "file://"

DATASET_URI = "dataset.json"
INVALID_DATASET_URI = "invalid_dataset"
DIRECTORY_URI = "dir1/dir2/"


class TestDataLoaderUtil:
    @patch("fmeval.data_loaders.util._get_data_loader", return_value=Mock())
    @patch("fmeval.data_loaders.util._get_data_loader_config", return_value=Mock())
    @patch("fmeval.data_loaders.util.get_data_source", return_value=Mock())
    def test_get_dataset(self, mock_get_data_source, mock_get_data_loader_config, mock_get_data_loader):
        """
        GIVEN a DataConfig
        WHEN get_dataset is called
        THEN all of get_dataset's helper methods get called with expected arguments
        """
        config = Mock(spec=DataConfig)
        config.dataset_name = "dataset1"
        config.dataset_uri = "unused"
        config.dataset_mime_type = "unused"
        items = [{"score": random.randint(0, 100)} for _ in range(2000)]
        mock_get_data_loader.return_value.load_dataset.return_value = ray.data.from_items(items)
        data = get_dataset(config, 100)
        mock_get_data_loader.assert_called_once_with(config.dataset_mime_type)
        assert data.count() == 100
        second_data = get_dataset(config, 100)
        assert data.take_all() == second_data.take_all()
        calls = [call(config.dataset_uri), call(config.dataset_uri)]
        mock_get_data_source.assert_has_calls(calls)
        calls = [call(mock_get_data_source.return_value, config), call(mock_get_data_source.return_value, config)]
        mock_get_data_loader_config.assert_has_calls(calls)

    @patch("fmeval.data_loaders.util._get_data_loader", return_value=Mock())
    @patch("fmeval.data_loaders.util._get_data_loader_config", return_value=Mock())
    @patch("fmeval.data_loaders.util.get_data_source", return_value=Mock())
    def test_get_dataset_with_negative_num_records(
        self, mock_get_data_source, mock_get_data_loader_config, mock_get_data_loader
    ):
        """
        GIVEN a DataConfig
        WHEN get_dataset is called
        THEN all of get_dataset's helper methods get called with expected arguments
        """
        config = Mock(spec=DataConfig)
        config.dataset_name = "dataset1"
        config.dataset_uri = "unused"
        config.dataset_mime_type = "unused"

        mock_get_data_loader.return_value.load_dataset.return_value = ray.data.from_items([{"score": 3}] * 2000)
        data = get_dataset(config, -1)
        mock_get_data_loader.assert_called_once_with(config.dataset_mime_type)
        assert data.count() == 2000
        mock_get_data_source.assert_called_once_with(config.dataset_uri)
        mock_get_data_loader_config.assert_called_once_with(mock_get_data_source.return_value, config)

    @patch("fmeval.data_loaders.util._get_data_loader", return_value=Mock())
    @patch("fmeval.data_loaders.util._get_data_loader_config", return_value=Mock())
    @patch("fmeval.data_loaders.util.get_data_source", return_value=Mock())
    def test_get_dataset_sampling_determinism_in_large_datasets(
        self, mock_get_data_source, mock_get_data_loader_config, mock_get_data_loader
    ):
        data = ray.data.from_items([i for i in range(200000)])
        config = Mock(spec=DataConfig)
        config.dataset_name = "dataset1"
        config.dataset_uri = "unused"
        config.dataset_mime_type = "unused"
        mock_get_data_loader.return_value.load_dataset.return_value = data
        data = get_dataset(config, 20)
        assert data.take_all() == get_dataset(config, 20).take_all()

    class TestCaseGetDataLoaderConfigError(NamedTuple):
        dataset_mime_type: str
        error_message: str

    @pytest.mark.parametrize(
        "dataset_mime_type, error_message",
        [
            TestCaseGetDataLoaderConfigError(
                dataset_mime_type=MIME_TYPE_JSON,
                error_message="JSON datasets must be stored in a single file",
            ),
            TestCaseGetDataLoaderConfigError(
                dataset_mime_type=MIME_TYPE_JSONLINES,
                error_message="JSONLines datasets must be stored in a single file",
            ),
        ],
    )
    def test_get_data_loader_config_error(self, dataset_mime_type, error_message):
        """
        GIVEN isinstance returns False
        WHEN _get_data_loader_config is validating the type of its `data_source` argument
        THEN an EvalAlgorithmInternalError is raised
        """
        with patch("fmeval.data_loaders.util.isinstance", return_value=False):
            config = Mock(spec=DataConfig)
            config.dataset_mime_type = dataset_mime_type
            config.dataset_name = "dataset"
            with pytest.raises(EvalAlgorithmInternalError, match=error_message):
                _get_data_loader_config(Mock(spec=DataSource), config)

    class TestCaseGetDataLoaderSuccess(NamedTuple):
        dataset_mime_type: str
        data_loader_config_class: type[JsonDataLoaderConfig]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseGetDataLoaderSuccess(
                dataset_mime_type=MIME_TYPE_JSON, data_loader_config_class=JsonDataLoaderConfig
            ),
            TestCaseGetDataLoaderSuccess(
                dataset_mime_type=MIME_TYPE_JSONLINES, data_loader_config_class=JsonDataLoaderConfig
            ),
        ],
    )
    @patch("fmeval.data_loaders.util.isinstance", return_value=True)
    def test_get_data_loader_config_success(self, mock_isinstance, test_case):
        """
        GIVEN argument validations pass
        WHEN _get_data_loader_config is called
        THEN the correct dataloader class is returned
        """
        config = Mock(spec=DataConfig)
        config.dataset_name = "unused"
        config.dataset_mime_type = test_case.dataset_mime_type
        assert isinstance(_get_data_loader_config(Mock(spec=DataSource), config), test_case.data_loader_config_class)

    class TestCaseGetDataLoader(NamedTuple):
        dataset_mime_type: str
        expected_class: type[JsonDataLoader]

    @pytest.mark.parametrize(
        "dataset_mime_type, expected_class",
        [
            TestCaseGetDataLoader(dataset_mime_type=MIME_TYPE_JSON, expected_class=JsonDataLoader),
            TestCaseGetDataLoader(dataset_mime_type=MIME_TYPE_JSONLINES, expected_class=JsonDataLoader),
        ],
    )
    def test_get_data_loader(self, dataset_mime_type, expected_class):
        """
        GIVEN a dataset MIME type
        WHEN _get_data_loader is called
        THEN the correct dataloader class is returned
        """
        assert _get_data_loader(dataset_mime_type) == expected_class

    def test_get_data_source_provides_local_data_source(self):
        """
        GIVEN a dataset URI corresponding to a local file
        WHEN get_data_source is called
        THEN a LocalDataFile with the correct URI is returned
        """
        dataset_uri = LOCAL_PREFIX + DATASET_URI
        with (
            patch("os.path.isfile", return_value=True),
            patch("os.path.exists", return_value=True),
            patch("os.path.abspath", return_value=dataset_uri),
        ):
            data_source = get_data_source(dataset_uri)
            assert isinstance(data_source, LocalDataFile)
            assert data_source.uri == dataset_uri

    def test_get_data_source_provides_s3_data_source(self):
        """
        GIVEN a dataset URI corresponding to an S3 file
        WHEN get_data_source is called
        THEN an S3DataFile with the correct URI is returned
        """
        with patch("src.fmeval.data_loaders.data_sources.boto3.client") as mock_s3_client:
            mock_s3_client.get_object = Mock(return_value={"ContentType": "binary/octet-stream"})
            dataset_uri = S3_PREFIX + DATASET_URI
            data_source = get_data_source(dataset_uri)
            assert isinstance(data_source, S3DataFile)
            assert data_source.uri == dataset_uri

    def test_get_data_sources_local_directory_exception(self):
        """
        GIVEN a dataset URI corresponding to a local directory
        WHEN get_data_source is called
        THEN the correct exception is raised
        """
        dataset_uri = LOCAL_PREFIX + DIRECTORY_URI
        with (
            patch("os.path.isdir", return_value=True),
            patch("os.path.exists", return_value=True),
            patch("os.path.abspath", return_value=dataset_uri),
        ):
            with pytest.raises(
                EvalAlgorithmClientError, match="Please provide a local file path instead of a directory path."
            ):
                get_data_source(dataset_uri)

    def test_get_data_sources_s3_directory_exception(self):
        """
        GIVEN a dataset URI corresponding to an S3 directory
        WHEN get_data_source is called
        THEN the correct exception is raised
        """
        with patch("src.fmeval.data_loaders.data_sources.boto3.client") as mock_s3_client:
            mock_s3_client.return_value.get_object = Mock(
                return_value={"ContentType": "application/x-directory; charset=UTF-8"}
            )
            dataset_uri = S3_PREFIX + DIRECTORY_URI
            with pytest.raises(
                EvalAlgorithmClientError, match="Please provide a s3 file path instead of a directory path."
            ):
                get_data_source(dataset_uri)

    def test_get_data_source_invalid_local_path(self):
        """
        GIVEN a local path that does not correspond to a file or directory
        WHEN get_data_source is called
        THEN the correct exception is raised
        """
        dataset_uri = LOCAL_PREFIX + INVALID_DATASET_URI
        with patch("os.path.exists", return_value=True), patch("os.path.isfile", return_value=False), patch(
            "os.path.isdir", return_value=False
        ):
            with pytest.raises(EvalAlgorithmClientError, match="Invalid local path"):
                get_data_source(dataset_uri)

    def test_get_data_source_invalid_dataset_path(self):
        with (
            patch("src.fmeval.data_loaders.util._is_valid_local_path", return_value=False),
            patch("src.fmeval.data_loaders.util._is_valid_s3_uri", return_value=False),
        ):
            with pytest.raises(EvalAlgorithmClientError, match="Invalid dataset path"):
                get_data_source("unused")

    def test_is_valid_s3_uri_success(self):
        """
        GIVEN a valid S3 URI
        WHEN _is_valid_s3_uri is called
        THEN True is returned
        """
        dataset_uri = S3_PREFIX + DATASET_URI
        with patch("src.fmeval.data_loaders.data_sources.boto3.client") as mock_s3_client:
            mock_s3_client.return_value.get_object.return_value = Mock()
            assert _is_valid_s3_uri(dataset_uri)

    def test_is_valid_s3_uri_invalid_scheme(self):
        """
        GIVEN a URI whose parsed scheme does not belong to
            the list ["s3", "s3n", "s3a"]
        WHEN _is_valid_s3_uri is called
        THEN False is returned
        """
        dataset_uri = "s3b://some/dataset/uri"
        assert not _is_valid_s3_uri(dataset_uri)

    def test_is_valid_s3_uri_client_error(self):
        """
        GIVEN an S3 URI that does not correspond to a real object
        WHEN _is_valid_s3_uri is called
        THEN False is returned
        """
        with patch("src.fmeval.data_loaders.data_sources.boto3.client") as mock_s3_client:
            mock_s3_client.return_value.get_object = Mock(
                side_effect=botocore.errorfactory.ClientError({"error": "blah"}, "blah")
            )
            assert not _is_valid_s3_uri("s3://non/existent/object")

    @pytest.mark.parametrize("file_path", ["path/to/file", "file://path/to/file"])
    def test_is_valid_local_path_success(self, file_path):
        """
        GIVEN a valid local path
        WHEN _is_valid_local_path is called
        THEN True is returned
        """
        with patch("os.path.exists", return_value=True):
            assert _is_valid_local_path(file_path)

    def test_is_valid_local_path_invalid_scheme(self):
        """
        GIVEN a path with a parsed scheme that does not belong to ["", "file"]
        WHEN _is_valid_local_path is called
        THEN False is returned
        """
        assert not _is_valid_local_path("bad_scheme://path/to/file")

    @pytest.mark.parametrize("file_path", ["path/to/file", "file://path/to/file"])
    def test_is_valid_local_path_file_does_not_exist(self, file_path):
        """
        GIVEN a path to a file that doesn't exist
        WHEN _is_valid_local_path is called
        THEN False is returned
        """
        with patch("os.path.exists", return_value=False):
            assert not _is_valid_local_path(file_path)
