import random

import pytest
from typing import NamedTuple
from unittest.mock import call, Mock, patch

import ray.data

from amazon_fmeval.constants import MIME_TYPE_JSON, MIME_TYPE_JSONLINES
from amazon_fmeval.data_loaders.data_sources import LocalDataFile, S3DataFile, DataSource
from amazon_fmeval.data_loaders.json_data_loader import JsonDataLoaderConfig, JsonDataLoader
from amazon_fmeval.data_loaders.util import _get_data_loader, _get_data_loader_config, get_data_source, get_dataset
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.exceptions import EvalAlgorithmClientError, EvalAlgorithmInternalError

S3_PREFIX = "s3://"
LOCAL_PREFIX = "file://"

DATASET_URI = "dataset.json"
INVALID_DATASET_URI = "invalid_dataset"
DIRECTORY_URI = "dir1/dir2/"


class TestDataLoaderUtil:
    @patch("amazon_fmeval.data_loaders.util._get_data_loader", return_value=Mock())
    @patch("amazon_fmeval.data_loaders.util._get_data_loader_config", return_value=Mock())
    @patch("amazon_fmeval.data_loaders.util.get_data_source", return_value=Mock())
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

    @patch("amazon_fmeval.data_loaders.util._get_data_loader", return_value=Mock())
    @patch("amazon_fmeval.data_loaders.util._get_data_loader_config", return_value=Mock())
    @patch("amazon_fmeval.data_loaders.util.get_data_source", return_value=Mock())
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
        with patch("amazon_fmeval.data_loaders.util.isinstance", return_value=False):
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
    @patch("amazon_fmeval.data_loaders.util.isinstance", return_value=True)
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
        with patch("amazon_fmeval.data_loaders.util.s3") as mock_s3fs:
            mock_s3fs.info = Mock(return_value={"type": "file"})
            mock_s3fs.exists = Mock(return_value=True)
            dataset_uri = S3_PREFIX + DATASET_URI
            data_source = get_data_source(dataset_uri)
            assert isinstance(data_source, S3DataFile)
            assert data_source.uri == dataset_uri

    def test_get_data_sources_local_directory_exception(self):
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
        with patch("amazon_fmeval.data_loaders.util.s3") as mock_s3fs:
            mock_s3fs.info = Mock(return_value={"type": "directory"})
            mock_s3fs.exists = Mock(return_value=True)
            dataset_uri = S3_PREFIX + DIRECTORY_URI
            with pytest.raises(
                EvalAlgorithmClientError, match="Please provide a s3 file path instead of a directory path."
            ):
                get_data_source(dataset_uri)

    def test_get_data_source_invalid_local_path(self):
        dataset_uri = LOCAL_PREFIX + INVALID_DATASET_URI
        with patch("os.path.exists", return_value=True), patch("os.path.isfile", return_value=False), patch(
            "os.path.isdir", return_value=False
        ):
            with pytest.raises(EvalAlgorithmClientError, match="Invalid local path"):
                get_data_source(dataset_uri)

    def test_get_data_source_invalid_s3_path(self):
        dataset_uri = S3_PREFIX + INVALID_DATASET_URI
        with patch("amazon_fmeval.data_loaders.util.s3") as mock_s3fs:
            mock_s3fs.info = Mock(return_value={"type": "Other"})
            mock_s3fs.exists = Mock(return_value=True)
            with pytest.raises(EvalAlgorithmClientError, match="Invalid s3 path"):
                get_data_source(dataset_uri)

    def test_get_data_source_invalid_dataset_path(self):
        with pytest.raises(EvalAlgorithmClientError, match="Invalid dataset path"):
            get_data_source(INVALID_DATASET_URI)
