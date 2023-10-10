from unittest.mock import patch, Mock, mock_open
import pytest
from s3fs import S3FileSystem

from amazon_fmeval.data_loaders.data_sources import LocalDataFile, S3DataFile
from amazon_fmeval.exceptions import EvalAlgorithmClientError

S3_PREFIX = "s3://"
LOCAL_PREFIX = "file://"

FILE_PATHS = ["file1.json", "file2.jsonl", "dir1/file1.json", "dir1/dir2/file1.jsonl"]
INVALID_FILE_PATHS = ["invalid_file1.json", "dir1/invalid_file", "dir1/dir2/"]


class TestLocalDatafile:
    @pytest.mark.parametrize("file_path", FILE_PATHS)
    def test_open_local_data_file(self, file_path):
        with patch("builtins.open", mock_open(read_data="data")) as mocked_open:
            data_file = LocalDataFile(file_path=LOCAL_PREFIX + file_path)
            assert data_file.open().read() == "data"
            mocked_open.assert_called_once_with(LOCAL_PREFIX + file_path, "r")

    @pytest.mark.parametrize("invalid_file_path", INVALID_FILE_PATHS)
    def test_open_invalid_local_data_file(self, invalid_file_path):
        with patch("builtins.open", side_effect=Exception()):
            with pytest.raises(EvalAlgorithmClientError):
                LocalDataFile(file_path=LOCAL_PREFIX + invalid_file_path).open()


class TestS3DataFile:
    mock_s3 = Mock(spec=S3FileSystem)

    @pytest.mark.parametrize("file_path", FILE_PATHS)
    def test_open_s3_data_file(self, file_path):
        TestS3DataFile.mock_s3.open = mock_open(read_data="data")
        data_file = S3DataFile(s3=TestS3DataFile.mock_s3, file_path=S3_PREFIX + file_path)
        assert data_file.open().read() == "data"
        TestS3DataFile.mock_s3.open.assert_called_once_with(S3_PREFIX + file_path, mode="r")

    @pytest.mark.parametrize("invalid_file_path", INVALID_FILE_PATHS)
    def test_open_invalid_s3_data_file(self, invalid_file_path):
        TestS3DataFile.mock_s3.open = Mock(side_effect=Exception())
        with pytest.raises(EvalAlgorithmClientError):
            S3DataFile(s3=TestS3DataFile.mock_s3, file_path=S3_PREFIX + invalid_file_path).open()
