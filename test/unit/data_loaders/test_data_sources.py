import io
import pickle
import pytest
import botocore.response
import botocore.errorfactory

from unittest.mock import patch, Mock, mock_open

from fmeval.constants import BUILT_IN_DATASET_ISO_REGIONS
from fmeval.data_loaders.data_sources import LocalDataFile, S3DataFile, S3Uri, get_s3_client
from fmeval.eval_algorithms import DATASET_CONFIGS, TREX
from fmeval.exceptions import EvalAlgorithmClientError

S3_PREFIX = "s3://"
LOCAL_PREFIX = "file://"

DATASET_URI = "dataset.json"
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
    @pytest.mark.parametrize("file_path", FILE_PATHS)
    def test_open_s3_data_file(self, file_path):
        with patch("fmeval.data_loaders.data_sources.boto3.client") as mock_boto3_client:
            mock_s3_client = Mock()
            mock_boto3_client.return_value = mock_s3_client
            with io.StringIO() as buf:
                buf.write("data")
                buf.seek(0)
                mock_s3_client.get_object.return_value = {"Body": botocore.response.StreamingBody(buf, len("data"))}
                data_file = S3DataFile(file_path=S3_PREFIX + file_path)
                assert data_file.open().read() == "data"
                s3_uri = S3Uri(S3_PREFIX + file_path)
                mock_s3_client.get_object.assert_called_once_with(Bucket=s3_uri.bucket, Key=s3_uri.key)

    @pytest.mark.parametrize("invalid_file_path", INVALID_FILE_PATHS)
    def test_open_invalid_s3_data_file(self, invalid_file_path):
        with patch("fmeval.data_loaders.data_sources.boto3.client") as mock_boto3_client:
            mock_s3_client = Mock()
            mock_s3_client.get_object.side_effect = botocore.errorfactory.ClientError({"error": "blah"}, "blah")
            mock_boto3_client.return_value = mock_s3_client
            with pytest.raises(EvalAlgorithmClientError):
                S3DataFile(file_path=S3_PREFIX + invalid_file_path).open()

    @pytest.mark.parametrize("file_path", FILE_PATHS)
    def test_reduce(self, file_path):
        s3_data_file = S3DataFile(file_path=S3_PREFIX + file_path)
        deserialized = pickle.loads(pickle.dumps(s3_data_file))
        assert deserialized.uri == s3_data_file.uri


class TestS3Uri:
    def test_bucket(self):
        s3_uri = S3Uri("s3://bucket/hello/world")
        assert s3_uri.bucket == "bucket"

    @pytest.mark.parametrize(
        "uri, key",
        [
            ("s3://bucket/hello/world", "hello/world"),
            ("s3://bucket/hello/world?qwe1=3#ddd", "hello/world?qwe1=3#ddd"),
            ("s3://bucket/hello/world#foo?bar=2", "hello/world#foo?bar=2"),
        ],
    )
    def test_key(self, uri, key):
        s3_uri = S3Uri(uri)
        assert s3_uri.key == key


@pytest.mark.parametrize(
    "run_region, dataset_region",
    [
        ("us-west-2", "us-west-2"),
        ("ap-east-1", "us-west-2"),
        ("us-isof-south-1", "us-isof-south-1"),
        ("us-isof-east-1", "us-isof-south-1"),
    ],
)
@patch("boto3.session.Session")
def test_get_s3_client_built_in_dataset(mock_session_class, run_region, dataset_region):
    """
    GIVEN a built-in dataset s3 path
    WHEN get_s3_client is called
    THEN the boto3 s3 client is created with corresponding built-in dataset region name
    """
    with patch("boto3.client") as mock_client:
        mock_instance = mock_session_class.return_value
        mock_instance.region_name = run_region
        dataset_uri = DATASET_CONFIGS[TREX].dataset_uri
        s3_client = get_s3_client(dataset_uri)
        if dataset_region in BUILT_IN_DATASET_ISO_REGIONS.values():
            mock_client.assert_called_once_with("s3", region_name=dataset_region, verify=False)
        else:
            mock_client.assert_called_once_with("s3", region_name=dataset_region)


@pytest.mark.parametrize("region", ["us-west-2", "ap-east-1", "us-isof-south-1", "us-isof-east-1"])
@patch("boto3.session.Session")
def test_get_s3_client_custom_dataset(mock_session_class, region):
    """
    GIVEN a custom dataset s3 path
    WHEN get_s3_client is called
    THEN the boto3 s3 client is created without region name
    """
    with patch("boto3.client") as mock_client:
        mock_instance = mock_session_class.return_value
        mock_instance.region_name = region
        dataset_uri = dataset_uri = S3_PREFIX + DATASET_URI
        s3_client = get_s3_client(dataset_uri)
        if region in BUILT_IN_DATASET_ISO_REGIONS.keys():
            mock_client.assert_called_once_with("s3", verify=False)
        else:
            mock_client.assert_called_once_with("s3")
