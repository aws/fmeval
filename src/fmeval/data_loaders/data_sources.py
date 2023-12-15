import boto3
import botocore.response
import botocore.errorfactory
import urllib.parse
from typing import IO
from abc import ABC, abstractmethod
from fmeval.exceptions import EvalAlgorithmClientError


class DataSource(ABC):
    """
    Managed data resource
    """

    def __init__(self, uri: str):
        self._uri = uri

    @property
    def uri(self) -> str:
        """
        :return: path to the resource
        """
        return self._uri


class DataFile(DataSource):
    """
    Managed data file resource
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)

    @abstractmethod
    def open(self, mode="r") -> IO:
        """
        :param mode: optional mode to open file, default 'r' is readonly
        :return: File object
        """


class LocalDataFile(DataFile):
    """
    Datafile class for local files
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)

    def open(self, mode="r") -> IO:
        try:
            return open(self.uri, mode)
        except Exception as e:
            raise EvalAlgorithmClientError(
                f"Unable to open '{self.uri}'. Please make sure the local file path is valid."
            ) from e


class S3Uri:
    """
    This class represents an S3 URI, encapsulating the logic
    for parsing the S3 bucket and key from the raw URI.
    """

    def __init__(self, uri):
        self._parsed = urllib.parse.urlparse(uri, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip("/") + "?" + self._parsed.query
        else:
            return self._parsed.path.lstrip("/")


class S3DataFile(DataFile):
    """
    DataFile class for s3 files
    """

    def __init__(self, file_path: str):
        # We cannot inject the client b/c
        # it is not serializable by Ray.
        self._client = boto3.client("s3")
        super().__init__(file_path)

    def open(self, mode="r") -> botocore.response.StreamingBody:  # type: ignore
        try:
            s3_uri = S3Uri(self.uri)
            return self._client.get_object(Bucket=s3_uri.bucket, Key=s3_uri.key)["Body"]
        except botocore.errorfactory.ClientError as e:
            raise EvalAlgorithmClientError(
                f"Unable to open '{self.uri}'. Please make sure the s3 file path is valid."
            ) from e

    def __reduce__(self):
        """
        Custom serializer method used by Ray when it serializes
        JsonDataLoaderConfig objects during data loading
        (see the load_dataset method in src.fmeval.data_loaders.json_data_loader.py).
        """
        serialized_data = (self.uri,)
        return S3DataFile, serialized_data
