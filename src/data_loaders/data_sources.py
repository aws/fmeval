from typing import IO
from abc import ABC, abstractmethod
from s3fs import S3FileSystem


class DataSource(ABC):
    """
    Managed data resource
    """
    def __init__(self, uri: str):
        self._uri = uri
        s3 = S3FileSystem()

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
            raise UserError(f"Unable to open '{self.uri}'. Please make sure the local file path is valid.") from e


class S3DataFile(DataFile):
    """
    DataFile class for s3 files
    """

    def __init__(self, s3: S3FileSystem, file_path: str):
        self._s3 = s3
        super().__init__(file_path)

    def open(self, mode="r") -> IO:
        try:
            return self._s3.open(self.uri, mode=mode)
        except Exception as e:
            raise UserError(f"Unable to open '{self.uri}'. Please make sure the s3 file path is valid.") from e
