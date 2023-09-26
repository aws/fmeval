import urllib
import os
from s3fs import S3FileSystem
from infra.utils.sm_exceptions import CustomerError
from data_loaders.data_resources.data_resources import DataResource, LocalDataFile, S3DataFile


class DataResourceFactory:
    """
    This class implements get_data_resource, which returns a DataResource object from a dataset uri
    """

    s3 = S3FileSystem()

    @staticmethod
    def get_data_resource(dataset_uri: str) -> DataResource:
        """
        validate a dataset uri and return the corresponding DataResource object
        :param dataset_uri: local dataset path or s3 dataset uri
        :return: DataResource object
        """
        if DataResourceFactory._is_valid_local_path(dataset_uri):
            return DataResourceFactory._get_local_data_resource(dataset_uri)
        elif DataResourceFactory._is_valid_s3_uri(dataset_uri):
            return DataResourceFactory._get_s3_data_resource(dataset_uri)
        else:
            raise CustomerError(f"Invalid dataset path: {dataset_uri}")

    @staticmethod
    def _get_local_data_resource(dataset_uri) -> LocalDataFile:
        """
        :param dataset_uri: local dataset path
        :return: LocalDataFile object with dataset uri
        """
        absolute_local_path = os.path.abspath(urllib.parse.urlparse(dataset_uri).path)
        if os.path.isfile(absolute_local_path):
            return LocalDataFile(absolute_local_path)
        if os.path.isdir(absolute_local_path):
            # TODO: extend support to directories (not required for launch): https://sim.amazon.com/issues/RAI-6046
            raise CustomerError("Please provide a local file path instead of a directory path.")
        raise CustomerError(f"Invalid local path: {dataset_uri}")

    @staticmethod
    def _get_s3_data_resource(dataset_uri) -> S3DataFile:
        """
        :param dataset_uri: s3 dataset uri
        :return: S3DataFile object with dataset uri
        """
        s3_info = DataResourceFactory.s3.info(dataset_uri)
        if s3_info["type"] == "file":
            return S3DataFile(DataResourceFactory.s3, dataset_uri)
        if s3_info["type"] == "directory":
            # TODO: extend support to directories (not required for launch): https://sim.amazon.com/issues/RAI-6046
            raise CustomerError("Please provide a s3 file path instead of a directory path.")
        raise CustomerError(f"Invalid s3 path: {dataset_uri}")

    @staticmethod
    def _is_valid_s3_uri(uri: str) -> bool:
        """
        :param uri: s3 file path
        :return: True if uri is a valid s3 path, False otherwise
        """
        parsed_url = urllib.parse.urlparse(uri)
        return parsed_url.scheme.lower() in ["s3", "s3n", "s3a"] and DataResourceFactory.s3.exists(uri)

    @staticmethod
    def _is_valid_local_path(path: str) -> bool:
        """
        :param path: local file path
        :return: True if path is a valid local path, False otherwise
        """
        parsed_url = urllib.parse.urlparse(path)
        return parsed_url.scheme in ["", "file"] and os.path.exists(parsed_url.path)
