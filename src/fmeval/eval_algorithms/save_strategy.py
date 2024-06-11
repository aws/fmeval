import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import boto3
from sagemaker.s3_utils import parse_s3_url

from fmeval import util
from fmeval.constants import PARTS, UPLOAD_ID, PART_NUMBER, E_TAG
from fmeval.eval_algorithms.util import EvalOutputRecord

logger = logging.getLogger(__name__)


class SaveStrategy(ABC):
    """Interface that defines how to save the eval outputs.

    The save function of this interface may be called multiple times based on the size of the dataset. This is due to
    the distributed nature of the computations. If the dataset is large, and all of the data is pulled to the head node,
    it might lead to OOM errors. In order to avoid that, the data is pulled in batches, and `save` function is called on
    each batch at a time. In order to allow this mechanism, while allowing more flexbility in the way outputs are saved,
    this class works as a ContextManager.
    """

    def __enter__(self) -> "SaveStrategy":
        """Sets up the strategy to start saving the evaluation outputs."""
        self.start()
        return self

    @abstractmethod
    def start(self):
        """Sets up the strategy to write the evaluation output records."""

    @abstractmethod
    def save(self, records: List[EvalOutputRecord]):
        """Saves the given list of EvalOutputRecord based on the strategy.

        Each invocation of this function would be for one block of the data. There could be multiple invocations of save
        per invocation of the `evaluate` function (especially if the data is large).

        :param records: list of EvalOutputRecords to be saved
        """

    @abstractmethod
    def clean_up(self):
        """
        Clean up any leftover resources after saving the outputs.
        """

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clean_up()
        return False


class FileSaveStrategy(SaveStrategy):
    """Strategy to write evaluation outputs to local file system.

    The strategy appends multiple invocations of save to the same file. If the file already exists, it will be
    overwritten.
    """

    def __init__(self, file_path: str):
        self._file_path = file_path
        self._call_count = 0
        if os.path.exists(self._file_path):
            logger.warning(f"File {self._file_path} exists. Overwriting existing file")

    def start(self):
        """Sets up the strategy to write the evaluation output records"""
        with open(self._file_path, mode="w") as file:
            file.write("")

    def save(self, records: List[EvalOutputRecord]):
        """Append the list of evalution output records to the file at provided file path

        :param records: list of EvalOutputRecords to be saved
        """
        with open(self._file_path, mode="a") as file:
            if self._call_count > 0:
                file.write("\n")
            file.writelines("\n".join([json.dumps(record.to_dict()) for record in records]))
            self._call_count += 1

    def clean_up(self):
        self._call_count = 0


class S3SaveStrategy(SaveStrategy):
    """Strategy to write evaluation outputs to AWS S3.

    The strategy appends multiple invocations of save to the same file. If the file already exists, it will be
    overwritten.
    """

    def __init__(self, s3_uri: str, s3_boto_client, Optional=None, kms_key_id: Optional[str] = None):
        """Creates an instance of S3SaveStrategy

        :param s3_uri: The S3 uri where the outputs should be written to
        :param s3_boto_client: The boto3 client for S3. If not provided, the class will try to use the default S3 client
        :param kms_key_id: If provided, this KMS Key will be used for server side encryption
        """
        self._bucket, self._key_prefix = parse_s3_url(url=s3_uri)
        self._s3_client = s3_boto_client if s3_boto_client else boto3.client("s3")
        self._multi_part_upload = None
        self._part_info: Optional[Dict] = None
        self._kms_key_id = kms_key_id

    def start(self):
        """Sets up the strategy to write the evaluation output records to S3 using multi-part uploads."""
        self._multi_part_upload = self._s3_client.create_multipart_upload(
            Bucket=self._bucket, Key=self._key_prefix, SSEKMSKeyId=self._kms_key_id
        )
        self._part_info = {PARTS: []}
        return self

    def save(self, records: List[EvalOutputRecord]):
        """Creates and uploads a part using the list of evaluation output records so that they can be completed during
        the clean up stage.

        :param records: list of EvalOutputRecords to be saved
        """
        util.require(
            self._part_info and self._multi_part_upload, "S3SaveStrategy is meant to be used as a context manager"
        )
        assert self._part_info and self._multi_part_upload  # to satisfy mypy
        part_number = len(self._part_info[PARTS]) + 1
        part = self._s3_client.upload_part(
            Bucket=self._bucket,
            Key=self._key_prefix,
            PartNumber=part_number,
            UploadId=self._multi_part_upload[UPLOAD_ID],
            Body="\n".join([str(record) for record in records]),
        )
        self._part_info[PARTS].append({PART_NUMBER: part_number, E_TAG: part[E_TAG]})

    def clean_up(self):
        """Completes the multi-part upload to S3, which then collects and combines the parts together into one object"""
        self._s3_client.complete_multipart_upload(
            Bucket=self._bucket,
            Key=self._key_prefix,
            UploadId=self._multi_part_upload[UPLOAD_ID],
            MultipartUpload=self._part_info,
        )
        self._multi_part_upload = None
        self._part_info = {PARTS: []}
