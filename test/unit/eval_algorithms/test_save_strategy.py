import json
import logging
import os
import tempfile
from collections import OrderedDict
from unittest.mock import patch, Mock

from fmeval.constants import DatasetColumns
from fmeval.eval_algorithms import EvalScore
from fmeval.eval_algorithms.save_strategy import FileSaveStrategy, S3SaveStrategy
from fmeval.eval_algorithms.util import EvalOutputRecord


class TestFileSaveStrategy:
    def test_save_and_clean_up(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "file.jsonl")
            with open(file_path, "w") as file:
                file.write("Non-json Content")
            # GIVEN
            records = [
                EvalOutputRecord(
                    scores=[EvalScore(name="score1", value=0.3), EvalScore(name="score2", value=0.8)],
                    dataset_columns={
                        DatasetColumns.MODEL_INPUT.value.name: "Hello",
                        DatasetColumns.PROMPT.value.name: "Summarize: Hello",
                    },
                )
            ] * 3
            # Write 3 times to make sure append functionality works as expected
            num_of_save_times = 3
            logger = logging.getLogger("fmeval.eval_algorithms.save_strategy")
            with patch.object(logger, "warning") as logger:
                with FileSaveStrategy(file_path) as save_strategy:
                    for _ in range(num_of_save_times):
                        save_strategy.save(records)
                logger.assert_called_once_with(f"File {file_path} exists. Overwriting existing file")
            with open(file_path) as file:
                # If each file is valid JSON, we know that the original content was overriden
                json_objects = (json.loads(line, object_pairs_hook=OrderedDict) for line in file.readlines())
                for i, json_obj in enumerate(json_objects):
                    # want to ensure ordering of keys is correct, so we use list instead of set
                    assert list(json_obj.keys()) == [
                        DatasetColumns.MODEL_INPUT.value.name,
                        DatasetColumns.PROMPT.value.name,
                        "scores",
                    ]
                    assert json_obj[DatasetColumns.MODEL_INPUT.value.name] == "Hello"
                    assert json_obj[DatasetColumns.PROMPT.value.name] == "Summarize: Hello"


class TestS3SaveStrategy:
    def test_save_and_clean_up(self):
        # Write 3 times to make sure append functionality works as expected
        num_of_save_times = 3
        s3_client = Mock()
        s3_client.create_multipart_upload.return_value = {"UploadId": "1234"}
        s3_client.upload_part.side_effect = [{"ETag": 1}, {"ETag": 2}, {"ETag": 3}]
        s3_client.complete_multipart_upload.return_value = None
        # GIVEN
        records = [
            EvalOutputRecord(
                scores=[EvalScore(name="score1", value=0.3), EvalScore(name="score2", value=0.8)],
                dataset_columns={
                    DatasetColumns.MODEL_INPUT.value.name: "Hello",
                    DatasetColumns.PROMPT.value.name: "Summarize: Hello",
                },
            )
        ] * 3
        with patch.object(s3_client, "complete_multipart_upload", return_value=None) as complete_multipart_upload:
            with S3SaveStrategy(s3_uri="s3://bucket/key", s3_boto_client=s3_client) as save_strategy:
                for _ in range(num_of_save_times):
                    save_strategy.save(records)
            complete_multipart_upload.assert_called_once_with(
                Bucket="bucket",
                Key="key",
                UploadId="1234",
                MultipartUpload={
                    "Parts": [{"PartNumber": 1, "ETag": 1}, {"PartNumber": 2, "ETag": 2}, {"PartNumber": 3, "ETag": 3}]
                },
            )
