import json

import pytest
import pathlib

from fmeval.data_loaders.data_sources import LocalDataFile
from fmeval.data_loaders.json_parser import JsonParser
from fmeval.data_loaders.json_data_loader import (
    JsonDataLoader,
    JsonDataLoaderConfig,
    CustomJSONDatasource,
)
from fmeval.data_loaders.util import DataConfig
from typing import Any, Dict, List, NamedTuple, Optional
from fmeval.constants import (
    MODEL_INPUT_COLUMN_NAME,
    MODEL_OUTPUT_COLUMN_NAME,
    TARGET_OUTPUT_COLUMN_NAME,
    CATEGORY_COLUMN_NAME,
    MIME_TYPE_JSON,
    MIME_TYPE_JSONLINES,
)


def create_temp_json_data_file_from_input_dataset(path: pathlib.Path, input_dataset: Dict[str, Any]):
    """Creates a LocalDataFile for a JSON dataset at a temporary path.

    :param path: The output of the tmp_path fixture
    :param input_dataset: A dict representing a JSON dataset
    :returns: A LocalDataFile for the input dataset
    """
    dataset_path = path / "dataset.json"
    dataset_path.write_text(json.dumps(input_dataset))
    return LocalDataFile(dataset_path)


def create_temp_jsonlines_data_file_from_input_dataset(path: pathlib.Path, input_dataset: List[Dict[str, Any]]):
    """Creates a LocalDataFile for a JSON Lines dataset at a temporary path.

    :param path: The output of the tmp_path fixture
    :param input_dataset: A list of dicts representing a JSON Lines dataset
    :returns: A LocalDataFile for the input dataset
    """
    dataset_path = path / "dataset.jsonl"
    dataset_path.write_text("\n".join([json.dumps(sample) for sample in input_dataset]))
    return LocalDataFile(dataset_path)


class TestJsonDataLoader:
    class TestCaseReadDataset(NamedTuple):
        input_dataset: Dict[str, Any]
        expected_dataset: List[Dict[str, Any]]
        dataset_mime_type: str
        model_input_jmespath: Optional[str] = None
        model_output_jmespath: Optional[str] = None
        target_output_jmespath: Optional[str] = None
        category_jmespath: Optional[str] = None

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseReadDataset(
                input_dataset={"model_input_col": ["a", "b", "c"]},
                expected_dataset=[
                    {MODEL_INPUT_COLUMN_NAME: "a"},
                    {MODEL_INPUT_COLUMN_NAME: "b"},
                    {MODEL_INPUT_COLUMN_NAME: "c"},
                ],
                dataset_mime_type=MIME_TYPE_JSON,
                model_input_jmespath="model_input_col",
            ),
            # The reason this dataset has an unusual format (it's basically a poor man's jsonlines)
            # is that we want to ensure that JsonDataLoader.load_dataset can handle JSON datasets
            # containing heterogeneous lists.
            TestCaseReadDataset(
                input_dataset={
                    "row_1": ["a", "positive", "positive", 0],
                    "row_2": ["b", "negative", "negative", 1],
                    "row_3": ["c", "negative", "positive", 2],
                },
                expected_dataset=[
                    {
                        MODEL_INPUT_COLUMN_NAME: "a",
                        MODEL_OUTPUT_COLUMN_NAME: "positive",
                        TARGET_OUTPUT_COLUMN_NAME: "positive",
                        CATEGORY_COLUMN_NAME: 0,
                    },
                    {
                        MODEL_INPUT_COLUMN_NAME: "b",
                        MODEL_OUTPUT_COLUMN_NAME: "negative",
                        TARGET_OUTPUT_COLUMN_NAME: "negative",
                        CATEGORY_COLUMN_NAME: 1,
                    },
                    {
                        MODEL_INPUT_COLUMN_NAME: "c",
                        MODEL_OUTPUT_COLUMN_NAME: "positive",
                        TARGET_OUTPUT_COLUMN_NAME: "negative",
                        CATEGORY_COLUMN_NAME: 2,
                    },
                ],
                dataset_mime_type=MIME_TYPE_JSON,
                model_input_jmespath="[row_1[0], row_2[0], row_3[0]]",
                model_output_jmespath="[row_1[2], row_2[2], row_3[2]]",
                target_output_jmespath="[row_1[1], row_2[1], row_3[1]]",
                category_jmespath="[row_1[3], row_2[3], row_3[3]]",
            ),
            TestCaseReadDataset(
                input_dataset=[
                    {"input": "a", "output": "b"},
                    {"input": "c", "output": "d"},
                    {"input": "e", "output": "f"},
                ],
                expected_dataset=[
                    {MODEL_INPUT_COLUMN_NAME: "a", MODEL_OUTPUT_COLUMN_NAME: "b"},
                    {MODEL_INPUT_COLUMN_NAME: "c", MODEL_OUTPUT_COLUMN_NAME: "d"},
                    {MODEL_INPUT_COLUMN_NAME: "e", MODEL_OUTPUT_COLUMN_NAME: "f"},
                ],
                dataset_mime_type=MIME_TYPE_JSONLINES,
                model_input_jmespath="input",
                model_output_jmespath="output",
            ),
        ],
    )
    def test_read_dataset(self, tmp_path, test_case):
        """
        GIVEN a JsonDataLoaderConfig with a valid JsonParser and data_file
        WHEN load_dataset is called
        THEN the expected Ray Dataset is returned
        """
        data_file = (
            create_temp_json_data_file_from_input_dataset(tmp_path, test_case.input_dataset)
            if test_case.dataset_mime_type == MIME_TYPE_JSON
            else create_temp_jsonlines_data_file_from_input_dataset(tmp_path, test_case.input_dataset)
        )

        parser = JsonParser(
            DataConfig(
                dataset_name="dataset",
                dataset_uri="uri",
                dataset_mime_type=MIME_TYPE_JSON,
                model_input_location=test_case.model_input_jmespath,
                model_output_location=test_case.model_output_jmespath,
                target_output_location=test_case.target_output_jmespath,
                category_location=test_case.category_jmespath,
            )
        )
        config = JsonDataLoaderConfig(
            parser=parser, data_file=data_file, dataset_name="dataset", dataset_mime_type=test_case.dataset_mime_type
        )
        dataset = JsonDataLoader.load_dataset(config)
        assert dataset.columns() == list(test_case.expected_dataset[0].keys()), "dataset.columns() is {}".format(
            dataset.columns()
        )
        num_rows = 3
        assert dataset.count() == num_rows
        assert sorted(dataset.take(num_rows), key=lambda x: x[MODEL_INPUT_COLUMN_NAME]) == test_case.expected_dataset
