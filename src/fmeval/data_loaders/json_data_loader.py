import ray.data
import pyarrow
import json

from dataclasses import dataclass

from fmeval.constants import MIME_TYPE_JSON, MIME_TYPE_JSONLINES
from fmeval.data_loaders.json_parser import JsonParser
from fmeval.data_loaders.data_sources import DataFile

from ray.data.datasource.file_based_datasource import (
    FileBasedDatasource,
    _resolve_kwargs,
)

from fmeval.exceptions import EvalAlgorithmInternalError


@dataclass(frozen=True)
class JsonDataLoaderConfig:
    """Configures a JsonDataLoader or JsonLinesDataLoader.

    :param parser: The JsonParser object used to parse the dataset.
    :param data_file: The DataFile object representing the dataset.
    :param dataset_name: The name of the dataset for logging purposes.
    :param dataset_mime_type: Either MIME_TYPE_JSON or MIME_TYPE_JSONLINES
    """

    parser: JsonParser
    data_file: DataFile
    dataset_name: str
    dataset_mime_type: str


class JsonDataLoader:
    """Reads a JSON or JSON Lines dataset and returns a Ray Dataset."""

    @staticmethod
    def load_dataset(config: JsonDataLoaderConfig) -> ray.data.Dataset:
        """Reads a JSON dataset and returns a Ray Dataset that includes headers.

        :param config: see JsonDataLoaderConfig docstring.
        :return: a Ray Dataset object that includes headers.
        """
        return ray.data.read_datasource(datasource=CustomJSONDatasource(config=config), paths=config.data_file.uri)


class CustomJSONDatasource(FileBasedDatasource):
    """Custom datasource class for reading and writing JSON or JSON Lines files.

    See https://docs.ray.io/en/latest/data/examples/custom-datasource.html#custom-datasources
    for details on creating custom data sources.

    We use this class instead of Ray's own JSONDatasource class because
    Ray's implementation relies on pyarrow._json.read_json, which cannot
    handle JSON files that contain heterogeneous lists
    (lists with elements of different data types).

    Example JSON dataset that pyarrow._json.read_json cannot handle:
    {
       "key": [20, "hello"]
    }

    :param config: The config used by _read_stream to determine whether to treat the
            input file as a JSON or JSON Lines file.
    """

    # A list of file extensions to filter files by.
    # Since this class only reads a single file at a time,
    # this list effectively configures the allowed file
    # extensions for the dataset being read.
    _FILE_EXTENSIONS = ["json", "jsonl"]

    def __init__(self, config: JsonDataLoaderConfig):
        super().__init__(config.data_file.uri, file_extensions=CustomJSONDatasource._FILE_EXTENSIONS)
        self.config = config

    def _read_stream(self, f: "pyarrow.NativeFile", path: str) -> pyarrow.Table:  # pragma: no cover
        """
        Reads the JSON or JSON Lines dataset file given by `f`, parses the JSON/JSON Lines,
        then returns a pyarrow.Table representing the dataset.

        :param f: The file object to read. Note that pyarrow.NativeFile objects differ
            slightly from regular Python files.
        :param path: Unused. Required so that this class conforms to FileBasedDatasource.
        """
        parser = self.config.parser
        if self.config.dataset_mime_type == MIME_TYPE_JSON:
            dataset = json.load(f)
            pydict = parser.parse_dataset_columns(
                dataset=dataset, dataset_mime_type=MIME_TYPE_JSON, dataset_name=self.config.dataset_name
            )
            yield pyarrow.Table.from_pydict(pydict)
        elif self.config.dataset_mime_type == MIME_TYPE_JSONLINES:
            json_lines_strings = f.readall().decode().strip().split("\n")
            json_lines = [json.loads(line) for line in json_lines_strings]
            parsed_json_lines = [
                parser.parse_dataset_columns(
                    dataset=line, dataset_mime_type=MIME_TYPE_JSONLINES, dataset_name=self.config.dataset_name
                )
                for line in json_lines
            ]
            yield pyarrow.Table.from_pylist(parsed_json_lines)
        else:  # pragma: no cover
            raise EvalAlgorithmInternalError(
                f"Got an unexpected dataset MIME type {self.config.dataset_mime_type} "
                "that is not JSON or JSON Lines."
            )
