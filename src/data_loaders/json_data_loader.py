import ray.data
import pyarrow
import json

from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass

from constants import MIME_TYPE_JSON, MIME_TYPE_JSONLINES
from data_loaders.json_parser import JsonParser
from data_loaders.data_sources import DataFile

from ray.data.datasource.file_based_datasource import (
    FileBasedDatasource,
    _resolve_kwargs,
)
from ray.data.block import BlockAccessor

from exceptions import EvalAlgorithmInternalError


@dataclass(frozen=True)
class JsonDataLoaderConfig:
    """Configures a JsonDataLoader or JsonLinesDataLoader.

    Attributes:
        parser: The JsonParser object used to parse the dataset.
        data_file: The DataFile object representing the dataset.
        dataset_name: The name of the dataset for logging purposes.
        dataset_mime_type: Either MIME_TYPE_JSON or MIME_TYPE_JSONLINES
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

    Attributes:
          config: The config used by _read_file to determine whether to treat the
            input file as a JSON or JSON Lines file.
    """

    _FILE_EXTENSION = "json"  # configures the extension for files written by _write_block

    def __init__(self, config: Optional[JsonDataLoaderConfig] = None):
        super().__init__()
        if config:
            self.config = config

    def _read_file(self, f: "pyarrow.NativeFile", path: str, **reader_args) -> pyarrow.Table:  # pragma: no cover
        """
        Reads the JSON or JSON Lines dataset file given by `f`, parses the JSON/JSON Lines,
        then returns a pyarrow.Table representing the dataset.

        :param f: The file object to read. Note that pyarrow.NativeFile objects differ
            slightly from regular Python files.
        :param path: Unused. Required so that this class conforms to FileBasedDatasource.
        :param reader_args: Unused. Required so that this class conforms to FileBasedDatasource.
        """
        parser = self.config.parser
        if self.config.dataset_mime_type == MIME_TYPE_JSON:
            dataset = json.load(f)
            pydict = parser.parse_dataset_columns(
                dataset=dataset, dataset_mime_type=MIME_TYPE_JSON, dataset_name=self.config.dataset_name
            )
            return pyarrow.Table.from_pydict(pydict)
        elif self.config.dataset_mime_type == MIME_TYPE_JSONLINES:
            json_lines_strings = f.readall().decode().strip().split("\n")
            json_lines = [json.loads(line) for line in json_lines_strings]
            parsed_json_lines = [
                parser.parse_dataset_columns(
                    dataset=line, dataset_mime_type=MIME_TYPE_JSONLINES, dataset_name=self.config.dataset_name
                )
                for line in json_lines
            ]
            return pyarrow.Table.from_pylist(parsed_json_lines)
        else:  # pragma: no cover
            raise EvalAlgorithmInternalError(
                f"Got an unexpected dataset MIME type {self.config.dataset_mime_type} "
                "that is not JSON or JSON Lines."
            )

    def _write_block(
        self,
        f: "pyarrow.NativeFile",
        block: BlockAccessor,
        writer_args_fn: Callable[[], Dict[str, Any]] = lambda: {},
        **writer_args,
    ):  # pragma: no cover
        """
        Copied directly from Ray's own JSONDatasource class.
        """
        writer_args = _resolve_kwargs(writer_args_fn, **writer_args)
        orient = writer_args.pop("orient", "records")
        lines = writer_args.pop("lines", True)
        block.to_pandas().to_json(f, orient=orient, lines=lines, **writer_args)
