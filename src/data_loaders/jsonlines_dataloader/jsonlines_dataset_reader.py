from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass

from orchestrator.utils import util
from data_loaders.headers.header_factory import HeaderFactory
from data_loaders.data_resources.data_resources import DataFile
from data_loaders.jsonlines_dataloader.jsonlines_parser import JsonLinesParser
from data_loaders.headers.header_generator import HeaderGeneratorConfig

from ray.data.datasource.file_based_datasource import (
    FileBasedDatasource,
    _resolve_kwargs,
)
from ray.data.block import BlockAccessor
import ray.data
import pyarrow
import json


@dataclass(frozen=True)
class JsonLinesDatasetReaderConfig:
    """
    This class is used to configure the JsonLinesDatasetReader. Other parameters are inherited from DatasetReaderConfig.

    Attributes:
        jsonlines_parser: JsonLinesParser object created with customer provided JMESPaths, used to parse dataset lines.
        data_file: the DataFile object representing the dataset.
        dataset_name: the name of the dataset for logging purposes.
        headers: List of customer provided dataset headers.
    """

    jsonlines_parser: JsonLinesParser
    data_file: DataFile
    dataset_name: str
    headers: List[str]


class JsonLinesDatasetReader:
    """
    This class reads a JSON Lines dataset and returns a Ray Dataset.
    """

    @staticmethod
    def read_dataset(config: JsonLinesDatasetReaderConfig) -> ray.data.Dataset:
        """
        :param config: see JsonLinesDatasetReaderConfig docstring.
        :return: a Ray Dataset object of the parsed customer dataset with headers.
        """
        return ray.data.read_datasource(datasource=CustomJSONLinesDatasource(config), paths=config.data_file.uri)

    @staticmethod
    def write_dataset(dataset: ray.data.Dataset, path: str, dataset_uuid: str) -> None:
        """Writes a Ray Dataset to a directory specified by `path`.

        Performs the same logic as ray.data.Dataset.write_json, since our
        CustomJSONLinesDatasource copies the _write_block function from Ray's
        own JSONDatasource verbatim.

        https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.write_json.html

        :param dataset: The Ray Dataset to be written to output files
        :param path: The path to the directory that will store the written files
        :param dataset_uuid: An identifier used to format the names of the output files.
           Output files are named according to the pattern {self._uuid}_{block_idx}.jsonl
           (see the docs for write_json for more details). The jsonl extension comes from
           the _FILE_EXTENSION variable defined in CustomJSONLinesDatasource.
        """
        dataset.write_datasource(datasource=CustomJSONLinesDatasource(), path=path, dataset_uuid=dataset_uuid)

    @staticmethod
    def _get_headers(config: JsonLinesDatasetReaderConfig, dataset_line: Dict) -> List[str]:  # pragma: no cover
        """
        This method is used to get headers from the HeaderFactory.

        :param config: see JsonLinesDatasetReaderConfig docstring.
        :param dataset_line: dataset line used to deduce the number of dataset columns of each type.
        :return: list of dataset headers, including customer provided feature header and artificially generated headers
                 for target_output, inference_output and category if these columns exist.
        """
        line = config.jsonlines_parser.parse_dataset_line(dataset_line, config.dataset_name)
        header_config = HeaderGeneratorConfig.create_from_jsonlines(line, config.headers)
        output_headers = HeaderFactory.get_headers(config.headers, header_config)
        return output_headers

    @staticmethod
    def _parse_dataset_line(
        line: Dict, parser: JsonLinesParser, dataset_headers: List[str], dataset_name: str
    ) -> Dict:  # pragma: no cover
        """
        This method parses a line in the JsonLines dataset and creates a dictionary where each header is paired with
            one dataset value.
            For example: Given a parsed sample where features=["f1", "f2"], target_output=["t1"], inference_output="i1"
            and headers ["h1", "h2", "h3", "h4"], the resulting dict would be
            {"h1": "f1", "h2": "f2", "h3": "t1", "h4": "i4"}.

        :param line: a dataset line from the JSON Lines dataset.
        :param parser: a JsonLinesParser used to parse dataset lines.
        :param dataset_headers: a list of dataset headers, including customer provided features headers and artificial
                                headers for target_output, inference_output and category if the column(s) exist.
        :param dataset_name: the dataset name for logging purposes.
        :return: a dictionary representing the parsed dataset line, where keys are headers and values are dataset
                 column values. Key order is: features, target_outputs, inference_outputs, category.
        """
        parsed_sample = parser.parse_dataset_line(line, dataset_name)
        sample = parsed_sample.features if isinstance(parsed_sample.features, list) else [parsed_sample.features]
        for columns in [parsed_sample.target_outputs, parsed_sample.inference_outputs, parsed_sample.category]:
            if columns:
                sample.extend(columns) if isinstance(columns, list) else sample.append(columns)
        util.require(
            len(dataset_headers) == len(sample),
            f"Expected {len(dataset_headers)} dataset values for each sample but found a mismatch in at least one "
            f"sample with length {len(sample)}, please check that each dataset sample contains the same number of "
            f"features, target_output(s) (if provided) and inference_output(s) (if provided).",
        )
        result = dict(zip(dataset_headers, sample))
        return result


class CustomJSONLinesDatasource(FileBasedDatasource):
    """Custom JSON Lines datasource class.

    See https://docs.ray.io/en/latest/data/examples/custom-datasource.html#custom-datasources
    for details on creating custom datasources.

    We use this class in conjunction with ray.data.read_datasource instead of
    using ray.data.read_json because ray.data.read_json relies on pyarrow._json.read_json,
    which cannot handle JSON (or JSON Lines) files that contain heterogeneous lists
    (lists with elements of different data types).

    Example JSON Lines dataset that pyarrow._json.read_json cannot handle:
    {"key": [20, "hello"]}

    Attributes:
          config: The JSONLinesDatasetReaderConfig used by _read_file when calling
            JsonLinesDatasetReader._parse_dataset_line. Optional, because a config
            is not required for _write_block.
    """

    _FILE_EXTENSION = "jsonl"  # configures the extension for files written by _write_block

    def __init__(self, config: Optional[JsonLinesDatasetReaderConfig] = None):
        if config:
            self.config = config

    def _read_file(self, f: "pyarrow.NativeFile", path: str, **reader_args) -> pyarrow.Table:  # pragma: no cover
        """
        Reads the JSON Lines dataset file given by `path` using the standard json library
        and returns a pyarrow.Table representing the dataset.

        :param f: Unused. Required so that this class conforms to FileBasedDatasource.
        :param path: The path to the dataset JSON Lines file
        :param reader_args: Unused. Required so that this class conforms to FileBasedDatasource.
        """
        with open(path) as file_handle:
            json_lines = [json.loads(line) for line in file_handle]
            dataset_headers = JsonLinesDatasetReader._get_headers(self.config, json_lines[0])
            parsed_json_lines = [
                JsonLinesDatasetReader._parse_dataset_line(
                    line, self.config.jsonlines_parser, dataset_headers, self.config.dataset_name
                )
                for line in json_lines
            ]
            return pyarrow.Table.from_pylist(parsed_json_lines)

    def _write_block(
        self,
        f: "pyarrow.NativeFile",
        block: BlockAccessor,
        writer_args_fn: Callable[[], Dict[str, Any]] = lambda: {},
        **writer_args,
    ):  # pragma: no cover
        """
        Copied directly from Ray's JSONDatasource class.
        """
        writer_args = _resolve_kwargs(writer_args_fn, **writer_args)
        orient = writer_args.pop("orient", "records")
        lines = writer_args.pop("lines", True)
        block.to_pandas().to_json(f, orient=orient, lines=lines, **writer_args)
