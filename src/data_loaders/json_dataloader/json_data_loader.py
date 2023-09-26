import ray.data
import pyarrow
import numpy as np
import json

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from data_loaders.json_dataloader.json_parser import JsonParser
from data_loaders.data_resources.data_resources import DataFile

from ray.data.datasource.file_based_datasource import (
    FileBasedDatasource,
    _resolve_kwargs,
)
from ray.data.block import BlockAccessor


@dataclass(frozen=True)
class JsonDataLoaderConfig:
    """Configures the JsonLinesDataLoader.

    Attributes:
        json_parser: JsonParser object created with customer provided JMESPaths, used to parse the dataset.
        data_file: the DataFile object representing the dataset.
        dataset_name: the name of the dataset for logging purposes.
        headers: List of customer provided dataset headers for feature columns only.
    """

    json_parser: JsonParser
    data_file: DataFile
    dataset_name: str
    headers: List[str]


class JsonDataLoader:
    """Reads a JSON dataset and returns a Ray Dataset that includes headers."""

    @staticmethod
    def read_dataset(config: JsonDataLoaderConfig) -> ray.data.Dataset:
        """Reads a JSON dataset and returns a Ray Dataset that includes headers.

        :param config: see JsonDataLoaderConfig docstring.
        :return: a Ray Dataset object that includes headers.
        """
        return ray.data.read_datasource(datasource=CustomJSONDatasource(config=config), paths=config.data_file.uri)

    @staticmethod
    def write_dataset(dataset: ray.data.Dataset, path: str, dataset_uuid: str) -> None:
        """Writes a Ray Dataset to a directory specified by `path`.

        Performs the same logic as ray.data.Dataset.write_json, since our custom JSONDatasource
        copies the _write_block function from Ray's own JSONDatasource verbatim.

        https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.write_json.html

        :param dataset: The Ray Dataset to be written to output files
        :param path: The path to the directory that will store the written files
        :param dataset_uuid: An identifier used to format the names of the output files.
           Output files are named according to the pattern {self._uuid}_{block_idx}.json
           (again, see the docs for write_json for more details).
        """
        dataset.write_datasource(datasource=CustomJSONDatasource(), path=path, dataset_uuid=dataset_uuid)

    @staticmethod
    def _parse_dataset(
        dataset: Dict[str, Any],
        json_parser: JsonParser,
        feature_headers: List[str],
        dataset_name: str,
    ) -> Dict[str, List]:
        """
        Parses the required columns (i.e. feature columns, target output columns, etc.)
        from the JSON dataset and constructs a Dict that is used downstream by
        pyarrow.Table.from_pydict to create the final Ray Dataset.

        :param dataset: The deserialized JSON dataset.
        :param json_parser: The JsonParser used to parse the columns used to construct a new Dataset.
        :param feature_headers: The customer-provided headers, only for feature columns.
        :param dataset_name: The name associated with the Ray Dataset.
        :return: A dict that can directly be consumed by pyarrow.Table.from_pydict
            to create the Ray Dataset. It maps column names to homogeneous lists.
        """

        columns = json_parser.parse_dataset_columns(dataset, dataset_name)
        header_generator_config = HeaderGeneratorConfig.create_from_dataset_columns(columns)
        headers = get_headers(feature_headers, header_generator_config)

        columns_list = [columns.features]
        columns_list += [columns.target_outputs] if columns.target_outputs else []
        columns_list += [columns.inference_outputs] if columns.inference_outputs else []
        columns_list += [columns.categories] if columns.categories else []

        # The kth element of columns_list is a list storing the data corresponding
        # to the kth column (i.e. the kth header)
        columns_list = [
            lst for nested_lst in columns_list for lst in np.transpose(np.array(nested_lst, dtype=object)).tolist()
        ]
        return dict(zip(headers, columns_list))


class CustomJSONDatasource(FileBasedDatasource):
    """Custom JSON datasource class.

    See https://docs.ray.io/en/latest/data/examples/custom-datasource.html#custom-datasources
    for details on creating custom datasources.

    We use this class instead of Ray's own JSONDatasource class because
    Ray's implementation relies on pyarrow._json.read_json, which cannot
    handle JSON files that contain heterogeneous lists
    (lists with elements of different data types).

    Example JSON dataset that pyarrow._json.read_json cannot handle:
    {
       "key": [20, "hello"]
    }

    Attributes:
          config: The JSONDataLoaderConfig used by _read_file when calling
            JsonDataLoader._parse_dataset. Optional, because a config
            is not required for _write_block.
    """

    _FILE_EXTENSION = "json"  # configures the extension for files written by _write_block

    def __init__(self, config: Optional[JsonDataLoaderConfig] = None):
        super().__init__()
        if config:
            self.config = config

    def _read_file(self, f: "pyarrow.NativeFile", path: str, **loader_args) -> pyarrow.Table:  # pragma: no cover
        """
        Reads the JSON dataset file given by `path`, parses the JSON, then returns
        a pyarrow.Table representing the dataset.

        :param f: Unused. Required so that this class conforms to FileBasedDatasource.
        :param path: The path to the dataset JSON file
        :param loader_args: Unused. Required so that this class conforms to FileBasedDatasource.
        """
        with open(path) as file_handle:
            dataset = json.load(file_handle)
            pydict = JsonDataLoader._parse_dataset(
                dataset=dataset,
                json_parser=self.config.json_parser,
                feature_headers=self.config.headers,
                dataset_name=self.config.dataset_name,
            )
            return pyarrow.Table.from_pydict(pydict)

    def _write_block(
        self,
        f: "pyarrow.NativeFile",
        block: BlockAccessor,
        writer_args_fn: Callable[[], Dict[str, Any]] = lambda: {},
        **writer_args,
    ):  # pragma: no cover
        writer_args = _resolve_kwargs(writer_args_fn, **writer_args)
        orient = writer_args.pop("orient", "records")
        lines = writer_args.pop("lines", True)
        block.to_pandas().to_json(f, orient=orient, lines=lines, **writer_args)
