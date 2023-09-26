import ray.data
from typing import Union, Type
from ..data_loaders.data_sources import DataFile, DataSource
from ..data_loaders.util import DataConfig, get_data_source

from ..data_loaders.json_dataloader.json_data_loader import (
    JsonDataLoader,
    JsonDataLoaderConfig,
)
from ..data_loaders.json_dataloader.json_parser import JsonParser, JsonParserConfig

from ..data_loaders.jsonlines_dataloader.jsonlines_data_loader import (
    JsonLinesDataLoader,
    JsonLinesDataLoaderConfig,
)
from ..data_loaders.jsonlines_dataloader.jsonlines_parser import JsonLinesParser, JsonLinesParserConfig
from ..exceptions import AlgorithmError

class DataLoader:
    @staticmethod
    def get_dataset(config: DataConfig) -> ray.data.Dataset:
        parser = DataLoader._get_parser(config)
        data_source = get_data_source(config.dataset_uri)
        data_loader_config = DataLoader._get_data_loader_config(parser, data_source, config)
        data_loader = DataLoader._get_data_loader(config)
        return data_loader.read_dataset(data_loader_config)  # type: ignore

    @staticmethod
    def _get_parser(config: DataConfig) -> Union[JsonParser, JsonLinesParser]:
        if config.mime_type == DatasetMimeType.JSON:
            return JsonParser(
                JsonParserConfig(
                    features_jmespath=config.features,
                    target_jmespath=config.target_output,
                    inference_jmespath=config.inference_output,
                    category_jmespath=config.category,
                )
            )
        elif config.mime_type == DatasetMimeType.JSONLINES:
            return JsonLinesParser(
                JsonLinesParserConfig(
                    features_jmespath=config.features,
                    target_jmespath=config.target_output,
                    inference_jmespath=config.inference_output,
                    category_jmespath=config.category,
                )
            )
        else:  # pragma: no cover
            raise AlgorithmError(
                "Dataset MIME types other than JSON and JSON Lines are not supported. "
                f"MIME type detected from config is {config.mime_type}"
            )

    @staticmethod
    def _get_data_loader_config(
        parser: Union[JsonParser, JsonLinesParser], data_ource: DataSource, config: DataConfig
    ) -> Union[JsonDataLoaderConfig, JsonLinesDataLoaderConfig]:
        if config.mime_type == DatasetMimeType.JSON:
            if not isinstance(parser, JsonParser):
                raise AlgorithmError(f"parser should be a JsonParser but is {type(parser)}.")
            if not isinstance(data_ource, DataFile):
                raise AlgorithmError(
                    f"JSON datasets must be stored in a single file. "
                    f"Provided dataset has type {type(data_source)}."
                )
            return JsonDataLoaderConfig(
                json_parser=parser,
                data_file=data_source,
                dataset_name=config.dataset_name,
                headers=config.headers,
            )
        elif config.mime_type == DatasetMimeType.JSONLINES:
            if not isinstance(parser, JsonLinesParser):
                raise AlgorithmError(f"parser should be a JsonLinesParser but is {type(parser)}.")
            if not isinstance(data_source, DataFile):
                raise AlgorithmError(
                    f"JSONLines datasets must be stored in a single file. "
                    f"Provided dataset has type {type(data_source)}."
                )
            return JsonLinesDataLoaderConfig(
                jsonlines_parser=parser,
                data_file=data_source,
                dataset_name=config.dataset_name,
                headers=config.headers,
            )
        else:  # pragma: no cover
            raise AlgorithmError(
                "Dataset MIME types other than JSON and JSON Lines are not supported. "
                f"MIME type detected from config is {config.mime_type}"
            )

    @staticmethod
    def _get_data_loader(config: DatasetConfig) -> Type[JsonDataLoader | JsonLinesDataLoader]:
        if config.mime_type == DatasetMimeType.JSON:
            return JsonDataLoader
        elif config.mime_type == DatasetMimeType.JSONLINES:
            return JsonLinesDataLoader
        else:  # pragma: no cover
            raise AlgorithmError(
                "Dataset MIME types other than JSON and JSON Lines are not supported. "
                f"MIME type detected from config is {config.mime_type}"
            )
