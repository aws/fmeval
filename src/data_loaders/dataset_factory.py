import ray.data
from typing import Union, Type
from orchestrator.configs.analysis_config import DatasetConfig, DatasetMimeType
from data_loaders.data_resources.data_resource_factory import (
    DataResourceFactory,
)
from data_loaders.data_resources.data_resources import DataFile, DataResource
from data_loaders.json_dataloader.json_dataset_reader import (
    JsonDatasetReader,
    JsonDatasetReaderConfig,
)
from data_loaders.json_dataloader.json_parser import (
    JsonParser,
    JsonParserConfig,
)
from data_loaders.jsonlines_dataloader.jsonlines_dataset_reader import (
    JsonLinesDatasetReader,
    JsonLinesDatasetReaderConfig,
)
from data_loaders.jsonlines_dataloader.jsonlines_parser import (
    JsonLinesParser,
    JsonLinesParserConfig,
)
from infra.utils.sm_exceptions import AlgorithmError


class DataLoader:
    @staticmethod
    def get_dataset(config: DatasetConfig) -> ray.data.Dataset:
        parser = DataLoader._get_parser(config)
        data_resource = DataResourceFactory.get_data_resource(config.dataset_uri)
        dataset_reader_config = DataLoader._get_dataset_reader_config(parser, data_resource, config)
        dataset_reader = DataLoader._get_dataset_reader(config)
        return dataset_reader.read_dataset(dataset_reader_config)  # type: ignore

    @staticmethod
    def _get_parser(config: DatasetConfig) -> Union[JsonParser, JsonLinesParser]:
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
    def _get_dataset_reader_config(
        parser: Union[JsonParser, JsonLinesParser], data_resource: DataResource, config: DatasetConfig
    ) -> Union[JsonDatasetReaderConfig, JsonLinesDatasetReaderConfig]:
        if config.mime_type == DatasetMimeType.JSON:
            if not isinstance(parser, JsonParser):
                raise AlgorithmError(f"parser should be a JsonParser but is {type(parser)}.")
            if not isinstance(data_resource, DataFile):
                raise AlgorithmError(
                    f"JSON datasets must be stored in a single file. "
                    f"Provided dataset has type {type(data_resource)}."
                )
            return JsonDatasetReaderConfig(
                json_parser=parser,
                data_file=data_resource,
                dataset_name=config.dataset_name,
                headers=config.headers,
            )
        elif config.mime_type == DatasetMimeType.JSONLINES:
            if not isinstance(parser, JsonLinesParser):
                raise AlgorithmError(f"parser should be a JsonLinesParser but is {type(parser)}.")
            if not isinstance(data_resource, DataFile):
                raise AlgorithmError(
                    f"JSONLines datasets must be stored in a single file. "
                    f"Provided dataset has type {type(data_resource)}."
                )
            return JsonLinesDatasetReaderConfig(
                jsonlines_parser=parser,
                data_file=data_resource,
                dataset_name=config.dataset_name,
                headers=config.headers,
            )
        else:  # pragma: no cover
            raise AlgorithmError(
                "Dataset MIME types other than JSON and JSON Lines are not supported. "
                f"MIME type detected from config is {config.mime_type}"
            )

    @staticmethod
    def _get_dataset_reader(config: DatasetConfig) -> Type[JsonDatasetReader | JsonLinesDatasetReader]:
        if config.mime_type == DatasetMimeType.JSON:
            return JsonDatasetReader
        elif config.mime_type == DatasetMimeType.JSONLINES:
            return JsonLinesDatasetReader
        else:  # pragma: no cover
            raise AlgorithmError(
                "Dataset MIME types other than JSON and JSON Lines are not supported. "
                f"MIME type detected from config is {config.mime_type}"
            )
