from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
from orchestrator.utils import util
from data_loaders.json_dataloader.json_parser import DatasetColumns
from data_loaders.jsonlines_dataloader.jsonlines_parser import DatasetSample

# Column name for category column when category JMESPath is provided.
CATEGORY_COLUMN_NAME = "category"


class HeaderType(Enum):
    """
    Header types used to generate artificial headers
    """

    FEATURE = "feature"
    TARGET_OUTPUT = "target_output"
    INFERENCE_OUTPUT = "inference_output"


@dataclass(frozen=True)
class HeaderGeneratorConfig:
    """
    A dataclass to define config attributes used for header generation.

    Attributes:
        n_target_output: number of target outputs
        n_inference_output: number of inference outputs
        has_category_column: whether there exists an additional category column
    """

    n_target_output: Optional[int] = None
    n_inference_output: Optional[int] = None
    has_category_column: Optional[bool] = False

    @staticmethod
    def create_from_dataset_columns(columns: DatasetColumns) -> "HeaderGeneratorConfig":
        n_target_output = 0
        if columns.target_outputs:
            if isinstance(columns.target_outputs[0], list):
                n_target_output = len(columns.target_outputs[0])
            else:
                n_target_output = 1

        n_inference_output = 0
        if columns.inference_outputs:
            if isinstance(columns.inference_outputs[0], list):
                n_inference_output = len(columns.inference_outputs[0])
            else:
                n_inference_output = 1

        has_sub_category_column = columns.categories is not None

        return HeaderGeneratorConfig(n_target_output, n_inference_output, has_sub_category_column)

    @staticmethod
    def create_from_jsonlines(line: DatasetSample, headers: List[str]) -> "HeaderGeneratorConfig":  # pragma: no cover
        """
        :param line: DatasetSample object for one dataset line
        :param headers: list of customer provided dataset headers for features
        :return:  HeaderGeneratorConfig object
        """
        n_features = len(line.features) if isinstance(line.features, list) else 1
        util.require(
            n_features == len(headers),
            f"Expected {len(headers)} features based on the number of headers provided, got {n_features} instead. ",
        )
        n_target_output = (
            len(line.target_outputs)
            if (line.target_outputs and isinstance(line.target_outputs, list))
            else 1
            if line.target_outputs is not None
            else None
        )
        n_inference_output = (
            len(line.inference_outputs)
            if (line.inference_outputs and isinstance(line.inference_outputs, list))
            else 1
            if line.inference_outputs is not None
            else None
        )
        has_category_column = True if line.category is not None else False
        header_config = HeaderGeneratorConfig(
            n_target_output=n_target_output,
            n_inference_output=n_inference_output,
            has_category_column=has_category_column,
        )
        return header_config


class HeaderGenerator:
    """
    This class generates artificial headers for the target_output, inference_output or category columns of a dataset
    """

    def generate_headers(self, config: HeaderGeneratorConfig) -> List[str]:
        """
        :param config: header config object
        :return: list of artificial dataset headers
        """
        headers = []
        if config.n_target_output:
            headers.extend(self._generate_headers_by_type(HeaderType.TARGET_OUTPUT, config.n_target_output))
        if config.n_inference_output:
            headers.extend(self._generate_headers_by_type(HeaderType.INFERENCE_OUTPUT, config.n_inference_output))
        if config.has_category_column:
            headers.append(CATEGORY_COLUMN_NAME)
        return headers

    @staticmethod
    def _generate_headers_by_type(header_type: HeaderType, n_headers: int) -> List[str]:
        """
        :param header_type: one of features, target_output or inference_output
        :param n_headers: number of headers to generate
        :return: list of headers for a given header type
        """
        return [f"{header_type.value}_{i}" for i in range(n_headers)]
