from typing import Any, Dict, List, Optional, Tuple, Union
import jmespath
from jmespath.parser import ParsedResult
from dataclasses import dataclass
from data_loaders.utils.constants import JmespathQueryType
from data_loaders.utils.jmespath_util import search_jmespath


@dataclass(frozen=True)
class JsonParserConfig:
    """Config used by JsonParser. Attributes are a strict subset of those from DataLoaderConfig.

    Attributes:
        features_jmespath: JMESPath query for feature values. These features are used
                           to construct the prompt input to the LLM.
        target_jmespath: JMESPath query for the target output(s).
        inference_jmespath: JMESPath query for the inference output(s).
        category_jmespath: JMESPath query for the categories.
    """

    features_jmespath: str
    target_jmespath: Optional[str] = None
    inference_jmespath: Optional[str] = None
    category_jmespath: Optional[str] = None


@dataclass(frozen=True)
class JsonColumnParseConfig:
    """Data that is shared by various JsonParser methods.

       Note that while there is significant overlap between the data required
       by various methods, not all of them will use every single attribute
       listed below.

    Attributes:
        jmespath_parser: The JMESPath parser, used for parsing features, targets, or inference outputs.
        jmespath_query_type: The general type of the columns that we are parsing, where the possible
                             types are features, target outputs, inference outputs, or categories.
        dataset: The data to be searched, already deserialized into a dict/list.
        dataset_name: The name associated with the dataset being parsed.
        expected_dims: The expected dimensions of the parsed output. Dimensions set to None will not be validated.
    """

    jmespath_parser: ParsedResult
    jmespath_query_type: JmespathQueryType
    dataset: Union[Dict, List]
    dataset_name: str
    expected_dims: Tuple[Optional[int], Optional[int]] = (None, None)


@dataclass(frozen=True)
class DatasetColumns:
    """This class represents the four lists that are returned by parse_dataset_columns.

    These lists are used downstream by the JsonDataLoaderto create a ray.data.Dataset.

    Attributes:
        features: A 2D array, in which the kth entry of the array is a list containing
            the features for the kth sample.
        target_outputs: Same as above but for target outputs.
        inference_outputs: Same as above but for inference outputs.
        categories: Same as above but for categories.
            Note that we currently support only a single category column.
            However, we still make this a 2D array to simplify downstream code
            in JsonDataLoader.
    """

    features: List[List[Any]]
    target_outputs: Optional[List[List[Any]]] = None
    inference_outputs: Optional[List[List[Any]]] = None
    categories: Optional[List[List[Any]]] = None


class JsonParser:
    """Parser for JSON datasets using JMESPath queries for features, target outputs, and inference outputs."""

    def __init__(
        self,
        config: JsonParserConfig,
    ):
        """Initializes the instance using attributes from config.

        :param config: Configures the JMESPath queries for features, target outputs, inference outputs, and categories.
        """
        self._features_parser = jmespath.compile(config.features_jmespath)
        self._target_parser = jmespath.compile(config.target_jmespath) if config.target_jmespath else None
        self._inference_parser = jmespath.compile(config.inference_jmespath) if config.inference_jmespath else None
        self._category_parser = jmespath.compile(config.category_jmespath) if config.category_jmespath else None

    def parse_dataset_columns(self, dataset: Union[Dict, List], dataset_name: str) -> DatasetColumns:
        """Parses a dataset in JSON format, retrieving feature values, and optionally
           target outputs, inference outputs, and categories.

        :param dataset: The dataset's data, already deserialized into a dict/list.
        :param dataset_name: The name of the dataset, to be used for error logging.
        :returns: DatasetColumns object comprised of the columns parsed from the dataset.
        """
        if not isinstance(dataset, (dict, list)):
            raise AlgorithmError(
                f"parse_dataset_columns requires dataset with name `{dataset_name}` to be deserialized into a dict/list."
            )

        features = JsonParser._parse_columns(
            JsonColumnParseConfig(
                jmespath_parser=self._features_parser,
                jmespath_query_type=JmespathQueryType.FEATURES,
                dataset=dataset,
                dataset_name=dataset_name,
                expected_dims=(None, None),
            )
        )
        num_rows = len(features)
        target_outputs = (
            JsonParser._parse_columns(
                JsonColumnParseConfig(
                    jmespath_parser=self._target_parser,
                    jmespath_query_type=JmespathQueryType.TARGET,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    expected_dims=(num_rows, None),
                )
            )
            if self._target_parser
            else None
        )
        inference_outputs = (
            JsonParser._parse_columns(
                JsonColumnParseConfig(
                    jmespath_parser=self._inference_parser,
                    jmespath_query_type=JmespathQueryType.INFERENCE,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    expected_dims=(num_rows, None),
                )
            )
            if self._inference_parser
            else None
        )
        categories = (
            JsonParser._parse_columns(
                JsonColumnParseConfig(
                    jmespath_parser=self._category_parser,
                    jmespath_query_type=JmespathQueryType.CATEGORY,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    expected_dims=(num_rows, 1),
                )
            )
            if self._category_parser
            else None
        )
        return DatasetColumns(features, target_outputs, inference_outputs, categories)

    @staticmethod
    def _parse_columns(config: JsonColumnParseConfig) -> List[Any]:
        """Parses one or more columns from the dataset specified by config.

        :param config: See JsonColumnParseConfig docstring.
        :returns: A list representing the parsed feature/target output/inference output/category column(s).
            This list is always a 2D array (list of lists).
        """
        result = search_jmespath(
            jmespath_parser=config.jmespath_parser,
            jmespath_query_type=config.jmespath_query_type,
            dataset=config.dataset,
            dataset_name=config.dataset_name,
        )
        JsonParser._validate_jmespath_result_dimensions(result, config)

        # If the JMESPath query produces a 1D array,
        # convert it to a 2D array.
        if not isinstance(result[0], list):
            result = [[x] for x in result]
        return result

    @staticmethod
    def _validate_jmespath_result_dimensions(result: List[Any], config: JsonColumnParseConfig) -> None:
        """Validates JMESPath result dimensions are as expected.

        :param result: JMESPath query result to be validated.
        :param config: See JsonColumnParseConfig docstring.
        """
        util.require(
            result and isinstance(result, list),
            f"Expected to find a non-empty list of samples for {config.jmespath_query_type.value} columns using "
            f"JMESPath '{config.jmespath_parser.expression}' on {config.dataset_name}.",
        )

        first = result[0]
        is_2d = isinstance(first, list)

        if is_2d:
            util.require(
                first and all(isinstance(s, list) and len(s) == len(first) for s in result),
                f"Expected a well-formed 2D array using JMESPath '{config.jmespath_parser.expression}' on {config.dataset_name}, where each row "
                f"contains a sample's {config.jmespath_query_type.value} and is non-empty.",
            )
            if config.expected_dims[0]:
                JsonParser._validate_num_rows(result, config)
            if config.expected_dims[1]:
                JsonParser._validate_num_cols(result, config)
        else:
            util.require(
                first is not None and all(not isinstance(s, list) for s in result),
                f"Expected a 1D array using JMESPath '{config.jmespath_parser.expression}' on {config.dataset_name}, where each row "
                f"contains a sample's {config.jmespath_query_type.value} value, but found a nested array in at least one entry.",
            )
            if config.expected_dims[0]:
                JsonParser._validate_num_rows(result, config)

    @staticmethod
    def _validate_num_rows(result: List[Any], config: JsonColumnParseConfig) -> None:
        """Validates result has expected number of rows (i.e. samples).

        :param result: JMESPath query result to be validated.
        :param config: See JsonColumnParseConfig docstring.
        """
        expected_num_rows = config.expected_dims[0]
        util.require(
            len(result) == expected_num_rows,
            f"Expected {expected_num_rows} samples, but JMESPath '{config.jmespath_parser.expression}' for {config.jmespath_query_type.value} on "
            f"{config.dataset_name} gives results for {len(result)} samples.",
        )

    @staticmethod
    def _validate_num_cols(result: List[Any], config: JsonColumnParseConfig) -> None:
        """Validates result has expected number of columns.

        :param result: JMESPath query result to be validated.
        :param config: See JsonColumnParseConfig docstring.
        """
        expected_num_cols = config.expected_dims[1]
        util.require(
            len(result[0]) == expected_num_cols,
            f"Expected {expected_num_cols} features per sample, but JMESPath '{config.jmespath_parser.expression}' on "
            f"{config.dataset_name} gives {len(result[0])} {config.jmespath_query_type.value} for each sample.",
        )
