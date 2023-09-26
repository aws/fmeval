from typing import List, Dict, Optional, Union, Any
import jmespath
from jmespath.parser import ParsedResult
from dataclasses import dataclass
from data_loaders.utils.constants import JmespathQueryType
from data_loaders.utils.jmespath_util import search_jmespath
from orchestrator.utils import util


@dataclass(frozen=True)
class JsonLinesParserConfig:
    """
    Config used by JsonLinesParser.

    Attributes:
        features_jmespath: JMESPath query for feature values. These features are used to construct the prompt input to
                           the model.
        target_jmespath: JMESPath query for the target output(s).
        inference_jmespath: JMESPath query for the inference output(s).
        category_jmespath: JMESPath query for the category output.
    """

    features_jmespath: str
    target_jmespath: Optional[str] = None
    inference_jmespath: Optional[str] = None
    category_jmespath: Optional[str] = None


@dataclass(frozen=True)
class JsonLinesCategoryParseConfig:
    """
    Data that is shared by various JsonLinesParser methods.

    Attributes:
        jmespath_parser: The JMESPath parser, used for parsing features, target outputs, or inference outputs.
        jmespath_query_type: The type of the columns being parsed, where the available types are features, target
                             outputs, or inference outputs.
        dataset_line: The dataset line to be searched, already deserialized into a dict/list.
        dataset_name: A name associated with the dataset being parsed for logging purposes.
    """

    jmespath_parser: ParsedResult
    jmespath_query_type: JmespathQueryType
    dataset_line: Union[Dict, List]
    dataset_name: str = "dataset"


@dataclass(frozen=True)
class DatasetSample:
    """
    This class contains the values returned by parse_dataset_line.

    Attributes:
        features: A list of features in a 1D array.
        target_outputs: Same as above but for target outputs.
        inference_outputs: Same as above but for inference outputs.
        category: Same as above but for category.
    """

    features: Union[Any, List[Any]]
    target_outputs: Optional[Union[Any, List[Any]]] = None
    inference_outputs: Optional[Union[Any, List[Any]]] = None
    category: Optional[Union[Any, List[Any]]] = None


class JsonLinesParser:
    """
    Parser for Json Lines dataset lines using JMESPath queries for features, target outputs, inference outputs and
    category.
    """

    def __init__(self, config: JsonLinesParserConfig):
        """
        Initializes a Json Lines dataset parser for parsing JMESPaths.

        :param config: see JsonLinesParserConfig docstring for details.
        """
        self._features_parser = jmespath.compile(config.features_jmespath)
        self._target_parser = jmespath.compile(config.target_jmespath) if config.target_jmespath else None
        self._inference_parser = jmespath.compile(config.inference_jmespath) if config.inference_jmespath else None
        self._category_parser = jmespath.compile(config.category_jmespath) if config.category_jmespath else None

    def parse_dataset_line(self, dataset_line: Union[Dict, List], dataset_name: str) -> DatasetSample:
        """
        Parses a single record of a JSON Lines dataset, retrieving feature values, and optionally target outputs,
        inference outputs and category.

        :param dataset_line: A dataset line, already deserialized into a dict/list.
        :param dataset_name: The name of the dataset, to be used for error logging.
        :returns: DatasetSample object comprised of the parsed feature/target/inference/category column(s).
        """
        features = JsonLinesParser._parse_columns(
            JsonLinesCategoryParseConfig(
                jmespath_parser=self._features_parser,
                jmespath_query_type=JmespathQueryType.FEATURES,
                dataset_line=dataset_line,
                dataset_name=dataset_name,
            )
        )
        target = (
            JsonLinesParser._parse_columns(
                JsonLinesCategoryParseConfig(
                    jmespath_parser=self._target_parser,
                    jmespath_query_type=JmespathQueryType.TARGET,
                    dataset_line=dataset_line,
                    dataset_name=dataset_name,
                )
            )
            if self._target_parser
            else None
        )
        inference = (
            JsonLinesParser._parse_columns(
                JsonLinesCategoryParseConfig(
                    jmespath_parser=self._inference_parser,
                    jmespath_query_type=JmespathQueryType.INFERENCE,
                    dataset_line=dataset_line,
                    dataset_name=dataset_name,
                )
            )
            if self._inference_parser
            else None
        )
        category = (
            JsonLinesParser._parse_columns(
                JsonLinesCategoryParseConfig(
                    jmespath_parser=self._category_parser,
                    jmespath_query_type=JmespathQueryType.CATEGORY,
                    dataset_line=dataset_line,
                    dataset_name=dataset_name,
                )
            )
            if self._category_parser
            else None
        )
        return DatasetSample(features, target, inference, category)

    @staticmethod
    def _parse_columns(config: JsonLinesCategoryParseConfig) -> List[Any]:
        """
        Parses values for one of feature/target_output/inference_output/category

        :param config: The JsonLinesRowParseConfig object containing the JMESPath query.
        :returns: The parsed feature/target_output/inference_output/category values.
        """
        result = search_jmespath(
            jmespath_parser=config.jmespath_parser,
            jmespath_query_type=config.jmespath_query_type,
            dataset=config.dataset_line,
            dataset_name=config.dataset_name,
        )
        JsonLinesParser._validate_result_num_dimensions(result, config)
        return result

    @staticmethod
    def _validate_result_num_dimensions(result: Union[Any, List[Any]], config: JsonLinesCategoryParseConfig) -> None:
        """
        Validates that the JMESPath query result is a single value or a one dimensional list for features,
        target_outputs and inference_outputs, and a single value or list of length 1 for category.

        :param result: The result of a JMESPath query.
        :param config: The JsonLinesRowParseConfig object containing the JMESPath query.
        """
        util.require(
            result is not None,
            f"Found no values using {config.jmespath_query_type.value} JMESPath '{config.jmespath_parser.expression}' "
            f"on {config.dataset_name}.",
        )
        if config.jmespath_query_type == JmespathQueryType.CATEGORY and isinstance(result, list):
            util.require(
                len(result) == 1,
                f"Expected a single value for each {config.jmespath_query_type.value} using JMESPath "
                f"'{config.jmespath_parser.expression}' on {config.dataset_name}, but found an array with more than one value in at "
                f"least one record.",
            )
        elif isinstance(result, list):
            util.require(
                all(not isinstance(i, list) for i in result),
                f"Expected a 1D array for each {config.jmespath_query_type.value} using JMESPath "
                f"'{config.jmespath_parser.expression}' on {config.dataset_name}, but found a nested array in at least "
                f"one record.",
            )
