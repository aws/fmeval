from typing import Any, List, Dict, Union

import jmespath
from jmespath.exceptions import JMESPathError
from jmespath.parser import ParsedResult
from infra.utils.sm_exceptions import CustomerError
from data_loaders.utils.constants import JmespathQueryType


def search_jmespath(
    jmespath_parser: ParsedResult,
    jmespath_query_type: JmespathQueryType,
    dataset: Union[Dict, List],
    dataset_name: str,
) -> List[Any]:
    """Searches a dataset using a JMESPath query.

    :param jmespath_parser: The JMESPath parser, used for parsing features, targets, or inference outputs.
    :param jmespath_query_type: The general "category" of the columns that we are parsing, where the available
                                categories are features, target outputs, or inference outputs.
    :param dataset: The data to be searched, already deserialized into a dict/list.
    :param dataset_name: A name associated with the dataset being parsed for logging purposes.
    :returns: The result of executing the JMESPath query on the dataset.
    """
    try:
        result = jmespath_parser.search(dataset)
        if not result:
            raise CustomerError(
                f"Failed to find {jmespath_query_type.value} columns in {dataset_name} using JMESPath "
                f"query '{jmespath_parser.expression}'"
            )
        return result
    except ValueError as e:
        raise CustomerError(
            f"Failed to find {jmespath_query_type.value} columns in {dataset_name} using JMESPath query "
            f"'{jmespath_parser.expression}'"
        ) from e


def compile_jmespath(jmespath_expression: str):
    """
    Compiles a JMESPath expression to be used for JSON data.
    """
    try:
        return jmespath.compile(jmespath_expression)
    except (TypeError, JMESPathError) as e:
        raise CustomerError(f"Unable to compile JMESPath {jmespath_expression}") from e
