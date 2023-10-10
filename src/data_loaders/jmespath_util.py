from typing import Any, List, Dict, Union

import jmespath
from jmespath.exceptions import JMESPathError
from jmespath.parser import ParsedResult
from exceptions import EvalAlgorithmClientError


def compile_jmespath(jmespath_expression: str):
    """
    Compiles a JMESPath expression to be used for JSON data.
    """
    try:
        return jmespath.compile(jmespath_expression)
    except (TypeError, JMESPathError) as e:
        raise EvalAlgorithmClientError(f"Unable to compile JMESPath {jmespath_expression}") from e


def search_jmespath(
    jmespath_parser: ParsedResult,
    jmespath_query_type: str,
    dataset: Union[Dict, List],
    dataset_name: str,
) -> List[Any]:
    """Searches a dataset using a JMESPath query.

    :param jmespath_parser: The JMESPath parser, used for parsing model inputs, model outputs,
        target outputs, or categories.
    :param jmespath_query_type: Used for error logging. One of the following:
            constants.MODEL_INPUT_COLUMN_NAME, constants.MODEL_OUTPUT_COLUMN_NAME
            constants.TARGET_OUTPUT_COLUMN_NAME, constants.CATEGORY_COLUMN_NAME
    :param dataset: The data to be searched, already deserialized into a dict/list.
    :param dataset_name: A name associated with the dataset being parsed for logging purposes.
    :returns: The result of executing the JMESPath query on the dataset.
    """
    try:
        result = jmespath_parser.search(dataset)
        if result is None:
            raise EvalAlgorithmClientError(
                f"Failed to find {jmespath_query_type} columns in dataset `{dataset_name}` using JMESPath "
                f"query '{jmespath_parser.expression}'."
            )
        return result
    except ValueError as e:
        raise EvalAlgorithmClientError(
            f"Failed to find {jmespath_query_type} columns in dataset `{dataset_name}` using JMESPath query "
            f"'{jmespath_parser.expression}'."
        ) from e
