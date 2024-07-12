from typing import Any, Dict, List, Union, Optional
from jmespath.parser import ParsedResult
from dataclasses import dataclass
from fmeval.exceptions import EvalAlgorithmInternalError, EvalAlgorithmClientError

from fmeval.data_loaders.jmespath_util import compile_jmespath, search_jmespath
from fmeval.data_loaders.data_config import DataConfig
from fmeval.util import require, assert_condition
from fmeval.constants import (
    DatasetColumns,
    DATASET_COLUMNS,
    COLUMNS_WITH_LISTS,
    DATA_CONFIG_LOCATION_SUFFIX,
    MIME_TYPE_JSON,
    MIME_TYPE_JSONLINES,
)


@dataclass(frozen=True)
class ColumnParseArguments:
    """Data that is shared by various JsonParser methods.

    :param jmespath_parser: The JMESPath parser that parses columns from the dataset
            using JMESPath queries.
    :param column: An enum representing the column that is being parsed.
    :param dataset: The data to be searched, already deserialized into a dict/list.
    :param dataset_mime_type: Either MIME_TYPE_JSON or MIME_TYPE_JSON_LINES.
            Used to determine whether the result parsed by `jmespath_parser`
            is expected to be a list or a single value.
    :param dataset_name: The name associated with the dataset being parsed.
    """

    jmespath_parser: ParsedResult
    column: DatasetColumns
    dataset: Union[Dict[str, Any], List]
    dataset_mime_type: str
    dataset_name: str


class JsonParser:
    """Parser for JSON and JSON Lines datasets using JMESPath queries supplied by the DataConfig class.

    :param _parsers: A dict that maps column names (valid column names are defined by fmeval.constants.DatasetColumns)
        to ParsedResult objects. These ParsedResult objects (from the jmespath library)
        perform the JMESPath searching in the `search_jmespath` util function.
    """

    def __init__(
        self,
        config: DataConfig,
    ):
        """Initializes the instance's `_parsers` dict using attributes from `config` that correspond to JMESPath queries.

        Example:

        config = DataConfig(
            ...,
            model_input_location="my_model_input_jmespath",
            model_output_location="my_model_output_jmespath",
            category_location="my_category_jmespath"
        )

        When we create a JsonParser from this config, we find the attributes ending in DATA_CONFIG_LOCATION_SUFFIX
        (which is "_location" at the time of this writing) that are not None:
        model_input_location, model_output_location, and category_location.
        Note that all optional attributes in DataConfig have default value None.

        The keys we use for `self._parsers` will be "model_input", "model_output", and "category"
        (we strip away DATA_CONFIG_LOCATION_SUFFIX when generating the key).
        We validate that these keys match the `name` attribute of enumerations defined in
        fmeval.constants.DatasetColumns, since downstream code assumes that these names are always used.

        The values corresponding to these keys will be ParsedResult objects that get created by calling
        `compile_jmespath` on "my_model_input_jmespath", "my_model_output_jmespath", and "my_category_jmespath".

        :param config: see DataConfig docstring
        """
        self._parsers = dict()

        for attribute_name, attribute_value in config.__dict__.items():
            if attribute_name.endswith(DATA_CONFIG_LOCATION_SUFFIX) and attribute_value is not None:
                column_name = attribute_name.replace(DATA_CONFIG_LOCATION_SUFFIX, "")
                assert_condition(
                    column_name in DATASET_COLUMNS,
                    f"Found a DataConfig attribute `{attribute_name}` that ends in `{DATA_CONFIG_LOCATION_SUFFIX}` "
                    "but does not correspond to any enumeration defined in fmeval.constants.DatasetColumns.",
                )
                self._parsers[column_name] = compile_jmespath(attribute_value)

    def parse_dataset_columns(
        self, dataset: Union[Dict[str, Any], List], dataset_mime_type: str, dataset_name: str
    ) -> Dict[str, Union[Any, List[Any]]]:
        """Parses a JSON dataset (which could be a single line in a JSON Lines dataset)
           using the parsers stored in self._parsers to extract the desired columns.
           In the case that `dataset` corresponds to a single JSON Lines line, the
           "columns" are scalars instead of lists.

        :param dataset: The dataset's data, already deserialized into a dict/list.
        :param dataset_mime_type: Either MIME_TYPE_JSON or MIME_TYPE_JSONLINES.
        :param dataset_name: The name of the dataset, to be used for error logging.
        :returns: A dict that maps column names to extracted columns. The column names
            are exactly the same as the keys in self._parsers.
        """
        if not isinstance(dataset, (dict, list)):
            raise EvalAlgorithmInternalError(
                f"parse_dataset_columns requires dataset `{dataset_name}` to be deserialized into a dict/list."
            )
        parsed_columns_dict = {
            column_name: JsonParser._parse_column(
                ColumnParseArguments(
                    jmespath_parser=self._parsers[column_name],
                    column=DATASET_COLUMNS[column_name],
                    dataset=dataset,
                    dataset_mime_type=dataset_mime_type,
                    dataset_name=dataset_name,
                )
            )
            for column_name in self._parsers
        }

        filtered_parsed_columns_dict: Dict[str, Union[Any, List[Any]]] = {
            column_name: parsed_columns
            for column_name, parsed_columns in parsed_columns_dict.items()
            if parsed_columns is not None
        }

        if dataset_mime_type == MIME_TYPE_JSON:
            JsonParser._validate_parsed_columns_lengths(filtered_parsed_columns_dict)
        return filtered_parsed_columns_dict

    @staticmethod
    def _parse_column(args: ColumnParseArguments) -> Optional[Union[Any, List[Any]]]:
        """Parses a single column, specified by `args`, from a dataset
           and converts the contents of the column to strings if needed.

        :param args: See ColumnParseArgs docstring.
        :returns: If search_jmespath returns None, this function returns None.
            Otherwise, If `args.dataset_mime_type` is MIME_TYPE_JSON, then the return value
            is a list representing the parsed column. This list is always a 1D array.
            If MIME_TYPE_JSON_LINES, then the dataset being parsed is assumed to be
            a single JSON Lines row, in which case the return value is a single scalar
            value representing the sole value in the "column".
        """
        result = search_jmespath(
            jmespath_parser=args.jmespath_parser,
            jmespath_query_type=args.column.value.name,
            dataset=args.dataset,
            dataset_name=args.dataset_name,
        )
        if result is not None:
            JsonParser._validate_jmespath_result(result, args)
            if args.column.value.should_cast:
                result = JsonParser._cast_to_string(result, args)
        return result

    @staticmethod
    def _validate_jmespath_result(result: Union[Any, List[Any], List[List[Any]]], args: ColumnParseArguments) -> None:
        """Validates that the JMESPath result is as expected.

        For dataset columns in COLUMNS_WITH_LISTS, if `args.dataset_mime_type` is MIME_TYPE_JSON, then `result` is
        expected to be a 2D array. If MIME_TYPE_JSON_LINES, then `result` is expected to be a 1D array (list).

        For all other dataset columns, if `args.dataset_mime_type` is MIME_TYPE_JSON, then `result` is expected to be
        a 1D array (list). If MIME_TYPE_JSON_LINES, then `result` is expected to be a single scalar value.

        :param result: JMESPath query result to be validated.
        :param args: See ColumnParseArguments docstring.
        """
        if args.dataset_mime_type == MIME_TYPE_JSON:
            require(
                result and isinstance(result, list),
                f"Expected to find a non-empty list of samples for the {args.column.value.name} column using "
                f"JMESPath '{args.jmespath_parser.expression}' on {args.dataset_name}.",
            )
            require(
                all(x is not None for x in result),  # explicitly using "is not None" since values like 0 are false-y
                f"Expected an array of non-null values using JMESPath '{args.jmespath_parser.expression}' for "
                f"the {args.column.value.name} column of dataset `{args.dataset_name}`, but found at least "
                "one value that is None.",
            )
            if args.column.value.name in COLUMNS_WITH_LISTS:
                require(
                    all(isinstance(x, list) for x in result),
                    f"Expected a 2D array using JMESPath '{args.jmespath_parser.expression}' on dataset "
                    f"`{args.dataset_name}` but found at least one non-list object.",
                )
            else:
                require(
                    all(not isinstance(x, list) for x in result),
                    f"Expected a 1D array using JMESPath '{args.jmespath_parser.expression}' on dataset "
                    f"`{args.dataset_name}`, where each element of the array is a sample's {args.column.value.name}, "
                    f"but found at least one nested array.",
                )
        elif args.dataset_mime_type == MIME_TYPE_JSONLINES:
            require(
                result is not None,
                f"Found no values using {args.column.value.name} JMESPath '{args.jmespath_parser.expression}' "
                f"on dataset `{args.dataset_name}`.",
            )
            if args.column.value.name in COLUMNS_WITH_LISTS:
                require(
                    isinstance(result, list),
                    f"Expected to find a List using JMESPath '{args.jmespath_parser.expression}' on a dataset line in "
                    f"`{args.dataset_name}`, but found a non-list object instead.",
                )
            else:
                require(
                    not isinstance(result, list),
                    f"Expected to find a single value using {args.column.value.name} JMESPath "
                    f"'{args.jmespath_parser.expression}' on a dataset line in "
                    f"dataset `{args.dataset_name}`, but found a list instead.",
                )
        else:  # pragma: no cover
            raise EvalAlgorithmInternalError(
                f"args.dataset_mime_type is {args.dataset_mime_type}, but only JSON " "and JSON Lines are supported."
            )

    @staticmethod
    def _validate_parsed_columns_lengths(parsed_columns_dict: Dict[str, List[Any]]):
        """
        Validates that every column (represented by a list) in `parsed_columns_dict`
        has the same number of elements.

        :param parsed_columns_dict: The dict returned by the parse_dataset_columns method assuming
            that this validation succeeds.
        :raises: EvalAlgorithmClientError if validation fails
        """
        num_samples = len(parsed_columns_dict[next(iter(parsed_columns_dict))])
        require(
            all(len(column) == num_samples for column in parsed_columns_dict.values()),
            "Expected the number of samples that were parsed by provided JMESPath queries"
            "to be the same for the result of every JMESPath query, but not all queries"
            "resulted in the same number of samples.",
        )

    @staticmethod
    def _cast_to_string(
        result: Union[Any, List[Any], List[List[Any]]], args: ColumnParseArguments
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Casts the contents of `result` to string(s), raising an error if casting fails.
        It is extremely unlikely that the str() operation should fail; this basically
        only happens if the object has explicitly overwritten the __str__ method to raise
        an exception.

        For dataset columns in COLUMNS_WITH_LISTS, if `args.dataset_mime_type` is MIME_TYPE_JSON, then `result` is
        expected to be a 2D array. If MIME_TYPE_JSON_LINES, then `result` is expected to be a 1D array (list).
        For all other dataset columns, if `args.dataset_mime_type` is MIME_TYPE_JSON, then `result` is expected to be
        a 1D array (list). If MIME_TYPE_JSON_LINES, then `result` is expected to be a single scalar value.

        :param result: JMESPath query result to be casted.
        :param args: See ColumnParseArguments docstring.
        :returns: `result` casted to a string or list of strings.
        """
        try:
            if args.dataset_mime_type == MIME_TYPE_JSON:
                if args.column.value.name in COLUMNS_WITH_LISTS:
                    return [[str(x) for x in sample] for sample in result]
                else:
                    return [str(x) for x in result]
            elif args.dataset_mime_type == MIME_TYPE_JSONLINES:
                return [str(x) for x in result] if args.column.value.name in COLUMNS_WITH_LISTS else str(result)
            else:
                raise EvalAlgorithmInternalError(  # pragma: no cover
                    f"args.dataset_mime_type is {args.dataset_mime_type}, but only JSON and JSON Lines are supported."
                )
        except Exception:
            raise EvalAlgorithmClientError(
                "Failed to cast object to string in json_parser._cast_to_string. "
                f"Please inspect dataset {args.dataset_name} for columns containing "
                "objects that cannot be converted to strings. The column that failed "
                f"is {args.column.value.name} which was extracted using the JMESPath expression "
                f"{args.jmespath_parser.expression}."
            )
