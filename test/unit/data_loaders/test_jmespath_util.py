import pytest
from unittest.mock import patch
from typing import NamedTuple, Any

from fmeval.data_loaders.jmespath_util import compile_jmespath, search_jmespath
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.constants import ColumnNames


class TestJmespathUtil:
    class CompileJmespathTestCase(NamedTuple):
        function_input: Any
        error_type: Exception
        error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            CompileJmespathTestCase(
                function_input=1,
                error_type=EvalAlgorithmClientError,
                error_message="Unable to compile JMESPath",
            ),
            CompileJmespathTestCase(
                function_input="!!",
                error_type=EvalAlgorithmClientError,
                error_message="Unable to compile JMESPath",
            ),
            CompileJmespathTestCase(
                function_input=None,
                error_type=EvalAlgorithmClientError,
                error_message="Unable to compile JMESPath",
            ),
        ],
    )
    def test_compile_jmespath(self, test_case):
        with pytest.raises(test_case.error_type, match=test_case.error_message):
            compile_jmespath(test_case.function_input)

    @patch("src.fmeval.data_loaders.jmespath_util.logging.Logger.warning")
    def test_search_jmespath_no_result_found(self, mock_logger):
        """
        GIVEN a JMESPath query that finds an empty result when applied to a dataset
        WHEN search_jmespath is called
        THEN search_jmespath returns None and logs contain the appropriate warning
        """
        parser = compile_jmespath("column_c")
        result = search_jmespath(
            jmespath_parser=parser,
            jmespath_query_type=ColumnNames.MODEL_INPUT_COLUMN_NAME.value,
            dataset={"column_a": "hello", "column_b": "world"},
            dataset_name="my_dataset",
        )
        assert result is None
        mock_logger.assert_called_with(
            f"Failed to find {ColumnNames.MODEL_INPUT_COLUMN_NAME.value} columns in dataset `my_dataset` "
            f"using JMESPath query '{parser.expression}'."
        )

    @patch("src.fmeval.data_loaders.jmespath_util.logging.Logger.warning")
    def test_search_jmespath_value_error(self, mock_logger):
        """
        GIVEN a ValueError is raised by the jmespath library function
            (see https://github.com/jmespath/jmespath.py/issues/98)
        WHEN search_jmespath is called
        THEN search_jmespath returns None and logs contain the appropriate warning
        """
        with patch("jmespath.parser.ParsedResult.search", side_effect=ValueError):
            parser = compile_jmespath("column_a")
            result = search_jmespath(
                jmespath_parser=parser,
                jmespath_query_type=ColumnNames.MODEL_INPUT_COLUMN_NAME.value,
                dataset={"column_a": "hello", "column_b": "world"},
                dataset_name="my_dataset",
            )
            assert result is None
            mock_logger.assert_called_with(
                f"Failed to find {ColumnNames.MODEL_INPUT_COLUMN_NAME.value} columns in dataset `my_dataset` "
                f"using JMESPath query '{parser.expression}'."
            )
