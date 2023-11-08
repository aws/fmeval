import pytest
from unittest.mock import patch
from typing import NamedTuple, Any

from fmeval.data_loaders.jmespath_util import compile_jmespath, search_jmespath
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.constants import MODEL_INPUT_COLUMN_NAME


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

    def test_search_jmespath_no_result_found(self):
        """
        GIVEN a JMESPath query that finds an empty result when applied to a dataset
        WHEN search_jmespath is called
        THEN search_jmespath raises an exception
        """
        with pytest.raises(
            EvalAlgorithmClientError,
            match="Failed to find model_input columns in dataset `my_dataset` using JMESPath query 'column_c'.",
        ):
            parser = compile_jmespath("column_c")
            search_jmespath(
                jmespath_parser=parser,
                jmespath_query_type=MODEL_INPUT_COLUMN_NAME,
                dataset={"column_a": "hello", "column_b": "world"},
                dataset_name="my_dataset",
            )

    def test_search_jmespath_value_error(self):
        """
        GIVEN a ValueError is raised by the jmespath library function
            (see https://github.com/jmespath/jmespath.py/issues/98)
        WHEN search_jmespath is called
        THEN search_jmespath raises an exception
        """
        with pytest.raises(
            EvalAlgorithmClientError,
            match="Failed to find model_input columns in dataset `my_dataset` using JMESPath query 'column_a'.",
        ), patch("jmespath.parser.ParsedResult.search", side_effect=ValueError):
            parser = compile_jmespath("column_a")
            search_jmespath(
                jmespath_parser=parser,
                jmespath_query_type=MODEL_INPUT_COLUMN_NAME,
                dataset={"column_a": "hello", "column_b": "world"},
                dataset_name="my_dataset",
            )
