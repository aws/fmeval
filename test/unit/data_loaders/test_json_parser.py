from unittest.mock import patch, call, Mock
import pytest
from typing import Any, List, NamedTuple, Union, Dict
from fmeval.data_loaders.json_parser import JsonParser, ColumnParseArguments
from fmeval.exceptions import EvalAlgorithmClientError, EvalAlgorithmInternalError
from fmeval.data_loaders.data_config import DataConfig
from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON,
    MIME_TYPE_JSONLINES,
)


class TestJsonParser:
    class TestCaseInit(NamedTuple):
        config: DataConfig
        expected_call_args: List[str]
        expected_keys: List[str]

    @pytest.mark.parametrize(
        "config, expected_call_args, expected_keys",
        [
            TestCaseInit(
                config=DataConfig(
                    dataset_name="dataset",
                    dataset_uri="uri",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="model_input_jmespath",
                ),
                expected_call_args=["model_input_jmespath"],
                expected_keys=[DatasetColumns.MODEL_INPUT.value.name],
            ),
            TestCaseInit(
                config=DataConfig(
                    dataset_name="dataset",
                    dataset_uri="uri",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="model_input_jmespath",
                    model_output_location="model_output_jmespath",
                    target_output_location="target_output_jmespath",
                    category_location="category_jmespath",
                ),
                expected_call_args=[
                    "model_input_jmespath",
                    "model_output_jmespath",
                    "target_output_jmespath",
                    "category_jmespath",
                ],
                expected_keys=[
                    DatasetColumns.MODEL_INPUT.value.name,
                    DatasetColumns.MODEL_OUTPUT.value.name,
                    DatasetColumns.TARGET_OUTPUT.value.name,
                    DatasetColumns.CATEGORY.value.name,
                ],
            ),
        ],
    )
    def test_init_success(self, config, expected_call_args, expected_keys):
        """
        GIVEN a DataConfig with various attributes initialized
        WHEN a JsonParser is created
        THEN jmespath.compile is called for every single JMESPath query
            supplied by the DataConfig's *_location attributes, and the keys
            in the created JsonParser's `_parsers` dict match what is expected.
        """
        with patch("jmespath.compile") as mock_compile:
            json_parser = JsonParser(config)
            mock_compile.assert_has_calls([call(arg) for arg in expected_call_args])
            actual_keys = list(json_parser._parsers.keys())
            actual_keys.sort()
            expected_keys.sort()
            assert actual_keys == expected_keys

    def test_init_failure(self):
        """
        GIVEN a DataConfig that contains an attribute with a name ending in '_location'
            that doesn't have a corresponding COLUMN_NAME constant
        WHEN a JsonParser is created using this DataConfig
        THEN JsonParser's __init__ method raises an EvalAlgorithmError
        """
        data_config = Mock()
        data_config.fake_column_type_location = "jmespath_for_fake_column_type"
        with pytest.raises(
            EvalAlgorithmInternalError, match="Found a DataConfig attribute `fake_column_type_location`"
        ):
            JsonParser(data_config)

    class TestCaseParseColumnFailure(NamedTuple):
        result: List[Any]
        error_message: str

    @pytest.mark.parametrize(
        "result, error_message",
        [
            TestCaseParseColumnFailure(
                result="not a list",
                error_message="Expected to find a non-empty list of samples",
            ),
            TestCaseParseColumnFailure(
                result=[1, 2, None],
                error_message="Expected an array of non-null values",
            ),
            TestCaseParseColumnFailure(
                result=[1, 2, [3], 4],
                error_message="Expected a 1D array",
            ),
        ],
    )
    def test_validation_failure_json(self, result, error_message):
        """
        GIVEN a malformed `result` argument (obtained from a JSON dataset)
        WHEN _validate_jmespath_result is called
        THEN _validate_jmespath_result raises an EvalAlgorithmClientError
            with the correct error message
        """
        with pytest.raises(EvalAlgorithmClientError, match=error_message):
            args = ColumnParseArguments(
                jmespath_parser=Mock(),
                column=Mock(),
                dataset={},
                dataset_mime_type=MIME_TYPE_JSON,
                dataset_name="dataset",
            )
            JsonParser._validate_jmespath_result(result, args)

    @pytest.mark.parametrize(
        "result, error_message",
        [
            TestCaseParseColumnFailure(
                result=None,
                error_message="Found no values using",
            ),
            TestCaseParseColumnFailure(
                result=[1, 2, 3],
                error_message="Expected to find a single value",
            ),
        ],
    )
    def test_validation_failure_jsonlines(self, result, error_message):
        """
        GIVEN a malformed `result` argument (obtained from a JSON Lines dataset line)
        WHEN _validate_jmespath_result is called
        THEN _validate_jmespath_result raises an EvalAlgorithmClientError
            with the correct error message
        """
        with pytest.raises(EvalAlgorithmClientError, match=error_message):
            args = ColumnParseArguments(
                jmespath_parser=Mock(),
                column=Mock(),
                dataset={},
                dataset_mime_type=MIME_TYPE_JSONLINES,
                dataset_name="dataset",
            )
            JsonParser._validate_jmespath_result(result, args)

    class TestCaseJsonParseDatasetColumns(NamedTuple):
        config: DataConfig
        dataset: Union[Dict, List]

    @pytest.mark.parametrize(
        "config, dataset",
        [
            # dataset is a dict
            TestCaseJsonParseDatasetColumns(
                config=DataConfig(
                    dataset_name="dataset",
                    dataset_uri="uri",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="samples[*].name",
                    model_output_location="model_output.*",
                    target_output_location="targets_outer.targets_inner[*].sentiment",
                    category_location="category",
                    # this JMESPath query will fail to find any results, and should effectively get ignored
                    sent_more_input_location="invalid_jmespath_query",
                ),
                dataset={
                    "samples": [{"name": "A", "age": 1}, {"name": "B", "age": 2}],
                    "targets_outer": {
                        "targets_inner": [
                            {"sentiment": "negative"},
                            {"sentiment": "positive"},
                        ],
                    },
                    "model_output": {"sample_1": "positive", "sample_2": "negative"},
                    "category": ["category_0", "category_1"],
                },
            ),
            # dataset is a list
            TestCaseJsonParseDatasetColumns(
                config=DataConfig(
                    dataset_name="dataset",
                    dataset_uri="uri",
                    dataset_mime_type="application/json",
                    model_input_location="[*].model_input_col",
                    model_output_location="[*].model_output_col",
                    target_output_location="[*].target_output_col",
                    category_location="[*].category_col",
                    # this JMESPath query will fail to find any results, and should effectively get ignored
                    sent_more_input_location="invalid_jmespath_query",
                ),
                dataset=[
                    {
                        "model_input_col": "A",
                        "model_output_col": "positive",
                        "target_output_col": "negative",
                        "category_col": "category_0",
                    },
                    {
                        "model_input_col": "B",
                        "model_output_col": "negative",
                        "target_output_col": "positive",
                        "category_col": "category_1",
                    },
                ],
            ),
        ],
    )
    @patch("src.fmeval.data_loaders.jmespath_util.logging.Logger.warning")
    def test_json_parse_dataset_columns_success_json(self, mock_logger, config, dataset):
        """
        GIVEN valid JMESPath queries that extract model inputs, model outputs,
            target outputs, and categories, and a JSON dataset that is represented
            by either a dict or list
        WHEN parse_dataset_columns is called
        THEN parse_dataset_columns returns correct results
        """
        expected_model_inputs = ["A", "B"]
        expected_model_outputs = ["positive", "negative"]
        expected_target_outputs = ["negative", "positive"]
        expected_categories = ["category_0", "category_1"]

        parser = JsonParser(config)
        cols = parser.parse_dataset_columns(dataset=dataset, dataset_mime_type=MIME_TYPE_JSON, dataset_name="dataset")

        assert cols[DatasetColumns.MODEL_INPUT.value.name] == expected_model_inputs
        assert cols[DatasetColumns.MODEL_OUTPUT.value.name] == expected_model_outputs
        assert cols[DatasetColumns.TARGET_OUTPUT.value.name] == expected_target_outputs
        assert cols[DatasetColumns.CATEGORY.value.name] == expected_categories

        # ensure that ColumnNames.SENT_MORE_INPUT_COLUMN.value.name does not show up in `cols`
        assert set(cols.keys()) == {
            DatasetColumns.MODEL_INPUT.value.name,
            DatasetColumns.MODEL_OUTPUT.value.name,
            DatasetColumns.TARGET_OUTPUT.value.name,
            DatasetColumns.CATEGORY.value.name,
        }

        # ensure that logger generated a warning when search_jmespath
        # was called on ColumnNames.SENT_MORE_INPUT_COLUMN.value.name.
        mock_logger.assert_called_with(
            f"Failed to find {DatasetColumns.SENT_MORE_INPUT.value.name} columns in dataset `dataset` "
            f"using JMESPath query '{config.sent_more_input_location}'."
        )

    @patch("src.fmeval.data_loaders.jmespath_util.logging.Logger.warning")
    def test_parse_dataset_columns_success_jsonlines(self, mock_logger):
        """
        GIVEN valid JMESPath queries that extract model inputs, model outputs,
            target outputs, and categories, and a single line from a JSON Lines
            dataset (represented by a dict)
        WHEN parse_dataset_columns is called
        THEN parse_dataset_columns returns correct results
        """
        config = DataConfig(
            dataset_name="dataset",
            dataset_uri="uri",
            dataset_mime_type=MIME_TYPE_JSONLINES,
            model_input_location="input",
            model_output_location="output",
            target_output_location="target",
            category_location="category",
            # this JMESPath query will fail to find any results, and should effectively get ignored
            sent_more_input_location="invalid_jmespath_query",
        )
        parser = JsonParser(config)
        expected_model_input = "A"
        expected_model_output = "positive"
        expected_target_output = "negative"
        expected_category = "Red"

        dataset_line = {"input": "A", "output": "positive", "target": "negative", "category": "Red"}
        cols = parser.parse_dataset_columns(
            dataset=dataset_line, dataset_mime_type=MIME_TYPE_JSONLINES, dataset_name="dataset_line"
        )
        assert cols[DatasetColumns.MODEL_INPUT.value.name] == expected_model_input
        assert cols[DatasetColumns.MODEL_OUTPUT.value.name] == expected_model_output
        assert cols[DatasetColumns.TARGET_OUTPUT.value.name] == expected_target_output
        assert cols[DatasetColumns.CATEGORY.value.name] == expected_category

        # ensure that ColumnNames.SENT_MORE_INPUT_COLUMN.value.name does not show up in `cols`
        assert set(cols.keys()) == {
            DatasetColumns.MODEL_INPUT.value.name,
            DatasetColumns.MODEL_OUTPUT.value.name,
            DatasetColumns.TARGET_OUTPUT.value.name,
            DatasetColumns.CATEGORY.value.name,
        }

        # ensure that logger generated a warning when search_jmespath
        # was called on ColumnNames.SENT_MORE_INPUT_COLUMN.value.name.
        mock_logger.assert_called_with(
            f"Failed to find {DatasetColumns.SENT_MORE_INPUT.value.name} columns in dataset `dataset_line` "
            f"using JMESPath query '{config.sent_more_input_location}'."
        )

    def test_parse_dataset_columns_casting_to_string(self):
        """
        GIVEN a DataConfig that specifies both columns that should be casted
            to strings and columns that shouldn't be casted
        WHEN parse_dataset_columns is called
        THEN the returned results are casted and not casted appropriately
        """
        config = DataConfig(
            dataset_name="dataset",
            dataset_uri="uri",
            dataset_mime_type=MIME_TYPE_JSONLINES,
            # these columns should be casted to string
            model_input_location="input",
            model_output_location="output",
            target_output_location="target",
            category_location="category",
            # these columns shouldn't be casted
            sent_more_log_prob_location="sent_more_lp",
            sent_less_log_prob_location="sent_less_lp",
        )
        parser = JsonParser(config)
        expected_model_input = "1.0"
        expected_model_output = "True"
        expected_target_output = "False"
        expected_category = "2"
        expected_sent_more_log_prob = -162
        expected_sent_less_log_prob = -1.89

        dataset_line = {
            "input": 1.0,
            "output": True,
            "target": False,
            "category": 2,
            "sent_more_lp": -162,
            "sent_less_lp": -1.89,
        }
        cols = parser.parse_dataset_columns(
            dataset=dataset_line, dataset_mime_type=MIME_TYPE_JSONLINES, dataset_name="dataset_line"
        )
        assert cols[DatasetColumns.MODEL_INPUT.value.name] == expected_model_input
        assert cols[DatasetColumns.MODEL_OUTPUT.value.name] == expected_model_output
        assert cols[DatasetColumns.TARGET_OUTPUT.value.name] == expected_target_output
        assert cols[DatasetColumns.CATEGORY.value.name] == expected_category
        assert cols[DatasetColumns.SENT_MORE_LOG_PROB.value.name] == expected_sent_more_log_prob
        assert cols[DatasetColumns.SENT_LESS_LOG_PROB.value.name] == expected_sent_less_log_prob

    def test_parse_dataset_columns_invalid_dataset(self):
        """
        GIVEN a dataset that is not a dict or list
        WHEN parse_dataset_columns is called
        THEN parse_dataset_columns raises an EvalAlgorithmError
        """
        with pytest.raises(
            EvalAlgorithmInternalError,
            match="parse_dataset_columns requires dataset `dataset` to be deserialized into a dict/list.",
        ):
            parser = JsonParser(Mock())
            parser.parse_dataset_columns(dataset=17, dataset_mime_type=MIME_TYPE_JSON, dataset_name="dataset")

    def test_validate_parsed_columns_lengths(self):
        """
        GIVEN a dict storing lists of differing lengths
        WHEN _validate_parsed_columns_lengths is called
        THEN _validate_parsed_columns_lengths raises an EvalAlgorithmClientError
        """
        with pytest.raises(EvalAlgorithmClientError, match="Expected the number of samples"):
            dataset = {
                "model_input": {"sample_1": "a", "sample_2": "b", "sample_3": "c"},
                "model_output": {"sample_1": "d", "sample_2": "e"},
            }
            JsonParser._validate_parsed_columns_lengths(dataset)

    def test_cast_to_string_success(self):
        """
        GIVEN an input that can be cast to a string/list of strings without issues
        WHEN JsonParser._cast_to_string is called
        THEN the correct output is returned
        """
        original_data = ["a", True, False, 1, 2.0]
        expected = ["a", "True", "False", "1", "2.0"]

        # JSON case
        args = Mock()
        args.dataset_mime_type = MIME_TYPE_JSON
        assert JsonParser._cast_to_string(original_data, args) == expected

        # JSON Lines case
        args.dataset_mime_type = MIME_TYPE_JSONLINES
        for data, expected_data in zip(original_data, expected):
            assert JsonParser._cast_to_string(data, args) == expected_data

    def test_cast_to_string_failure(self):
        class BadObject:
            def __str__(self):
                raise Exception

        bad_object = BadObject()
        args = Mock()

        # JSON case
        with pytest.raises(
            EvalAlgorithmClientError, match="Failed to cast object to string in json_parser._cast_to_string."
        ):
            args.dataset_mime_type = MIME_TYPE_JSON
            JsonParser._cast_to_string([1, False, "hello", bad_object], args)

        # JSON Lines case
        with pytest.raises(
            EvalAlgorithmClientError, match="Failed to cast object to string in json_parser._cast_to_string."
        ):
            args.dataset_mime_type = MIME_TYPE_JSONLINES
            JsonParser._cast_to_string(bad_object, args)
