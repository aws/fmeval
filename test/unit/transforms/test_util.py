import pytest
from unittest.mock import patch
from typing import NamedTuple, List, Optional, Dict, Any
from fmeval.transforms.util import validate_added_keys, validate_existing_keys, GeneratePrompt, GetModelResponse
from fmeval.util import EvalAlgorithmInternalError


def test_validate_existing_keys_success():
    """
    GIVEN a record containing all expected keys.
    WHEN validate_existing_keys is called.
    THEN no exception is raised.
    """
    record = {"a": 1, "b": 2, "c": 3, "d": 4}
    keys = ["a", "b", "c"]
    validate_existing_keys(record, keys)


def test_validate_existing_keys_failure():
    """
    GIVEN a record that is missing an expected key.
    WHEN validate_existing_keys is called.
    THEN an EvalAlgorithmInternalError is raised.
    """
    record = {"a": 1, "b": 2, "d": 4}
    keys = ["a", "b", "c"]
    with pytest.raises(
        EvalAlgorithmInternalError, match=f"Record {record} is expected to contain key 'c', but does not."
    ):
        validate_existing_keys(record, keys)


def test_validate_added_keys_success():
    """
    GIVEN arguments that should not raise an exception.
    WHEN validate_added_keys is called.
    THEN no exception is raised.
    """
    current_keys = ["a", "b", "c", "d"]
    original_keys = ["a", "c"]
    keys_to_add = ["b", "d"]
    validate_added_keys(current_keys, original_keys, keys_to_add)


class TestCaseValidateAddedKeys(NamedTuple):
    current_keys: List[str]
    original_keys: List[str]
    keys_to_add: List[str]
    err_msg: str


@pytest.mark.parametrize(
    "current_keys, original_keys, keys_to_add, err_msg",
    [
        TestCaseValidateAddedKeys(
            current_keys=["a", "b", "b"], original_keys=["a"], keys_to_add=["b"], err_msg="current_keys list"
        ),
        TestCaseValidateAddedKeys(
            current_keys=["a", "b"], original_keys=["a", "a"], keys_to_add=["b"], err_msg="original_keys list"
        ),
        TestCaseValidateAddedKeys(
            current_keys=["a", "b"], original_keys=["a"], keys_to_add=["b", "b"], err_msg="keys_to_add list"
        ),
        TestCaseValidateAddedKeys(
            current_keys=["a", "b", "c", "d"], original_keys=["a", "c"], keys_to_add=["b"], err_msg="The set difference"
        ),
        TestCaseValidateAddedKeys(
            current_keys=["a", "b", "c"], original_keys=["a", "c"], keys_to_add=["b", "d"], err_msg="The set difference"
        ),
    ],
)
def test_validate_added_keys_failure(
    current_keys: List[str], original_keys: List[str], keys_to_add: List[str], err_msg: str
):
    """
    GIVEN arguments that should raise an exception.
    WHEN validate_added_keys is called.
    THEN an EvalAlgorithmInternalError is raised.
    """
    with pytest.raises(EvalAlgorithmInternalError, match=err_msg):
        validate_added_keys(current_keys, original_keys, keys_to_add)


def test_generate_prompt_init():
    """
    GIVEN valid initializer arguments.
    WHEN a GeneratePrompt object is instantiated.
    THEN the instance's attributes match what is expected.
    """
    with patch("fmeval.transforms.util.PromptComposer") as mock_prompt_composer:
        gen_prompt = GeneratePrompt(["model_input"], ["prompt"], "Summarize the following: $feature")
        assert gen_prompt.input_keys == ["model_input"]
        assert gen_prompt.output_keys == ["prompt"]
        mock_prompt_composer.assert_called_once_with("Summarize the following: $feature")


def test_generate_prompt_call():
    """
    GIVEN a GeneratePrompt instance.
    WHEN its __call__ method is called on a record.
    THEN the correct output is returned.
    """
    gen_prompt = GeneratePrompt(["input_1", "input_2"], ["prompt_1", "prompt_2"], "Summarize the following: $feature")
    sample = {"input_1": "Hello", "input_2": "world"}
    result = gen_prompt(sample)
    assert result == {
        "input_1": "Hello",
        "input_2": "world",
        "prompt_1": "Summarize the following: Hello",
        "prompt_2": "Summarize the following: world",
    }


def test_get_model_response_init_success():
    """
    GIVEN valid initializer arguments.
    WHEN a GeneratePrompt object is instantiated.
    THEN the instance's attributes match what is expected.
    """
    with patch("fmeval.transforms.util.ModelRunner") as mock_model_runner:
        get_model_response = GetModelResponse(["prompt"], ["model_output"], mock_model_runner)
        assert get_model_response.input_keys == ["prompt"]
        assert get_model_response.output_keys == ["model_output"]
        assert get_model_response.model_runner == mock_model_runner


def test_get_model_response_init_failure():
    """
    GIVEN a list of input keys where the number of keys != 1.
    WHEN a GeneratePrompt object is instantiated.
    THEN an EvalAlgorithmInternalError is raised.
    """
    with pytest.raises(EvalAlgorithmInternalError, match="GetModelResponse takes a single input key."):
        with patch("fmeval.transforms.util.ModelRunner") as mock_model_runner:
            GetModelResponse(["prompt_1", "prompt_2"], ["model_output_1", "model_output_2"], mock_model_runner)


class TestCaseGetModelResponseSuccess(NamedTuple):
    model_output: Optional[str]
    log_prob: Optional[float]
    output_keys: List[str]
    expected_result: Dict[str, Any]


@pytest.mark.parametrize(
    "model_output, log_prob, output_keys, expected_result",
    [
        TestCaseGetModelResponseSuccess(
            model_output="some output",
            log_prob=-0.162,
            output_keys=["model_output", "log_prob"],
            expected_result={"input": "Hello", "model_output": "some output", "log_prob": -0.162},
        ),
        TestCaseGetModelResponseSuccess(
            model_output="some output",
            log_prob=None,
            output_keys=["model_output"],
            expected_result={"input": "Hello", "model_output": "some output"},
        ),
        TestCaseGetModelResponseSuccess(
            model_output=None,
            log_prob=-0.162,
            output_keys=["log_prob"],
            expected_result={"input": "Hello", "log_prob": -0.162},
        ),
    ],
)
def test_get_model_response_call_success(model_output, log_prob, output_keys, expected_result):
    """
    GIVEN a GetModelResponse instance.
    WHEN its __call__ method is called on a record.
    THEN the correct output is returned.
    """
    with patch("fmeval.transforms.util.ModelRunner") as mock_model_runner:
        mock_model_runner.predict.return_value = (model_output, log_prob)
        get_model_response = GetModelResponse(["input"], output_keys, mock_model_runner)
        sample = {"input": "Hello"}
        result = get_model_response(sample)
        assert result == expected_result


class TestCaseGetModelResponseFailure(NamedTuple):
    model_output: Optional[str]
    log_prob: Optional[float]
    output_keys: List[str]


@pytest.mark.parametrize(
    "model_output, log_prob, output_keys",
    [
        TestCaseGetModelResponseFailure(
            model_output="some output",
            log_prob=-0.162,
            output_keys=["model_output"],
        ),
        TestCaseGetModelResponseFailure(
            model_output="some output",
            log_prob=None,
            output_keys=["model_output", "log_prob"],
        ),
        TestCaseGetModelResponseFailure(
            model_output=None,
            log_prob=-0.162,
            output_keys=["model_output", "log_prob"],
        ),
    ],
)
def test_get_model_response_call_failure(model_output, log_prob, output_keys):
    """
    GIVEN a GetModelResponse instance whose `output_keys` attribute has a different number of elements
        than the number of non-null elements in its model runner's predict() response.
    WHEN its __call__ method is called.
    THEN an EvalAlgorithmInternalError is raised.
    """
    sample = {"input": "Hello"}
    with patch("fmeval.transforms.util.ModelRunner") as mock_model_runner:
        mock_model_runner.predict.return_value = (model_output, log_prob)
        get_model_response = GetModelResponse(["input"], output_keys, mock_model_runner)
        with pytest.raises(EvalAlgorithmInternalError, match="The number of elements in model response"):
            get_model_response(sample)
