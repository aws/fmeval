import pytest
from unittest.mock import patch
from typing import NamedTuple, List, Optional, Dict, Any

from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.transforms.common import GeneratePrompt, GetModelResponse
from fmeval.util import EvalAlgorithmInternalError


def test_generate_prompt_init():
    """
    GIVEN valid initializer arguments.
    WHEN a GeneratePrompt object is instantiated.
    THEN the instance's attributes match what is expected.
    """
    with patch("fmeval.transforms.common.PromptComposer") as mock_prompt_composer:
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
    with patch("fmeval.transforms.common.ModelRunner") as mock_model_runner:
        get_model_response = GetModelResponse(["prompt"], ["model_output"], mock_model_runner)
        assert get_model_response.input_keys == ["prompt"]
        assert get_model_response.output_keys == ["model_output"]
        assert get_model_response.model_runner == mock_model_runner


def test_get_model_response_init_failure():
    """
    GIVEN a list of input keys where the number of keys != 1.
    WHEN a GeneratePrompt object is instantiated.
    THEN an EvalAlgorithmClientError is raised.
    """
    with pytest.raises(EvalAlgorithmClientError, match="GetModelResponse takes a single input key."):
        with patch("fmeval.transforms.common.ModelRunner") as mock_model_runner:
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
    with patch("fmeval.transforms.common.ModelRunner") as mock_model_runner:
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
    with patch("fmeval.transforms.common.ModelRunner") as mock_model_runner:
        mock_model_runner.predict.return_value = (model_output, log_prob)
        get_model_response = GetModelResponse(["input"], output_keys, mock_model_runner)
        with pytest.raises(EvalAlgorithmInternalError, match="The number of elements in model response"):
            get_model_response(sample)
