import pytest
from unittest.mock import patch
from typing import NamedTuple, List, Optional, Dict, Any, Tuple

from fmeval.transforms.common import GeneratePrompt, GetModelResponses, Mean
from fmeval.util import EvalAlgorithmInternalError


def test_generate_prompt_init():
    """
    GIVEN valid initializer arguments.
    WHEN a GeneratePrompt object is instantiated.
    THEN the instance's attributes match what is expected.
    """
    with patch("fmeval.transforms.common.PromptComposer") as mock_prompt_composer:
        gen_prompt = GeneratePrompt(["model_input"], ["prompt"], "Summarize the following: $model_input")
        assert gen_prompt.input_keys == ["model_input"]
        assert gen_prompt.output_keys == ["prompt"]
        mock_prompt_composer.assert_called_once_with("Summarize the following: $model_input")


def test_generate_prompt_call():
    """
    GIVEN a GeneratePrompt instance.
    WHEN its __call__ method is called on a record.
    THEN the correct output is returned.
    """
    gen_prompt = GeneratePrompt(
        ["input_1", "input_2"], ["prompt_1", "prompt_2"], "Summarize the following: $model_input"
    )
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
        get_model_response = GetModelResponses(
            input_key_to_response_keys={"prompt": [("model_output",)]}, model_runner=mock_model_runner
        )
        assert get_model_response.input_keys == ["prompt"]
        assert get_model_response.output_keys == ["model_output"]
        assert get_model_response.model_runner == mock_model_runner


class TestCaseGetModelResponseSuccess(NamedTuple):
    model_output: Optional[str]
    log_prob: Optional[float]
    response_keys: List[Tuple[str]]
    expected_result: Dict[str, Any]


@pytest.mark.parametrize(
    "model_output, log_prob, response_keys, expected_result",
    [
        TestCaseGetModelResponseSuccess(
            model_output="some output",
            log_prob=-0.162,
            response_keys=[("model_output", "log_prob")],
            expected_result={"input": "Hello", "model_output": "some output", "log_prob": -0.162},
        ),
        TestCaseGetModelResponseSuccess(
            model_output="some output",
            log_prob=None,
            response_keys=[("model_output",)],
            expected_result={"input": "Hello", "model_output": "some output"},
        ),
        TestCaseGetModelResponseSuccess(
            model_output=None,
            log_prob=-0.162,
            response_keys=[("log_prob",)],
            expected_result={"input": "Hello", "log_prob": -0.162},
        ),
    ],
)
def test_get_model_response_call_success(model_output, log_prob, response_keys, expected_result):
    """
    GIVEN a GetModelResponses instance.
    WHEN its __call__ method is called on a record.
    THEN the correct output is returned.
    """
    with patch("fmeval.transforms.common.ModelRunner") as mock_model_runner:
        mock_model_runner.predict.return_value = (model_output, log_prob)
        get_model_response = GetModelResponses(
            input_key_to_response_keys={"input": response_keys}, model_runner=mock_model_runner
        )
        sample = {"input": "Hello"}
        result = get_model_response(sample)
        assert result == expected_result


def test_get_model_response_call_multiple_inputs():
    """
    GIVEN a GetModelResponses instance with multiple input keys configured.
    WHEN its __call__ method is called.
    THEN the correct output is returned.
    """
    with patch("fmeval.transforms.common.ModelRunner") as mock_model_runner:
        mock_model_runner.predict.side_effect = [("output 1", -0.162), ("output 2", -0.189), ("output 3", -0.126)]
        get_model_response = GetModelResponses(
            input_key_to_response_keys={
                "input_1": [("output_key_1", "log_prob_key_1"), ("output_key_2", "log_prob_key_2")],
                "input_2": [("output_key_3", "log_prob_key_3")],
            },
            model_runner=mock_model_runner,
        )
        sample = {"input_1": "input 1", "input_2": "input 2"}
        expected_result = {
            "input_1": "input 1",
            "output_key_1": "output 1",
            "log_prob_key_1": -0.162,
            "input_2": "input 2",
            "output_key_2": "output 2",
            "log_prob_key_2": -0.189,
            "output_key_3": "output 3",
            "log_prob_key_3": -0.126,
        }
        result = get_model_response(sample)
        assert result == expected_result


class TestCaseGetModelResponseFailure(NamedTuple):
    model_output: Optional[str]
    log_prob: Optional[float]
    response_keys: List[Tuple[str]]


@pytest.mark.parametrize(
    "model_output, log_prob, response_keys",
    [
        TestCaseGetModelResponseFailure(
            model_output="some output",
            log_prob=-0.162,
            response_keys=[("model_output",)],
        ),
        TestCaseGetModelResponseFailure(
            model_output="some output",
            log_prob=None,
            response_keys=[("model_output", "log_prob")],
        ),
        TestCaseGetModelResponseFailure(
            model_output=None,
            log_prob=-0.162,
            response_keys=[("model_output", "log_prob")],
        ),
    ],
)
def test_get_model_response_call_failure(model_output, log_prob, response_keys):
    """
    GIVEN a GetModelResponses instance where the number of output keys corresponding to
        a particular input key does not match the number of non-null elements in its model runner's
        predict() response.
    WHEN its __call__ method is called.
    THEN an EvalAlgorithmInternalError is raised.
    """
    sample = {"input": "Hello"}
    with patch("fmeval.transforms.common.ModelRunner") as mock_model_runner:
        mock_model_runner.predict.return_value = (model_output, log_prob)
        get_model_response = GetModelResponses(
            input_key_to_response_keys={"input": response_keys},
            model_runner=mock_model_runner,
        )
        with pytest.raises(EvalAlgorithmInternalError, match="The number of elements in model response"):
            get_model_response(sample)


def test_mean_call():
    """
    GIVEN a Mean instance.
    WHEN its __call__ method is called.
    THEN the correct output is returned.
    """
    m = Mean(input_keys=["a", "b", "c"], output_key="mean")
    sample = {"a": 1, "b": 6, "c": 2}
    assert m(sample)["mean"] == 3.0
