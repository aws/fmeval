import pytest
from unittest.mock import patch

from fmeval.transforms.common import GeneratePrompt, GetModelOutputs, GetLogProbabilities, Mean


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


def test_get_model_outputs_init_success():
    """
    GIVEN valid initializer arguments.
    WHEN a GetModelOutputs object is instantiated.
    THEN the instance's attributes match what is expected.
    """
    with patch("fmeval.transforms.common.ModelRunner") as mock_model_runner:
        get_model_outputs = GetModelOutputs(
            input_to_output_keys={
                "prompt_a": ["model_output_1", "model_output_2"],
                "prompt_b": ["model_output_3", "model_output_4"],
            },
            model_runner=mock_model_runner,
        )
        assert get_model_outputs.input_keys == ["prompt_a", "prompt_b"]
        assert get_model_outputs.output_keys == ["model_output_1", "model_output_2", "model_output_3", "model_output_4"]
        assert get_model_outputs.model_runner == mock_model_runner


@pytest.mark.parametrize("model_output", [None, "some output"])
@pytest.mark.parametrize("log_prob", [None, -0.162])
def test_get_model_outputs_call_success(model_output, log_prob):
    """
    GIVEN a GetModelOutputs instance.
    WHEN its __call__ method is called on a record.
    THEN the output contains the model_output portion of the model
        response payload and does *not* include the log probability
        portion of the response payload, even if it is non-null.
    """
    with patch("fmeval.transforms.common.ModelRunner") as mock_model_runner:
        mock_model_runner.predict.return_value = (model_output, log_prob)
        get_model_outputs = GetModelOutputs(
            input_to_output_keys={"input": ["model_output"]}, model_runner=mock_model_runner
        )
        sample = {"input": "Hello"}
        result = get_model_outputs(sample)
        assert result == {"input": "Hello", "model_output": model_output}


def test_get_model_outputs_call_multiple_inputs():
    """
    GIVEN a GetModelOutputs instance with multiple input keys configured.
    WHEN its __call__ method is called.
    THEN the correct output is returned.
    """
    with patch("fmeval.transforms.common.ModelRunner") as mock_model_runner:
        mock_model_runner.predict.side_effect = [("output 1", -0.162), ("output 2", -0.189)]
        get_model_outputs = GetModelOutputs(
            input_to_output_keys={
                "input_1": ["output_key_1"],
                "input_2": ["output_key_2"],
            },
            model_runner=mock_model_runner,
        )
        sample = {"input_1": "input 1", "input_2": "input 2"}
        expected_result = {
            "input_1": "input 1",
            "output_key_1": "output 1",
            "input_2": "input 2",
            "output_key_2": "output 2",
        }
        result = get_model_outputs(sample)
        assert result == expected_result


def test_get_model_outputs_call_multiple_output_keys():
    """
    GIVEN a GetModelOutputs instance with multiple output keys configured for a single input key.
    WHEN its __call__ method is called.
    THEN the correct output is returned.
    """
    with patch("fmeval.transforms.common.ModelRunner") as mock_model_runner:
        mock_model_runner.predict.side_effect = [("output 1", -0.162), ("output 2", -0.189)]
        get_model_outputs = GetModelOutputs(
            input_to_output_keys={
                "input": ["output_key_1", "output_key_2"],
            },
            model_runner=mock_model_runner,
        )
        sample = {"input": "some input"}
        expected_result = {
            "input": "some input",
            "output_key_1": "output 1",
            "output_key_2": "output 2",
        }
        result = get_model_outputs(sample)
        assert result == expected_result


@pytest.mark.parametrize("model_output", [None, "some output"])
@pytest.mark.parametrize("log_prob", [None, -0.162])
def test_get_log_probs_call(model_output, log_prob):
    """
    GIVEN a GetLogProbabilities instance.
    WHEN its __call__ method is called on a record.
    THEN the output contains the log_prob portion of the model
        response payload and does *not* include the model_output
        portion of the response payload, even if it is non-null.
    """
    with patch("fmeval.transforms.common.ModelRunner") as mock_model_runner:
        mock_model_runner.predict.return_value = (model_output, log_prob)
        get_model_outputs = GetLogProbabilities(
            input_keys=["input"], output_keys=["log_prob"], model_runner=mock_model_runner
        )
        sample = {"input": "Hello"}
        result = get_model_outputs(sample)
        assert result == {"input": "Hello", "log_prob": log_prob}


def test_get_log_probs_call_multiple_inputs():
    """
    GIVEN a GetLogProbabilities instance configured with multiple input keys.
    WHEN its __call__ method is called on a record.
    THEN the correct output is returned.
    """
    with patch("fmeval.transforms.common.ModelRunner") as mock_model_runner:
        mock_model_runner.predict.side_effect = [(None, -0.162), ("some output", -0.189)]
        get_model_outputs = GetLogProbabilities(
            input_keys=["input_1", "input_2"], output_keys=["log_prob_1", "log_prob_2"], model_runner=mock_model_runner
        )
        sample = {"input_1": "Hello", "input_2": "Hi"}
        result = get_model_outputs(sample)
        assert result == {
            "input_1": "Hello",
            "input_2": "Hi",
            "log_prob_1": -0.162,
            "log_prob_2": -0.189,
        }


def test_mean_call():
    """
    GIVEN a Mean instance.
    WHEN its __call__ method is called.
    THEN the correct output is returned.
    """
    m = Mean(input_keys=["a", "b", "c"], output_key="mean")
    sample = {"a": 1, "b": 6, "c": 2}
    assert m(sample)["mean"] == 3.0
