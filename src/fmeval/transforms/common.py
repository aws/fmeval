import numpy as np
from typing import Any, Dict, List, Tuple

from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.transform import Transform
from fmeval.transforms.util import validate_call
from fmeval.util import assert_condition


class GeneratePrompt(Transform):
    """This transform augments an input record with LLM prompts constructed according to a template.

    If multiple input keys are provided, this transform creates prompts out of all of them,
    applying the same prompt template to each input.
    """

    def __init__(self, input_keys: List[str], output_keys: List[str], prompt_template: str):
        """GeneratePrompt initializer.

        :param input_keys: The keys corresponding to the text that will be used to create prompts.
            If multiple input keys are provided, a prompt will be constructed from each input, but
            the created prompts will all utilize the same prompt template.
        :param output_keys: The keys corresponding to the prompts that get added by this Transform.
        :param prompt_template: The template used to construct the prompt.
            Example: "Summarize the following text: $model_input".
        """
        super().__init__(input_keys, output_keys, prompt_template)
        self.register_input_output_keys(input_keys, output_keys)
        self.prompt_composer = PromptComposer(prompt_template)

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with LLM prompts and returns said record.

        :param record: The input record.
        :returns: The input record with prompts added in.
        """
        for input_key, prompt_key in zip(self.input_keys, self.output_keys):
            record[prompt_key] = self.prompt_composer.compose(record[input_key])
        return record


class GetModelResponse(Transform):
    """This transform invokes a ModelRunner's `predict` method and augments the input record with the response payload.

    An instance of this transform can be configured to get model responses for multiple inputs.
    See __init__ docstring for more details.
    """

    def __init__(
        self,
        input_key_to_response_keys: Dict[str, List[Tuple[str]]],
        model_runner: ModelRunner,
    ):
        """GetModelResponse initializer.

        :param input_key_to_response_keys: Maps an input key (corresponding to
            the input payload to the model) to a list of tuples, where each tuple
            contains the keys for the model response payload that results from
            invoking the model on the input. Since the model can be invoked on
            the same input multiple times, we map the input key to a list of tuples
            instead of a single tuple.

            Note that the format of the response key tuple depends on the behavior of
            model_runner's `predict` method. For ModelRunners whose `predict` method
            returns both a model output and a log probability, the response key tuple
            will be of the form (model_output_key, log_probability_key). If `predict`
            returns only the model output or only the log probability, the response key
            tuple will correspondingly be a single-element tuple.

            Example:
                input_key_to_response_keys = {
                    "input_1": [
                        ("input_1_model_output_1", "input_1_log_prob_1"),
                        ("input_1_model_output_2", "input_1_log_prob_2")
                    ],
                    "input_2": [
                        ("input_2_model_output_1", "input_2_log_prob_1"),
                        ("input_2_model_output_2", "input_2_log_prob_2")
                    ],
                }

        :param model_runner: The ModelRunner instance whose responses will be obtained.
        """
        super().__init__(input_key_to_response_keys, model_runner)
        self.register_input_output_keys(
            input_keys=list(input_key_to_response_keys.keys()),
            output_keys=[
                response_key
                for list_of_response_key_tuples in input_key_to_response_keys.values()
                for response_key_tuple in list_of_response_key_tuples
                for response_key in response_key_tuple
            ],
        )
        self.input_key_to_response_keys = input_key_to_response_keys
        self.model_runner = model_runner

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with model responses and returns said record.

        :param record: The input record.
        :returns: The input record with model response data added in.
        """
        for input_key in self.input_keys:
            response_key_tuples = self.input_key_to_response_keys[input_key]
            for response_key_tuple in response_key_tuples:
                model_output, log_prob = self.model_runner.predict(record[input_key])
                model_response = ((model_output,) if model_output is not None else ()) + (
                    (log_prob,) if log_prob is not None else ()
                )
                assert_condition(
                    len(model_response) == len(response_key_tuple),
                    f"The number of elements in model response {model_response} "
                    f"does not match number of response keys in {response_key_tuple}.",
                )
                for model_response_key, model_response_item in zip(response_key_tuple, model_response):
                    record[model_response_key] = model_response_item
        return record


class Mean(Transform):
    """This transform computes the arithmetic mean of specified values in a record and augments said record."""

    def __init__(self, input_keys: List[str], output_key: str):
        """Mean initializer.
        :param input_keys: The keys corresponding to the values to take the mean of.
        :param output_key: The key corresponding to the mean value, which gets
            added to the record.
        """
        super().__init__(input_keys, output_key)
        self.register_input_output_keys(input_keys, [output_key])
        self.output_key = output_key

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with the computed mean.
        :param record: The input record.
        :returns: The input record with the mean added in.
        """
        avg = np.mean([record[input_key] for input_key in self.input_keys])
        record[self.output_key] = avg
        return record
