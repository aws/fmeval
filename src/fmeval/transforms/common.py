from typing import Any, Dict, List


from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.transform import Transform
from fmeval.transforms.util import (
    validate_key_uniqueness,
    validate_existing_keys,
    validate_added_keys,
    validate_call,
)
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
        input_to_output_keys: Dict[str, List[str]],
        model_runner: ModelRunner,
    ):
        """GetModelResponse initializer.

        :param input_to_output_keys: Maps an input key to a list of output keys.
            The input key corresponds to the model input (i.e. the input payload
            to `model_runner`) while the output keys correspond to the response
            payload resulting from invoking `model_runner`.
            Note that the list of output keys is dependent on the behavior of the
            model runner's `predict` method. For example, for ModelRunners that do
            not return log probabilities, the output keys list should not contain a key
            for log probabilities.
        :param model_runner: The ModelRunner instance whose responses will be obtained.
        """
        super().__init__(input_to_output_keys, model_runner)
        self.register_input_output_keys(
            list(input_to_output_keys.keys()),
            [output_key for output_keys in input_to_output_keys.values() for output_key in output_keys],
        )
        self.input_to_output_keys = input_to_output_keys
        self.model_runner = model_runner

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with model responses and returns said record.

        :param record: The input record.
        :returns: The input record with model response data added in.
        """
        for input_key in self.input_keys:
            model_output, log_prob = self.model_runner.predict(record[input_key])
            model_response = ((model_output,) if model_output is not None else ()) + (
                (log_prob,) if log_prob is not None else ()
            )
            output_keys = self.input_to_output_keys[input_key]
            assert_condition(
                len(model_response) == len(output_keys),
                f"The number of elements in model response {model_response} "
                f"does not match number of output keys in {output_keys}.",
            )
            for model_response_item, model_output_key in zip(model_response, output_keys):
                record[model_output_key] = model_response_item
        return record
