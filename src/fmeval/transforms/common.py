from typing import Any, Dict, List
from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.transform import Transform
from fmeval.transforms.util import (
    assert_condition,
    validate_key_uniqueness,
    validate_existing_keys,
    validate_added_keys,
)


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
            Example: "Summarize the following text: $feature".
        """
        super().__init__(input_keys, output_keys, prompt_template)
        self.prompt_composer = PromptComposer(prompt_template)

    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augments the input record with LLM prompts and returns said record.

        :param record: The input record.
        :returns: The input record with prompts added in.
        """
        validate_existing_keys(record, self.input_keys)
        original_keys = set(record.keys())

        for input_key, prompt_key in zip(self.input_keys, self.output_keys):
            record[prompt_key] = self.prompt_composer.compose(record[input_key])

        # A defensive sanity check; self.output_keys should never be mutated.
        validate_key_uniqueness(self.output_keys)
        validate_added_keys(
            current_keys=set(record.keys()), original_keys=original_keys, keys_to_add=set(self.output_keys)
        )
        return record


class GetModelResponse(Transform):
    """This transform augments an input record with a ModelRunner's `predict` response.

    At the moment, this transform only accepts a single input key, meaning that
    if you wish to invoke the same ModelRunner on a different input, you should
    instantiate another instance of this class with said input key.
    """

    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        model_runner: ModelRunner,
    ):
        """GetModelResponse initializer.

        :param input_keys: A single-element list containing the key corresponding to the ModelRunner's input.
        :param output_keys: The keys corresponding to the data in the model response payload. Note that
            this parameter is dependent on the behavior of model_runner's `predict` method. For example,
            for ModelRunners that do not return log probabilities, `output_keys` should not contain a key
            for log probabilities.
        :param model_runner: The ModelRunner instance whose responses will be obtained.
        """
        assert_condition(len(input_keys) == 1, "GetModelResponse takes a single input key.")
        super().__init__(input_keys, output_keys, model_runner)
        self.model_runner = model_runner

    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augments the input record with model responses and returns said record.

        :param record: The input record.
        :returns: The input record with model response data added in.
        """
        validate_existing_keys(record, self.input_keys)
        original_keys = set(record.keys())

        input_key = self.input_keys[0]
        model_output, log_prob = self.model_runner.predict(record[input_key])
        model_response = ((model_output,) if model_output is not None else ()) + (
            (log_prob,) if log_prob is not None else ()
        )
        assert_condition(
            len(model_response) == len(self.output_keys),
            f"The number of elements in model response {model_response} "
            f"does not match number of output keys in {self.output_keys}.",
        )

        for model_response_item, model_output_key in zip(model_response, self.output_keys):
            record[model_output_key] = model_response_item

        # A defensive sanity check; self.output_keys should never be mutated.
        validate_key_uniqueness(self.output_keys)
        validate_added_keys(
            current_keys=set(record.keys()), original_keys=original_keys, keys_to_add=set(self.output_keys)
        )
        return record
