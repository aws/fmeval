from typing import Any, Dict, List
from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.transform import Transform
from fmeval.util import assert_condition


def validate_existing_keys(record: Dict[str, Any], keys: List[str]) -> None:
    """Validate that all expected keys are present in a record.

    :param record: The record to be validated.
    :param keys: The keys that are expected to be present in the record.
    :raises: EvalAlgorithmInternalError if any validation fails.
    """
    for key in keys:
        assert_condition(key in record, f"Record {record} is expected to contain key '{key}', but does not.")


def validate_added_keys(current_keys: List[str], original_keys: List[str], keys_to_add: List[str]) -> None:
    """Validate that the set difference between `current_keys` and `original_keys` is `keys_to_add`.

    Note: this function should only be used when by Transforms that mutate their input record.
    It doesn't make sense to call this function if the Transform constructs a new record to
    be used as output, since said output record need not contain all the keys in the input record.

    :param current_keys: The keys that are currently present in a record.
    :param original_keys: The keys that were originally present in a record
        (prior to performing transform-specific logic, which adds new keys).
    :param keys_to_add: The keys that a transform should have added to its input record.
        When this function is called from within a Transform's __call__ method (which should
        be the primary use case for this function), this parameter should be the transform's
        `output_keys` attribute.
    :raises: EvalAlgorithmInternalError if any validation fails.
    """
    current_keys_set = set(current_keys)
    assert_condition(
        len(current_keys_set) == len(current_keys), f"current_keys list {current_keys} contains a duplicate key."
    )
    original_keys_set = set(original_keys)
    assert_condition(
        len(original_keys_set) == len(original_keys), f"original_keys list {original_keys} contains a duplicate key."
    )
    keys_to_add_set = set(keys_to_add)
    assert_condition(
        len(keys_to_add_set) == len(keys_to_add), f"keys_to_add list {keys_to_add} contains a duplicate key."
    )
    assert_condition(
        current_keys_set - original_keys_set == keys_to_add_set,
        f"The set difference between the current keys: {current_keys_set} "
        f"and the original keys: {original_keys_set} does not match "
        f"the expected keys to be added: {keys_to_add_set}.",
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
        original_keys = list(record.keys())

        for input_key, prompt_key in zip(self.input_keys, self.output_keys):
            record[prompt_key] = self.prompt_composer.compose(record[input_key])

        validate_added_keys(current_keys=list(record.keys()), original_keys=original_keys, keys_to_add=self.output_keys)
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
        original_keys = list(record.keys())

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

        validate_added_keys(current_keys=list(record.keys()), original_keys=original_keys, keys_to_add=self.output_keys)
        return record
