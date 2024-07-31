import numpy as np
from typing import Any, Dict, List, Optional

from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.transform import Transform
from fmeval.transforms.util import validate_call


class GeneratePrompt(Transform):
    """This transform augments an input record with LLM prompts constructed according to a template.

    If multiple input keys are provided, this transform creates prompts out of all of them,
    applying the same prompt template to each input.
    """

    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        prompt_template: str,
        placeholder_to_record_key: Optional[Dict[str, str]] = None,
    ):
        """GeneratePrompt initializer.

                :param input_keys: The keys corresponding to the text that will be used to create prompts.
                    If multiple input keys are provided, a prompt will be constructed from each input, but
                    the created prompts will all utilize the same prompt template.
                :param output_keys: The keys corresponding to the prompts that get added by this Transform.
                :param prompt_template: The template used to construct the prompt.
                    Example: "Summarize the following text: $model_input".
                :param placeholder_to_record_key: The placeholders and the corresponding record keys dict.
                    Note that when using `placeholder_to_record_key`, having input keys or more than one output key
                    doesn't make much sense, as all composed prompts will be identical.
                    Example:
                        Inputs:
                            prompt_template = "Summarize $x and $y"
                            input_keys = []
                            output_keys = ["my_prompt"]
                            placeholder_to_record_key = {"x": "statement_1", "y": "statement_2"}
                            record = {"statement_1": "some long text", "statement_2": "some other long text"}
                        Output record (only new keys and values are shown):
                            {"my_prompt": "Summarize some long text and some other long text"}

        Output record (only new keys and values are shown):
        {"my_prompt": "Summarize some long text and some other long text"}
        """
        super().__init__(input_keys, output_keys, prompt_template, placeholder_to_record_key)
        input_keys_to_register = list(placeholder_to_record_key.values()) if placeholder_to_record_key else input_keys
        self.register_input_output_keys(input_keys_to_register, output_keys)
        self.placeholder_to_record_key = placeholder_to_record_key
        self.prompt_template = prompt_template
        self.prompt_composer = PromptComposer(prompt_template)

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with LLM prompts and returns said record.

        :param record: The input record.
        :returns: The input record with prompts added in.
        """
        if self.placeholder_to_record_key is not None:
            placeholder_data_dict = {
                placeholder_key: record[self.placeholder_to_record_key[placeholder_key]]
                for placeholder_key in self.placeholder_to_record_key
            }
            for prompt_key in self.output_keys:
                record[prompt_key] = self.prompt_composer.compose(placeholder_data_dict=placeholder_data_dict)
        else:
            for input_key, prompt_key in zip(self.input_keys, self.output_keys):
                record[prompt_key] = self.prompt_composer.compose(record[input_key])
        return record


class GetModelOutputs(Transform):
    """Invokes a ModelRunner's `predict` method and augments the input record with the model output.

    An instance of this transform can be configured to get model outputs for multiple inputs.
    See __init__ docstring for more details.
    """

    def __init__(
        self,
        input_to_output_keys: Dict[str, List[str]],
        model_runner: ModelRunner,
    ):
        """GetModelOutputs initializer.

        :param input_to_output_keys: Maps an input key (corresponding to
            the input payload to the model) to a list of output keys, where
            each output key corresponds to the model output that is returned
            when calling the `predict` method of `model_runner` on the input.

            Note that the reason a list of output keys is used (as opposed to
            a singular key) is so that `model_runner` can be invoked on the
            same input multiple times.

            Note that the response payload from calling `predict` will be a tuple of the
            form (model_output, log_probability), and this transform is only concerned with
            the model_output element.
        :param model_runner: The ModelRunner instance whose outputs will be obtained.
        """
        super().__init__(input_to_output_keys, model_runner)
        self.register_input_output_keys(
            input_keys=list(input_to_output_keys.keys()),
            output_keys=[
                output_key for output_key_list in input_to_output_keys.values() for output_key in output_key_list
            ],
        )
        self.input_to_output_keys = input_to_output_keys
        self.model_runner = model_runner

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with model outputs and return said record.

        :param record: The input record.
        :returns: The input record with model output data added in.
        """
        for input_key, output_keys in self.input_to_output_keys.items():
            for output_key in output_keys:
                model_output, _ = self.model_runner.predict(record[input_key])
                record[output_key] = model_output
        return record


class GetLogProbabilities(Transform):
    """Invokes a ModelRunner's `predict` method and augments the input record with the returned log probability.

    This transform can obtain multiple log probabilities, by invoking the provided model on multiple inputs.
    See the __init__ docstring for more details.
    """

    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        model_runner: ModelRunner,
    ):
        """GetModelOutputs initializer.

        Note that the ith element of input_keys should correspond to the ith element of
        output_keys. In other words, the log probability obtained from invoking the model
        on the input with key input_keys[i] will be assigned the key output_keys[i].

        :param input_keys: The keys within the input record corresponding to model inputs.
        :param output_keys: The keys corresponding to the log probability data that will get
            added to the record by this transform.
        :param model_runner: The ModelRunner instance whose `predict` method wil be invoked
            to obtain the log probability.
        """
        super().__init__(input_keys, output_keys, model_runner)
        self.register_input_output_keys(
            input_keys=input_keys,
            output_keys=output_keys,
        )
        self.model_runner = model_runner

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with the log probability that is returned by the model.

        :param record: The input record.
        :returns: The input record with log probability data added in.
        """
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            _, log_prob = self.model_runner.predict(record[input_key])
            record[output_key] = log_prob
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
