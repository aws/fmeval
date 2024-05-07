import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List, Optional
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.composers.template import VanillaTemplate


class Composer(ABC):
    def __init__(self, template: str, placeholder: str):
        """
        :param template: A template string. Ex: '{"data":$prompt}'
        :param placeholder: A placeholder keyword. This keyword appears
            in `template` with a $ sign prepended. In the above example,
            the placeholder is "prompt".
        """
        self.placeholder = placeholder
        self.vanilla_template = VanillaTemplate(template)

    def _get_filled_in_template(self, data: Optional[str], **kwargs) -> str:
        """
        Returns the string that results from replacing self.placeholder
        in self.template with `data`. When both data and kwargs are given and there are duplicates,
        the placeholders from kwargs take precedence.

        :param data: Data to replace placeholder.
        :return: A template that has its placeholders "filled in".
        """
        mapping_obj = {}
        if data:
            mapping_obj = {self.placeholder: data}
        mapping_obj.update(**kwargs)
        return self.vanilla_template.substitute(**mapping_obj)

    @abstractmethod
    def compose(self, data: Optional[str], **kwargs) -> Any:
        """
        Composes an object using the input data, self.vanilla_template, and self.placeholder.

        :param data: The data used to compose a new object.
        :param **kwargs: Arbitrary keyword arguments used for composing.
        :return: A new object composed using `data`, self.vanilla_template, and self.placeholder.
        """


# mypy: ignore-errors
class JsonContentComposer(Composer):
    """
    Composer for models that expect a JSON payload, i.e. models
    with content_type == "application/json".
    """

    PLACEHOLDER = "prompt"

    def __init__(self, template: str):
        super().__init__(template=template, placeholder=self.PLACEHOLDER)

    def compose(self, data: str) -> Union[str, List, Dict]:
        """
        The placeholder $prompt is replaced by a single JSON prompt. E.g.,
        template: '{"data": $prompt}'
        data:     "[\"John\",40]"
        result:   {"data": "[\"John\",40]"}
        This composer uses json.dumps to make sure the double quotes included are properly escaped.

        :param data: The data used to replace self.placeholder in self.vanilla_template.
        :return: A JSON object representing a prompt that will be consumed by a model.
        """
        try:
            return json.loads(self._get_filled_in_template(json.dumps(data)))
        except Exception as e:
            raise EvalAlgorithmClientError(
                f"Unable to load a JSON object with template '{self.vanilla_template.template}' using data {data} ",
                e,
            )


class PromptComposer(Composer):
    """
    Composes LLM prompt inputs.
    """

    PLACEHOLDER = "model_input"

    def __init__(self, template: str):
        super().__init__(template=template, placeholder=self.PLACEHOLDER)

    def compose(self, data: Optional[str] = None, **kwargs: Any) -> str:
        """
        Composes a prompt with data and/or from keyword arguments that will be fed to an LLM.
        Example:
            data = "London is the capital of"
            template =
                "<s>[INST] <<SYS>>Answer the following question in as few words as possible.<</SYS>>
                Question: $model_input [/INST]"
            composed prompt =
                "<s>[INST] <<SYS>>Answer the following question in as few words as possible.<</SYS>>
                Question: London is the capital of [/INST]"

        :param data: The original string that forms the basis of the returned prompt.
        :param **kwargs: Keyword arguments that used for filling in template.
        :return: A prompt composed by replacing self.placeholder in self.vanilla_template with `data`.
        """
        return self._get_filled_in_template(data, **kwargs)
