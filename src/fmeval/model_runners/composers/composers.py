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

    def _get_filled_in_template(self, placeholder_data_dict: Dict) -> str:
        """
        Returns the string that results from replacing keywords of placeholder_data_dict
        in self.template with corresponding value.

        :param data: Data to replace placeholder.
        :return: A template that has its placeholders "filled in".
        """
        return self.vanilla_template.substitute(**placeholder_data_dict)

    @abstractmethod
    def compose(self, data: Optional[str], placeholder_data_dict: Optional[Dict[str, str]]) -> Any:
        """
        Composes an object using the input data, self.vanilla_template, self.placeholder,
        and placeholder and data dictionary.

        :param data: The data used to compose a new object.
        :param placeholder_data_dict: The placeholder and original data dict used for composing.
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
            return json.loads(self._get_filled_in_template({self.placeholder: json.dumps(data)}))
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

    def compose(self, data: Optional[str] = None, placeholder_data_dict: Optional[Dict[str, str]] = {}) -> str:
        """
        Composes a prompt with data and/or from placeholder_data_dict that will be fed to an LLM.
        When both `data` and `placeholder_data_dict` are given and there are duplicates,
        the placeholders from placeholder_data_dict take precedence.
        Example:
            data = "London is the capital of"
            template =
                "<s>[INST] <<SYS>>Answer the following question in as few words as possible.<</SYS>>
                Question: $model_input [/INST]"
            composed prompt =
                "<s>[INST] <<SYS>>Answer the following question in as few words as possible.<</SYS>>
                Question: London is the capital of [/INST]"

        :param data: The original string that forms the basis of the returned prompt.
        :param placeholder_data_dict: The placeholder and original string dict.
        :return: A prompt composed by replacing self.placeholder in self.vanilla_template with `data`,
                and/or replacing keys of `placeholder_data_dict` with its corresponding value.
        """
        mapping_obj = {}
        if data:
            mapping_obj = {self.placeholder: data}
        mapping_obj.update(**placeholder_data_dict)
        return self._get_filled_in_template(placeholder_data_dict=mapping_obj)
