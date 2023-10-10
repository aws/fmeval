import abc
import json

from typing import Dict

import amazon_fmeval.util as util
from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.composers.template import VanillaTemplate


class Composer(abc.ABC):
    def __init__(self, template: str, keyword: str):
        self.keyword = keyword
        self.vanilla_template = VanillaTemplate(template)

    def compose(self, prompt: str) -> str:
        """
        Compose content using the provided prompt and the template.

        :param prompts: List of prompts to be used for composition.
        :return: Composed content as a JSON object.
        """
        util.require(isinstance(prompt, str), "Prompt must be an instance of string")
        return self.vanilla_template.substitute(**{self.keyword: prompt})


# mypy: ignore-errors
class ContentComposer(Composer):
    KEYWORD = "prompt"

    def __init__(self, template: str):
        super().__init__(template=template, keyword=self.KEYWORD)

    def compose(self, prompt: str) -> Dict:
        # The placeholder $prompt is replaced by a single JSON prompt. E.g.,
        # template: '{"data":$prompt}'
        # prompts: ['["John",40]']
        # result: '{"data":["John",40],"names":["Name","Age"]}'
        try:
            return json.loads(Composer.compose(self, prompt))
        except Exception as e:
            raise EvalAlgorithmClientError(
                f"Unable to load a JSON object with content_template '{self.vanilla_template.template}' for prompt {prompt} ",
                e,
            )


class PromptComposer(Composer):
    KEYWORD = "feature"

    def __init__(self, template: str):
        super().__init__(template=template, keyword=self.KEYWORD)
