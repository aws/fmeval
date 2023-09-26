import abc
import json

from typing import List, Dict, Union

from model_runners.composers.template import VanillaTemplate
from infra.utils.sm_exceptions import CustomerError


class ContentComposer(abc.ABC):
    def __init__(self, template: VanillaTemplate):
        self.vanilla_template = template

    def compose(self, prompts: Union[List[str], str]) -> Union[Dict, List]:
        """
        Compose content using the provided prompts and the template.

        :param prompts: List of prompts to be used for composition.
        :return: Composed content as a JSON object.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        substituted_content = self._compose(prompts)
        try:
            return json.loads(substituted_content)
        except Exception as e:
            raise CustomerError(
                f"Unable to load a JSON object with content_template '{self.vanilla_template.template}'. ", e
            )

    @abc.abstractmethod
    def _compose(self, prompts: List[str]) -> str:
        """
        Compose content using the provided prompts and the template.

        :param prompts: List of prompts to be used for composition.
        :return: Composed content as a string.
        """
