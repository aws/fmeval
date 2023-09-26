from typing import List

from model_runners.composers.content_composer import ContentComposer


class SinglePromptContentComposer(ContentComposer):
    KEYWORD = "prompt"

    def _compose(self, prompts: List[str]) -> str:
        # The placeholder $prompt is replaced by a single JSON prompt. E.g.,
        # template: '{"data":$prompt}'
        # prompts: ['["John",40]']
        # result: '{"data":["John",40],"names":["Name","Age"]}'
        assert (
            len(prompts) == 1
        ), f"content_template with ${self.KEYWORD} placeholder can only handle single prompt at a time"
        return self.vanilla_template.substitute(**{self.KEYWORD: prompts[0]})
