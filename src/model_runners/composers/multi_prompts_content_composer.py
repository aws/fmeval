from typing import List

from model_runners.composers.content_composer import ContentComposer


class MultiPromptsContentComposer(ContentComposer):
    KEYWORD = "prompts"

    def _compose(self, prompts: List[str]) -> str:
        # The placeholder $prompts is replaced by a list of JSON prompts. E.g.,
        # template: '{"data":$prompts}'
        # prompts: ['["John",40]','["Jane",30]']
        # result: '{"data":[["John",40],["Jane",30]],"names":["Name","Age"]}'

        return self.vanilla_template.substitute(**{self.KEYWORD: f'[{",".join(prompts)}]'})
