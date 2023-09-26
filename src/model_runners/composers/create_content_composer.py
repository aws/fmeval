import logging
from typing import Optional

from model_runners.composers.content_composer import ContentComposer
from model_runners.composers.template import VanillaTemplate
from model_runners.composers.single_prompt_content_composer import SinglePromptContentComposer
from model_runners.composers.multi_prompts_content_composer import MultiPromptsContentComposer
from infra.utils.sm_exceptions import CustomerError

logger = logging.getLogger(__name__)


def create_content_composer(template: str) -> Optional[ContentComposer]:
    composer: Optional[ContentComposer] = None
    vanilla_template = VanillaTemplate(template)

    if identifiers := vanilla_template.get_unique_identifiers():
        if SinglePromptContentComposer.KEYWORD in identifiers:
            composer = SinglePromptContentComposer(vanilla_template)
        elif MultiPromptsContentComposer.KEYWORD in identifiers:
            composer = MultiPromptsContentComposer(vanilla_template)
        else:
            logger.error(f"Found placeholders {identifiers} in template '{template}'.")
    else:
        logger.error(f"Could not find any identifier in template '{template}'.")

    if composer is None:
        raise CustomerError(
            "Invalid content_template. View job logs for details. "
            "The template must contain the placeholder $prompts for an array of prompts, "
            "or the placeholder $prompt for single prompt at a time."
        )
    return composer
