import logging
from typing import Optional

import util
from constants import MIME_TYPE_JSON
from exceptions import EvalAlgorithmClientError
from model_runners.composers.composers import Composer, ContentComposer
from model_runners.composers.template import VanillaTemplate

logger = logging.getLogger(__name__)


def create_content_composer(template: Optional[str] = None, content_type: str = MIME_TYPE_JSON) -> Composer:
    composer: Optional[Composer] = None
    if content_type == MIME_TYPE_JSON:
        util.require(template, "Content template must be provided for JSON content type")
        vanilla_template = VanillaTemplate(template)  # type: ignore

        if identifiers := vanilla_template.get_unique_identifiers():
            if ContentComposer.KEYWORD in identifiers:
                composer = ContentComposer(template=template)  # type: ignore
            else:
                logger.error(f"Found placeholders {identifiers} in template '{template}'.")
        else:
            logger.error(f"Could not find any identifier in template '{template}'.")
    else:  # pragma: no cover
        raise EvalAlgorithmClientError(f"Invalid accept type: {content_type} ")

    if composer is None:
        raise EvalAlgorithmClientError("Invalid input - unable to create a content composer")
    return composer
