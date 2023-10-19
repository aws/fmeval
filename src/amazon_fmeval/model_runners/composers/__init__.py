import logging
from typing import Optional

import amazon_fmeval.util as util
from amazon_fmeval.constants import MIME_TYPE_JSON, MIME_TYPE_X_TEXT
from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.composers.composers import Composer, JsonContentComposer, StringContentComposer
from amazon_fmeval.model_runners.composers.template import VanillaTemplate

logger = logging.getLogger(__name__)

MIME_TYPE_TO_CONTENT_COMPOSER = {MIME_TYPE_JSON: JsonContentComposer, MIME_TYPE_X_TEXT: StringContentComposer}


def create_content_composer(template: Optional[str] = None, content_type: str = MIME_TYPE_JSON) -> Composer:
    composer: Optional[Composer] = None
    if content_type == MIME_TYPE_JSON or content_type == MIME_TYPE_X_TEXT:
        util.require(template, f"Content template must be provided for {content_type} content type")
        vanilla_template = VanillaTemplate(template)  # type: ignore
        if identifiers := vanilla_template.get_unique_identifiers():
            composer_class = MIME_TYPE_TO_CONTENT_COMPOSER[content_type]
            if composer_class.PLACEHOLDER in identifiers:
                composer = composer_class(template=template)  # type: ignore
            else:
                logger.error(f"Found placeholders {identifiers} in template '{template}'.")
        else:
            logger.error(f"Could not find any identifier in template '{template}'.")
    else:  # pragma: no cover
        raise EvalAlgorithmClientError(f"Invalid accept type: {content_type} ")

    if composer is None:
        raise EvalAlgorithmClientError("Invalid input - unable to create a content composer")
    return composer
