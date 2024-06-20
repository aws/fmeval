import logging
from typing import Optional

import fmeval.util as util
from fmeval.constants import MIME_TYPE_JSON, JUMPSTART_MODEL_ID, JUMPSTART_MODEL_VERSION, IS_EMBEDDING_MODEL
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.composers.composers import Composer, JsonContentComposer
from fmeval.model_runners.composers.jumpstart_composer import JumpStartComposer
from fmeval.model_runners.composers.template import VanillaTemplate

logger = logging.getLogger(__name__)


def create_content_composer(template: Optional[str] = None, content_type: str = MIME_TYPE_JSON, **kwargs) -> Composer:
    composer: Optional[Composer] = None
    if content_type == MIME_TYPE_JSON and template is not None:
        util.require(template, "Content template must be provided for JSON content type")
        vanilla_template = VanillaTemplate(template)  # type: ignore

        if identifiers := vanilla_template.get_unique_identifiers():
            if JsonContentComposer.PLACEHOLDER in identifiers:
                composer = JsonContentComposer(template=template)  # type: ignore
            else:
                logger.error(f"Found placeholders {identifiers} in template '{template}'.")
        else:
            logger.error(f"Could not find any identifier in template '{template}'.")
    elif JUMPSTART_MODEL_ID in kwargs:
        composer = JumpStartComposer(
            jumpstart_model_id=kwargs[JUMPSTART_MODEL_ID],
            jumpstart_model_version=kwargs[JUMPSTART_MODEL_VERSION] if JUMPSTART_MODEL_VERSION in kwargs else "*",
            is_embedding_model=kwargs[IS_EMBEDDING_MODEL] if IS_EMBEDDING_MODEL in kwargs else False,
        )
    else:  # pragma: no cover
        raise EvalAlgorithmClientError(f"Invalid accept type: {content_type} ")

    if composer is None:
        raise EvalAlgorithmClientError("Invalid input - unable to create a content composer")
    return composer
