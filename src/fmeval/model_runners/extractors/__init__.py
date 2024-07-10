from typing import Optional

from sagemaker.jumpstart.enums import JumpStartModelType

from fmeval.constants import (
    MIME_TYPE_JSON,
    JUMPSTART_MODEL_ID,
    JUMPSTART_MODEL_VERSION,
    JUMPSTART_MODEL_TYPE,
    IS_EMBEDDING_MODEL,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.extractors.json_extractor import JsonExtractor
from fmeval.model_runners.extractors.jumpstart_extractor import JumpStartExtractor


def create_extractor(
    model_accept_type: str = MIME_TYPE_JSON,
    output_location: Optional[str] = None,
    log_probability_location: Optional[str] = None,
    embedding_location: Optional[str] = None,
    **kwargs,
):
    if model_accept_type == MIME_TYPE_JSON and (
        output_location is not None or log_probability_location is not None or embedding_location is not None
    ):
        extractor = JsonExtractor(
            output_jmespath_expression=output_location,
            log_probability_jmespath_expression=log_probability_location,
            embedding_jmespath_expression=embedding_location,
        )
    elif JUMPSTART_MODEL_ID in kwargs:
        extractor = JumpStartExtractor(
            jumpstart_model_id=kwargs[JUMPSTART_MODEL_ID],
            jumpstart_model_version=kwargs[JUMPSTART_MODEL_VERSION] if JUMPSTART_MODEL_VERSION in kwargs else "*",
            jumpstart_model_type=kwargs[JUMPSTART_MODEL_TYPE]
            if JUMPSTART_MODEL_TYPE in kwargs
            else JumpStartModelType.OPEN_WEIGHTS,
            is_embedding_model=kwargs[IS_EMBEDDING_MODEL] if IS_EMBEDDING_MODEL in kwargs else False,
        )
    else:  # pragma: no cover
        error_message = (
            f"Invalid accept type: {model_accept_type}."
            if model_accept_type is None
            else "One of output jmespath expression, log probability or embedding jmespath expression must be provided"
        )
        raise EvalAlgorithmClientError(error_message)

    return extractor
