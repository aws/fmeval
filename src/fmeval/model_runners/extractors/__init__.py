from typing import Optional

from fmeval.constants import MIME_TYPE_JSON, JUMPSTART_MODEL_ID, JUMPSTART_MODEL_VERSION
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.extractors.json_extractor import JsonExtractor
from fmeval.model_runners.extractors.jumpstart_extractor import JumpStartExtractor


def create_extractor(
    model_accept_type: str = MIME_TYPE_JSON,
    output_location: Optional[str] = None,
    log_probability_location: Optional[str] = None,
    **kwargs,
):
    if model_accept_type == MIME_TYPE_JSON and (output_location is not None or log_probability_location is not None):
        extractor = JsonExtractor(
            output_jmespath_expression=output_location,
            log_probability_jmespath_expression=log_probability_location,
        )
    elif JUMPSTART_MODEL_ID in kwargs:
        extractor = JumpStartExtractor(
            jumpstart_model_id=kwargs[JUMPSTART_MODEL_ID],
            jumpstart_model_version=kwargs[JUMPSTART_MODEL_VERSION] if JUMPSTART_MODEL_VERSION in kwargs else "*",
        )
    else:  # pragma: no cover
        error_message = (
            f"Invalid accept type: {model_accept_type}."
            if model_accept_type is None
            else "One of output jmespath expression or log probability jmespath expression must be provided"
        )
        raise EvalAlgorithmClientError(error_message)

    return extractor
