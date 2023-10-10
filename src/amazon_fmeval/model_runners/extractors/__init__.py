from typing import Optional

from amazon_fmeval.constants import MIME_TYPE_JSON
from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.extractors.json_extractor import JsonExtractor


def create_extractor(
    model_accept_type: str = MIME_TYPE_JSON,
    output_location: Optional[str] = None,
    log_probability_location: Optional[str] = None,
):
    if model_accept_type == MIME_TYPE_JSON:
        extractor = JsonExtractor(
            output_jmespath_expression=output_location,
            log_probability_jmespath_expression=log_probability_location,
        )
    else:  # pragma: no cover
        raise EvalAlgorithmClientError(f"Invalid accept type: {MIME_TYPE_JSON}.")

    return extractor
