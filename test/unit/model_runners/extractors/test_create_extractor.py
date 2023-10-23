import pytest

from amazon_fmeval import EvalAlgorithmClientError
from amazon_fmeval.constants import MIME_TYPE_JSON
from amazon_fmeval.model_runners.extractors import create_extractor, JsonExtractor, JumpStartExtractor


def test_create_extractor():
    assert isinstance(create_extractor(model_accept_type=MIME_TYPE_JSON, output_location="output"), JsonExtractor)


def test_create_extractor_jumpstart():
    assert isinstance(
        create_extractor(model_accept_type=MIME_TYPE_JSON, jumpstart_model_id="model_id"), JumpStartExtractor
    )


def test_create_extractor_exception():
    with pytest.raises(
        EvalAlgorithmClientError,
        match="One of output jmespath expression or log probability jmespath expression must be provided",
    ):
        assert isinstance(create_extractor(model_accept_type=MIME_TYPE_JSON), JsonExtractor)
