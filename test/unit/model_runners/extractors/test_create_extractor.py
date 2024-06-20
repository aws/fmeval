import pytest

from fmeval.constants import MIME_TYPE_JSON
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.extractors import create_extractor, JsonExtractor, JumpStartExtractor


def test_create_extractor():
    assert isinstance(create_extractor(model_accept_type=MIME_TYPE_JSON, output_location="output"), JsonExtractor)


def test_create_extractor_jumpstart():
    assert isinstance(
        create_extractor(model_accept_type=MIME_TYPE_JSON, jumpstart_model_id="huggingface-llm-falcon-7b-bf16"),
        JumpStartExtractor,
    )


def test_create_extractor_jumpstart_proprietary():
    assert isinstance(
        create_extractor(model_accept_type=MIME_TYPE_JSON, jumpstart_model_id="cohere-gpt-medium"),
        JumpStartExtractor,
    )


def test_create_extractor_exception():
    with pytest.raises(
        EvalAlgorithmClientError,
        match="One of output jmespath expression, log probability or embedding jmespath expression must be provided",
    ):
        assert isinstance(create_extractor(model_accept_type=MIME_TYPE_JSON), JsonExtractor)
