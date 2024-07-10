import pytest

from fmeval.constants import MIME_TYPE_JSON
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.extractors import create_extractor, JsonExtractor, JumpStartExtractor


def test_create_extractor():
    assert isinstance(create_extractor(model_accept_type=MIME_TYPE_JSON, output_location="output"), JsonExtractor)


@pytest.mark.parametrize(
    "jumpstart_model_id",
    [
        "huggingface-llm-falcon-7b-bf16",  # default payloads found top level of model spec
        "huggingface-llm-mistral-7b",  # default payloads found in inference_component_configs
    ],
)
def test_create_extractor_jumpstart(jumpstart_model_id):
    assert isinstance(
        create_extractor(model_accept_type=MIME_TYPE_JSON, jumpstart_model_id=jumpstart_model_id),
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
