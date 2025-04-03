import pytest

from fmeval.constants import MIME_TYPE_JSON
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.extractors import create_extractor, JsonExtractor, JumpStartExtractor
from sagemaker.jumpstart.enums import JumpStartModelType


def test_create_extractor():
    assert isinstance(create_extractor(model_accept_type=MIME_TYPE_JSON, output_location="output"), JsonExtractor)


@pytest.mark.parametrize(
    "jumpstart_model_id", ["huggingface-llm-falcon-7b-bf16"]  # default payloads found top level of model spec
)
def test_create_extractor_jumpstart(jumpstart_model_id):
    """
    Note: the test case for a model whose default payloads are found in inference_configs
        (instead of as a top-level attribute of the model spec) is an integration test,
        since unit tests don't run with the credentials required.
    """
    assert isinstance(
        create_extractor(model_accept_type=MIME_TYPE_JSON, jumpstart_model_id=jumpstart_model_id),
        JumpStartExtractor,
    )


def test_create_extractor_jumpstart_proprietary():
    assert isinstance(
        create_extractor(
            model_accept_type=MIME_TYPE_JSON,
            jumpstart_model_id="ai21-summarization",
            jumpstart_model_type=JumpStartModelType.PROPRIETARY,
        ),
        JumpStartExtractor,
    )


def test_create_extractor_exception():
    with pytest.raises(
        EvalAlgorithmClientError,
        match="One of output jmespath expression, log probability or embedding jmespath expression must be provided",
    ):
        assert isinstance(create_extractor(model_accept_type=MIME_TYPE_JSON), JsonExtractor)
