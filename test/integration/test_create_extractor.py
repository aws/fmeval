from fmeval.constants import MIME_TYPE_JSON
from fmeval.model_runners.extractors import create_extractor, JumpStartExtractor


def test_create_extractor_jumpstart():
    """
    GIVEN a model whose default payloads are not found at the top level of
        the model spec, but instead nested under the inference_configs attribute.
    WHEN create_extractor is called with this model id.
    THEN a JumpStartExtractor is successfully created for this model.

    This test case is under integration tests instead of unit tests because
    credentials are required to call the JumpStart util function
    verify_model_region_and_return_specs.

    See test/unit/model_runners/extractors/test_create_extractor.py
    for corresponding unit tests.
    """
    # default payloads found in inference_component_configs
    jumpstart_model_id = "huggingface-llm-mistral-7b"
    assert isinstance(
        create_extractor(model_accept_type=MIME_TYPE_JSON, jumpstart_model_id=jumpstart_model_id),
        JumpStartExtractor,
    )
