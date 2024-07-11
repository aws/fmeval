import pytest

from fmeval.constants import MIME_TYPE_JSON
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.extractors import create_extractor, JumpStartExtractor


class TestCreateExtractor:
    """
    These tests are under integration tests instead of unit tests because
    credentials are required to call the JumpStart util function
    verify_model_region_and_return_specs.

    See test/unit/model_runners/extractors/test_create_extractor.py
    for corresponding unit tests.
    """

    def test_create_extractor_jumpstart(self):
        """
        GIVEN a model whose default payloads are not found at the top level of
            the model spec, but instead nested under the inference_configs attribute.
        WHEN create_extractor is called with this model id.
        THEN a JumpStartExtractor is successfully created for this model.
        """
        # default payloads found in inference_component_configs
        jumpstart_model_id = "huggingface-llm-mistral-7b"
        assert isinstance(
            create_extractor(model_accept_type=MIME_TYPE_JSON, jumpstart_model_id=jumpstart_model_id),
            JumpStartExtractor,
        )

    def test_create_extractor_jumpstart_no_default_payloads(self):
        """
        GIVEN a model whose spec does not contain default payloads data anywhere.
        WHEN a create_extractor is called with this model id.
        THEN the correct exception is raised.
        """
        with pytest.raises(
            EvalAlgorithmClientError,
            match="JumpStart Model: xgboost-regression-snowflake is not supported at this time",
        ):
            create_extractor(model_accept_type=MIME_TYPE_JSON, jumpstart_model_id="xgboost-regression-snowflake")
