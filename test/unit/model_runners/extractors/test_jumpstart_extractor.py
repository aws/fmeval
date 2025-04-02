from unittest import mock
from unittest.mock import patch, Mock

import pytest
from _pytest.fixtures import fixture
from _pytest.python_api import approx
from sagemaker.jumpstart.enums import JumpStartModelType

from fmeval.exceptions import EvalAlgorithmClientError, EvalAlgorithmInternalError
from fmeval.model_runners.extractors.jumpstart_extractor import JumpStartExtractor

EXAMPLE_JUMPSTART_RESPONSE = [
    {
        "generated_text": "\n\nI am trying",
        "details": {
            "finish_reason": "length",
            "generated_tokens": 5,
            "seed": None,
            "prefill": [
                {"id": 1, "text": "<s>", "logprob": None},
                {"id": 22557, "text": "Hello", "logprob": -6.7734375},
                {"id": 28747, "text": ":", "logprob": -6.6210938},
            ],
            "tokens": [
                {"id": 13, "text": "\n", "logprob": -0.25634766, "special": False},
                {"id": 13, "text": "\n", "logprob": -0.19262695, "special": False},
                {"id": 28737, "text": "I", "logprob": -0.64501953, "special": False},
                {"id": 837, "text": " am", "logprob": -1.3876953, "special": False},
                {"id": 2942, "text": " trying", "logprob": -1.9433594, "special": False},
            ],
            "top_tokens": [
                [
                    {"id": 13, "text": "\n", "logprob": -0.25634766, "special": False},
                    {"id": 315, "text": "I", "logprob": -2.6308594, "special": False},
                    {"id": 28731, "text": ")", "logprob": -3.4667969, "special": False},
                ],
                [
                    {"id": 13, "text": "\n", "logprob": -0.19262695, "special": False},
                    {"id": 28737, "text": "I", "logprob": -2.3183594, "special": False},
                    {"id": 5183, "text": "My", "logprob": -4.3007812, "special": False},
                ],
                [
                    {"id": 28737, "text": "I", "logprob": -0.64501953, "special": False},
                    {"id": 5183, "text": "My", "logprob": -3.1132812, "special": False},
                    {"id": 2324, "text": "We", "logprob": -3.2617188, "special": False},
                ],
                [
                    {"id": 837, "text": "am", "logprob": -1.3876953, "special": False},
                    {"id": 506, "text": "have", "logprob": -1.4345703, "special": False},
                    {"id": 28742, "text": "'", "logprob": -1.9658203, "special": False},
                ],
                [
                    {"id": 2942, "text": "trying", "logprob": -1.9433594, "special": False},
                    {"id": 264, "text": "a", "logprob": -2.4433594, "special": False},
                    {"id": 633, "text": "new", "logprob": -2.5449219, "special": False},
                ],
            ],
        },
    }
]
EMBEDDING_MODEL_ID = "tcembedding-model-id"
EXAMPLE_JUMPSTART_EMBEDDING_RESPONSE = {"embedding": [0.1, 0.2, 0.1]}


class TestJumpStartExtractor:
    @fixture(scope="module")
    @patch("sagemaker.session.Session")
    def extractor(self, sagemaker_session):
        sagemaker_session.boto_region_name = "us-west-2"
        return JumpStartExtractor(
            jumpstart_model_id="huggingface-llm-falcon-7b-bf16",
            jumpstart_model_version="*",
            jumpstart_model_type=JumpStartModelType.OPEN_WEIGHTS,
            sagemaker_session=sagemaker_session,
        )

    @fixture(scope="module")
    def embedding_model_extractor(self):
        return JumpStartExtractor(
            jumpstart_model_id=EMBEDDING_MODEL_ID,
            jumpstart_model_version="*",
            jumpstart_model_type=JumpStartModelType.OPEN_WEIGHTS,
            is_embedding_model=True,
        )

    def test_log_probability_fails_for_batch_request(self, extractor):
        with pytest.raises(AssertionError, match="Jumpstart extractor does not support batch requests"):
            extractor.extract_log_probability({"log_pob": 0.8}, 2)

    def test_output_fails_for_batch_request(self, extractor):
        with pytest.raises(AssertionError, match="Jumpstart extractor does not support batch requests"):
            extractor.extract_output({"log_pob": 0.8}, 2)

    def test_embedding_fails_for_batch_request(self, embedding_model_extractor):
        with pytest.raises(AssertionError, match="Jumpstart extractor does not support batch requests"):
            embedding_model_extractor.extract_embedding({"embedding": [0.1, 0.2, 0.1]}, 2)

    def test_extract_output(self, extractor):
        assert extractor.extract_output(EXAMPLE_JUMPSTART_RESPONSE) == "\n\nI am trying"

    def test_extract_output_failure(self, extractor):
        response = {"response": "Hey! I am good. How are you?"}

        with pytest.raises(EvalAlgorithmClientError, match="Unable to extract output from Jumpstart model:"):
            extractor.extract_output(response)

    def test_log_probability(self, extractor):
        assert extractor.extract_log_probability(EXAMPLE_JUMPSTART_RESPONSE) == approx(-13.3945313)

    def test_log_probability_missing_log_prob(self, extractor):
        with pytest.raises(EvalAlgorithmClientError, match="Unable to extract log probability from Jumpstart model:"):
            extractor.extract_log_probability({})

    def test_extract_embedding(self, embedding_model_extractor):
        assert embedding_model_extractor.extract_embedding(EXAMPLE_JUMPSTART_EMBEDDING_RESPONSE) == [0.1, 0.2, 0.1]

    def test_extract_embedding_failure(self, embedding_model_extractor):
        response = {"response": "sample response"}
        with pytest.raises(EvalAlgorithmClientError, match="Unable to extract embedding from Jumpstart model:"):
            embedding_model_extractor.extract_embedding(response)

    @patch("sagemaker.session.Session")
    def test_extractor_with_bad_output_expression(self, sagemaker_session):
        sagemaker_session.boto_region_name = "us-west-2"
        bad_output_expression = {"default_payloads": {"test": {"output_keys": {"generated_text": "{"}}}}
        with patch(
            "fmeval.model_runners.extractors.jumpstart_extractor.JumpStartExtractor.get_jumpstart_sdk_spec",
            return_value=bad_output_expression,
        ), pytest.raises(
            EvalAlgorithmInternalError,
            match="Output jmespath expression found for Jumpstart model huggingface-llm-falcon-7b-bf16 is not valid "
            "jmespath. Please provide output jmespath to the JumpStartModelRunner",
        ):
            JumpStartExtractor(
                jumpstart_model_id="huggingface-llm-falcon-7b-bf16",
                jumpstart_model_version="*",
                jumpstart_model_type=JumpStartModelType.OPEN_WEIGHTS,
                sagemaker_session=sagemaker_session,
            )

    @patch("sagemaker.session.Session")
    def test_extractor_with_bad_input_log_probability(self, sagemaker_session):
        sagemaker_session.boto_region_name = "us-west-2"
        bad_output_expression = {
            "default_payloads": {"test": {"output_keys": {"generated_text": "generated_text", "input_logprobs": "{"}}}
        }
        with patch(
            "fmeval.model_runners.extractors.jumpstart_extractor.JumpStartExtractor.get_jumpstart_sdk_spec",
            return_value=bad_output_expression,
        ), pytest.raises(
            EvalAlgorithmInternalError,
            match="Input log probability jmespath expression found for Jumpstart model huggingface-llm-falcon-7b-bf16 "
            "is not valid jmespath. Please provide correct input log probability jmespath to the "
            "JumpStartModelRunner",
        ):
            JumpStartExtractor(
                jumpstart_model_id="huggingface-llm-falcon-7b-bf16",
                jumpstart_model_version="*",
                jumpstart_model_type=JumpStartModelType.OPEN_WEIGHTS,
                sagemaker_session=sagemaker_session,
            )

    @patch("sagemaker.session.Session")
    def test_extractor_with_invalid_default_payload(self, sagemaker_session):
        sagemaker_session.boto_region_name = "us-west-2"

        exception_mock = mock.Mock()
        exception_mock.side_effect = TypeError
        with patch("jmespath.parser.ParsedResult.search", side_effect=exception_mock), pytest.raises(
            EvalAlgorithmInternalError,
            match="Unable find the generated_text key in the default payload for JumpStart model",
        ):
            JumpStartExtractor(
                jumpstart_model_id="huggingface-llm-falcon-7b-bf16",
                jumpstart_model_version="*",
                jumpstart_model_type=JumpStartModelType.OPEN_WEIGHTS,
                sagemaker_session=sagemaker_session,
            )

    @patch("sagemaker.session.Session")
    def test_extractor_when_default_payloads_found_in_inference_configs(self, sagemaker_session):
        sagemaker_session.boto_region_name = "us-west-2"
        model_spec_json = {}
        model_spec_js_object = Mock()
        inference_configs = Mock()
        top_config = Mock()

        resolved_metadata_config = {"default_payloads": {"test": {"output_keys": {"generated_text": "Hi"}}}}
        top_config.resolved_metadata_config = resolved_metadata_config
        inference_configs.get_top_config_from_ranking.return_value = top_config
        model_spec_js_object.inference_configs = inference_configs

        with patch(
            "fmeval.model_runners.extractors.jumpstart_extractor.JumpStartExtractor.get_jumpstart_sdk_spec",
            return_value=model_spec_json,
        ), patch(
            "fmeval.model_runners.extractors.jumpstart_extractor.verify_model_region_and_return_specs",
            return_value=model_spec_js_object,
        ):
            JumpStartExtractor(
                jumpstart_model_id="huggingface-llm-falcon-7b-bf16",
                jumpstart_model_version="*",
                jumpstart_model_type=JumpStartModelType.OPEN_WEIGHTS,
                sagemaker_session=sagemaker_session,
            )

    @patch("sagemaker.session.Session")
    def test_extractor_when_model_spec_is_missing_inference_configs(self, sagemaker_session):
        """
        GIVEN a model whose spec does not contain default_payloads as a top-level attribute
        WHEN the `inference_configs` field of the JumpStartModelSpecs object returned by
            verify_model_region_and_return_specs is None
        THEN the correct exception is raised.
        """
        sagemaker_session.boto_region_name = "us-west-2"
        model_spec_json = {"not_default_payloads": {"test": {"output_keys": {"generated_text": "{"}}}}
        model_spec_js_object = Mock()
        model_spec_js_object.inference_configs = None

        with patch(
            "fmeval.model_runners.extractors.jumpstart_extractor.JumpStartExtractor.get_jumpstart_sdk_spec",
            return_value=model_spec_json,
        ), patch(
            "fmeval.model_runners.extractors.jumpstart_extractor.verify_model_region_and_return_specs",
            return_value=model_spec_js_object,
        ), pytest.raises(
            EvalAlgorithmClientError,
            match="JumpStart Model: huggingface-llm-falcon-7b-bf16 is not supported at this time",
        ):
            JumpStartExtractor(
                jumpstart_model_id="huggingface-llm-falcon-7b-bf16",
                jumpstart_model_version="*",
                jumpstart_model_type=JumpStartModelType.OPEN_WEIGHTS,
                sagemaker_session=sagemaker_session,
            )

    @patch("sagemaker.session.Session")
    def test_extractor_when_default_payloads_found_outside_inference_configs(self, sagemaker_session):
        sagemaker_session.boto_region_name = "us-west-2"
        model_spec_json = {"default_payloads": {"test": {"output_keys": {"generated_text": "Hi"}}}}

        with patch(
            "fmeval.model_runners.extractors.jumpstart_extractor.JumpStartExtractor.get_jumpstart_sdk_spec",
            return_value=model_spec_json,
        ):
            JumpStartExtractor(
                jumpstart_model_id="huggingface-llm-falcon-7b-bf16",
                jumpstart_model_version="*",
                jumpstart_model_type=JumpStartModelType.OPEN_WEIGHTS,
                sagemaker_session=sagemaker_session,
            )

    @patch("sagemaker.session.Session")
    def test_extractor_when_default_payloads_is_empty(self, sagemaker_session):
        """
        GIVEN the default payloads data that is found is empty.
        WHEN a JumpStartExtractor is being initialized.
        THEN the correct exception is raised.
        """
        sagemaker_session.boto_region_name = "us-west-2"
        model_spec_json = {"default_payloads": None}

        with patch(
            "fmeval.model_runners.extractors.jumpstart_extractor.JumpStartExtractor.get_jumpstart_sdk_spec",
            return_value=model_spec_json,
        ), pytest.raises(
            EvalAlgorithmClientError,
            match="JumpStart Model: huggingface-llm-falcon-7b-bf16 is not supported at this time",
        ):
            JumpStartExtractor(
                jumpstart_model_id="huggingface-llm-falcon-7b-bf16",
                jumpstart_model_version="*",
                jumpstart_model_type=JumpStartModelType.OPEN_WEIGHTS,
                sagemaker_session=sagemaker_session,
            )
