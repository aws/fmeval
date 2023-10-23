from unittest import mock
from unittest.mock import patch

import pytest
from _pytest.python_api import approx

from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.extractors import jumpstart_extractor
from amazon_fmeval.model_runners.extractors.jumpstart_extractor import JumpStartExtractor

EXAMPLE_JUMPSTART_RESPONSE = {
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


class TestJumpStartExtractor:
    def test_log_probabilty_fails_for_batch_request(self):
        extractor = JumpStartExtractor(
            jumpstart_model_id="jumpstart_model_id", jumpstart_model_version="jumpstart_model_version"
        )
        with pytest.raises(AssertionError, match="Jumpstart extractor does not support batch requests"):
            extractor.extract_log_probability({"log_pob": 0.8}, 2)

    def test_output_fails_for_batch_request(self):
        extractor = JumpStartExtractor(
            jumpstart_model_id="jumpstart_model_id", jumpstart_model_version="jumpstart_model_version"
        )
        with pytest.raises(AssertionError, match="Jumpstart extractor does not support batch requests"):
            extractor.extract_output({"log_pob": 0.8}, 2)

    @patch(
        "amazon_fmeval.model_runners.extractors.jumpstart_extractor._extract_generated_text_from_response",
        return_value="\n\nI am trying",
    )
    def test_extract_output(self, _extract_generated_text_from_response):
        extractor = JumpStartExtractor(
            jumpstart_model_id="jumpstart_model_id", jumpstart_model_version="jumpstart_model_version"
        )
        assert extractor.extract_output(EXAMPLE_JUMPSTART_RESPONSE) == "\n\nI am trying"
        _extract_generated_text_from_response.assert_called_with(
            response=EXAMPLE_JUMPSTART_RESPONSE,
            model_id="jumpstart_model_id",
            model_version="jumpstart_model_version",
            sagemaker_session=None,
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
        )

    def test_extract_output_failure(self):
        extractor = JumpStartExtractor(
            jumpstart_model_id="jumpstart_model_id", jumpstart_model_version="jumpstart_model_version"
        )
        response = {"response": "Hey! I am good. How are you?"}

        exception_mock = mock.Mock()
        exception_mock.side_effect = ValueError
        with patch.object(
            jumpstart_extractor, "_extract_generated_text_from_response", exception_mock
        ) as _extract_generated_text_from_response:
            with pytest.raises(EvalAlgorithmClientError, match="Unable to extract output from Jumpstart model:"):
                extractor.extract_output(response)
                _extract_generated_text_from_response.assert_called_with(
                    response=response,
                    model_id="jumpstart_model_id",
                    model_version="jumpstart_model_version",
                    sagemaker_session=None,
                    tolerate_vulnerable_model=True,
                    tolerate_deprecated_model=True,
                )

    def test_log_probability(self):
        extractor = JumpStartExtractor(
            jumpstart_model_id="jumpstart_model_id", jumpstart_model_version="jumpstart_model_version"
        )
        assert extractor.extract_log_probability(EXAMPLE_JUMPSTART_RESPONSE) == approx(-13.3945313)

    def test_log_probability_missing_log_prob(self):
        extractor = JumpStartExtractor(
            jumpstart_model_id="jumpstart_model_id", jumpstart_model_version="jumpstart_model_version"
        )
        with pytest.raises(EvalAlgorithmClientError, match="Unable to extract output from Jumpstart model:"):
            extractor.extract_log_probability({})
