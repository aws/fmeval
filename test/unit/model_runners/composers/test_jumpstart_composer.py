from typing import NamedTuple, Optional
from unittest.mock import patch

import pytest
from sagemaker.jumpstart.types import JumpStartSerializablePayload

from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.composers.jumpstart_composer import JumpStartComposer


class TestJumpStartComposer:
    class TestCaseCompose(NamedTuple):
        model_id: str
        prompt: str
        expected_payload: Optional[JumpStartSerializablePayload]
        model_version: str = "*"

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseCompose(
                model_id="huggingface-eqa-roberta-large",
                prompt="Hello, how are you?",
                expected_payload=JumpStartSerializablePayload(
                    {
                        "content_type": "application/json",
                        "body": '{"data": "Hello, how are you?"}',
                        "accept": "application/json",
                    }
                ),
            )
        ],
    )
    @patch(
        "fmeval.model_runners.composers.jumpstart_composer._construct_payload",
        return_value=JumpStartSerializablePayload(
            {
                "content_type": "application/json",
                "body": '{"data": "Hello, how are you?"}',
                "accept": "application/json",
            }
        ),
    )
    def test_compose(self, construct_payload, test_case: TestCaseCompose):
        js_composer = JumpStartComposer(
            jumpstart_model_id=test_case.model_id, jumpstart_model_version=test_case.model_version
        )
        assert js_composer.compose(test_case.prompt) == test_case.expected_payload

    @patch(
        "fmeval.model_runners.composers.jumpstart_composer._construct_payload",
        return_value=None,
    )
    def test_compose_failure(self, construct_payload):
        js_composer = JumpStartComposer(jumpstart_model_id="model_id", jumpstart_model_version="model_version")
        with pytest.raises(
            EvalAlgorithmClientError, match="Unable to fetch default model payload for JumpStart model: model_id"
        ):
            js_composer.compose("prompt")
            construct_payload.assert_called_with(
                "prompt",
                model_id="model_id",
                model_version="model_version",
                tolerate_deprecated_model=True,
                tolerate_vulnerable_model=True,
            )
