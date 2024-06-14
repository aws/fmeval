from typing import NamedTuple, Optional
from unittest.mock import patch

import pytest
from sagemaker.jumpstart.types import JumpStartSerializablePayload
from sagemaker.jumpstart.enums import JumpStartModelType

from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.composers.jumpstart_composer import JumpStartComposer

OSS_MODEL_ID = "huggingface-eqa-roberta-large"
PROPRIETARY_MODEL_ID = "cohere-gpt-medium"
EMBEDDING_MODEL_ID = "tcembedding-model-id"
PROMPT = "Hello, how are you?"


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
                model_id=OSS_MODEL_ID,
                prompt="Hello, how are you?",
                expected_payload=JumpStartSerializablePayload(
                    {
                        "content_type": "application/json",
                        "body": '{"data": "Hello, how are you?"}',
                        "accept": "application/json",
                    }
                ),
            ),
            TestCaseCompose(
                model_id=PROPRIETARY_MODEL_ID,
                prompt="Hello, how are you?",
                expected_payload=JumpStartSerializablePayload(
                    {
                        "content_type": "application/json",
                        "body": '{"data": "Hello, how are you?"}',
                        "accept": "application/json",
                    }
                ),
            ),
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
        if test_case.model_id == PROPRIETARY_MODEL_ID:
            construct_payload.assert_called_with(
                test_case.prompt,
                model_id=test_case.model_id,
                model_type=JumpStartModelType.PROPRIETARY,
                model_version=test_case.model_version,
                tolerate_deprecated_model=True,
                tolerate_vulnerable_model=True,
            )
        else:
            construct_payload.assert_called_with(
                test_case.prompt,
                model_id=test_case.model_id,
                model_type=JumpStartModelType.OPEN_WEIGHTS,
                model_version=test_case.model_version,
                tolerate_deprecated_model=True,
                tolerate_vulnerable_model=True,
            )

    @patch("fmeval.model_runners.composers.jumpstart_composer._construct_payload")
    def test_compose_embedding_model(self, construct_payload):
        js_composer = JumpStartComposer(
            jumpstart_model_id=EMBEDDING_MODEL_ID, jumpstart_model_version="*", is_embedding_model=True
        )
        assert js_composer.compose(PROMPT) == PROMPT
        construct_payload.assert_not_called()

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
