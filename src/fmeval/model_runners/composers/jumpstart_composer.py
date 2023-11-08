from sagemaker.jumpstart.payload_utils import _construct_payload
from sagemaker.jumpstart.types import JumpStartSerializablePayload

from fmeval import util
from fmeval.model_runners.composers import Composer


class JumpStartComposer(Composer):
    """
    Jumpstart model request composer
    """

    def __init__(self, jumpstart_model_id: str, jumpstart_model_version: str):
        """
        Initialize the JumpStartComposer for the given JumpStart model_id and model_version.
        """
        self._model_id = jumpstart_model_id
        self._model_version = jumpstart_model_version

    def compose(self, prompt: str) -> JumpStartSerializablePayload:
        """
        Composes the payload for the given JumpStartModel from the provided prompt.
        """
        payload = _construct_payload(
            prompt,
            model_id=self._model_id,
            model_version=self._model_version,
            tolerate_deprecated_model=True,
            tolerate_vulnerable_model=True,
        )
        util.require(payload, f"Unable to fetch default model payload for JumpStart model: {self._model_id}")
        return payload
