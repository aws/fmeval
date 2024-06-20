import json
from typing import Optional, Union

from sagemaker.jumpstart.payload_utils import _construct_payload
from sagemaker.jumpstart.types import JumpStartSerializablePayload
from sagemaker.jumpstart.enums import JumpStartModelType

from fmeval import util
from fmeval.model_runners.composers import Composer
from fmeval.model_runners.util import get_sagemaker_session, is_proprietary_js_model


class JumpStartComposer(Composer):
    """
    Jumpstart model request composer
    """

    def __init__(
        self, jumpstart_model_id: str, jumpstart_model_version: str, is_embedding_model: Optional[bool] = False
    ):
        """
        Initialize the JumpStartComposer for the given JumpStart model_id and model_version.
        """
        self._model_id = jumpstart_model_id
        self._model_version = jumpstart_model_version
        self._is_embedding_model = is_embedding_model

    def compose(self, prompt: str) -> Union[JumpStartSerializablePayload, bytes]:
        """
        Composes the payload for the given JumpStartModel from the provided prompt.
        """
        # embedding models input to the endpoint is any string of text dumped in json and encoded in `utf-8` format
        if self._is_embedding_model:
            return json.dumps(prompt).encode("utf-8")
        sagemaker_session = get_sagemaker_session()
        # Default model type is always OPEN_WEIGHTS. See https://tinyurl.com/yc58s6wj
        jumpstart_model_type = JumpStartModelType.OPEN_WEIGHTS
        if is_proprietary_js_model(sagemaker_session.boto_region_name, self._model_id):
            jumpstart_model_type = JumpStartModelType.PROPRIETARY

        payload = _construct_payload(
            prompt,
            model_id=self._model_id,
            model_type=jumpstart_model_type,
            model_version=self._model_version,
            tolerate_deprecated_model=True,
            tolerate_vulnerable_model=True,
        )
        util.require(payload, f"Unable to fetch default model payload for JumpStart model: {self._model_id}")
        return payload
