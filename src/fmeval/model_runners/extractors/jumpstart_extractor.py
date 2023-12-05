import json
import os
from typing import Union, List, Dict, Optional
from urllib import request

import jmespath
from functional import seq
from jmespath.exceptions import JMESPathError
from sagemaker import Session

from fmeval import util
from fmeval.constants import (
    MODEL_ID,
    SPEC_KEY,
    GENERATED_TEXT_JMESPATH_EXPRESSION,
    SDK_MANIFEST_FILE,
    DEFAULT_PAYLOADS,
    JUMPSTART_BUCKET_BASE_URL_FORMAT,
    JUMPSTART_BUCKET_BASE_URL_FORMAT_ENV_VAR,
)
from fmeval.exceptions import EvalAlgorithmClientError, EvalAlgorithmInternalError
from fmeval.data_loaders.jmespath_util import compile_jmespath
from fmeval.model_runners.extractors.extractor import Extractor

# The expected model response location for Jumpstart that do produce the log probabilities
from fmeval.model_runners.util import get_sagemaker_session

JS_LOG_PROB_JMESPATH = "[0].details.prefill[*].logprob"


class JumpStartExtractor(Extractor):
    """
    JumpStart model response extractor
    """

    def __init__(
        self, jumpstart_model_id: str, jumpstart_model_version: str, sagemaker_session: Optional[Session] = None
    ):
        """
        Initializes  JumpStartExtractor for the given model and version.
        This extractor does not support batching at this time.

        :param jumpstart_model_id: The model id of the JumpStart Model
        :param jumpstart_model_id: The model version of the JumpStart Model
        :param sagemaker_session: Optional. An object of SageMaker session
        """
        self._model_id = jumpstart_model_id
        self._model_version = jumpstart_model_version
        self._sagemaker_session = sagemaker_session if sagemaker_session else get_sagemaker_session()
        self._log_prob_compiler = compile_jmespath(JS_LOG_PROB_JMESPATH)

        model_manifest = seq(self.get_jumpstart_sdk_manifest(self._sagemaker_session.boto_region_name)).find(
            lambda x: x.get(MODEL_ID, None) == jumpstart_model_id
        )
        util.require(model_manifest, f"Model {jumpstart_model_id} is not a valid JumpStart Model")
        model_spec_key = self.get_jumpstart_sdk_spec(
            model_manifest.get(SPEC_KEY, None),
            self._sagemaker_session.boto_region_name,
        )
        util.require(
            DEFAULT_PAYLOADS in model_spec_key, f"JumpStart Model: {jumpstart_model_id} is not supported at this time"
        )

        try:
            output_jmespath_expressions = jmespath.compile(GENERATED_TEXT_JMESPATH_EXPRESSION).search(
                model_spec_key[DEFAULT_PAYLOADS]
            )
        except (TypeError, JMESPathError) as e:
            raise EvalAlgorithmInternalError(
                f"Unable find the generated_text key in the default payload for JumpStart model: {jumpstart_model_id}. "
                f"Please provide output jmespath to the JumpStartModelRunner"
            ) from e
        util.assert_condition(
            output_jmespath_expressions,
            f"Output jmespath expression for Jumpstart model {jumpstart_model_id} is empty. "
            f"Please provide output jmespath to the JumpStartModelRunner",
        )
        try:
            self._output_jmespath_compiler = jmespath.compile(output_jmespath_expressions[0])
        except (TypeError, JMESPathError) as e:
            raise EvalAlgorithmInternalError(
                f"Output jmespath expression found for Jumpstart model {jumpstart_model_id} is not valid jmespath. "
                f"Please provide output jmespath to the JumpStartModelRunner"
            ) from e

    def extract_log_probability(self, data: Union[List, Dict], num_records: int = 1) -> float:
        """
        Extracts the log probability from the JumpStartModel response. This value is not provided by all JS text models.

        :param data: The model response from the JumpStart Model
        :param num_records: The number of records in the model response. Must be 1.
        """
        assert num_records == 1, "Jumpstart extractor does not support batch requests"
        try:
            log_probs = self._log_prob_compiler.search(data)
            if log_probs is None and not isinstance(log_probs, list):
                raise EvalAlgorithmClientError(
                    f"Unable to extract log probability from Jumpstart model: {self._model_id}"
                )
            return sum(log_probs)
        except ValueError as e:
            raise EvalAlgorithmClientError(
                f"Unable to extract log probability from Jumpstart model: {self._model_id}", e
            )

    def extract_output(self, data: Union[List, Dict], num_records: int = 1) -> str:
        """
        Extracts the output string from the JumpStartModel response. This value is provided by all JS text models, but
        not all JS FM models. This only supported for text-to-text models.

        :param data: The model response from the JumpStart Model
        :param num_records: The number of records in the model response. Must be 1.
        """
        assert num_records == 1, "Jumpstart extractor does not support batch requests"
        try:
            output = self._output_jmespath_compiler.search(data)
            if output is None and not isinstance(output, str):
                raise EvalAlgorithmClientError(f"Unable to extract output from Jumpstart model: {self._model_id}")
            return output
        except ValueError as e:
            raise EvalAlgorithmClientError(f"Unable to extract output from Jumpstart model: {self._model_id}", e)

    @staticmethod
    def get_jumpstart_sdk_manifest(region: str) -> Dict:
        jumpstart_bucket_base_url = os.environ.get(
            JUMPSTART_BUCKET_BASE_URL_FORMAT_ENV_VAR, JUMPSTART_BUCKET_BASE_URL_FORMAT
        ).format(region, region)
        url = "{}/{}".format(jumpstart_bucket_base_url, SDK_MANIFEST_FILE)
        with request.urlopen(url) as f:
            models_manifest = f.read().decode("utf-8")
        return json.loads(models_manifest)

    @staticmethod
    def get_jumpstart_sdk_spec(key: str, region: str) -> Dict:
        jumpstart_bucket_base_url = os.environ.get(
            JUMPSTART_BUCKET_BASE_URL_FORMAT_ENV_VAR, JUMPSTART_BUCKET_BASE_URL_FORMAT
        ).format(region, region)
        url = "{}/{}".format(jumpstart_bucket_base_url, key)
        with request.urlopen(url) as f:
            model_spec = f.read().decode("utf-8")
        return json.loads(model_spec)
