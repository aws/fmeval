import json
import logging
import os
from typing import Union, List, Dict, Optional
from urllib import request

import jmespath
from functional import seq
from jmespath.exceptions import JMESPathError
from sagemaker import Session
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.utils import verify_model_region_and_return_specs

from fmeval import util
from fmeval.constants import (
    MODEL_ID,
    GENERATED_TEXT_JMESPATH_EXPRESSION,
    SDK_MANIFEST_FILE,
    PROPRIETARY_SDK_MANIFEST_FILE,
    JUMPSTART_BUCKET_BASE_URL_FORMAT,
    JUMPSTART_BUCKET_BASE_URL_FORMAT_ENV_VAR,
    INPUT_LOG_PROBS_JMESPATH_EXPRESSION,
    EMBEDDING_JMESPATH_EXPRESSION,
    SPEC_KEY,
    DEFAULT_PAYLOADS,
)
from fmeval.exceptions import EvalAlgorithmClientError, EvalAlgorithmInternalError
from fmeval.model_runners.extractors.extractor import Extractor

# The expected model response location for Jumpstart that do produce the log probabilities
from fmeval.model_runners.util import get_sagemaker_session

logger = logging.getLogger(__name__)


class JumpStartExtractor(Extractor):
    """
    JumpStart model response extractor
    """

    def __init__(
        self,
        jumpstart_model_id: str,
        jumpstart_model_version: str,
        jumpstart_model_type: str,
        is_embedding_model: Optional[bool] = False,
        sagemaker_session: Optional[Session] = None,
    ):
        """
        Initializes  JumpStartExtractor for the given model and version.
        This extractor does not support batching at this time.

        :param jumpstart_model_id: The model id of the JumpStart Model
        :param jumpstart_model_id: The model version of the JumpStart Model
        :param is_embedding_model: Whether this model is an embedding model or not
        :param sagemaker_session: Optional. An object of SageMaker session
        """
        self._model_id = jumpstart_model_id
        self._model_version = jumpstart_model_version
        self._model_type = jumpstart_model_type
        self._sagemaker_session = sagemaker_session if sagemaker_session else get_sagemaker_session()
        self._is_embedding_model = is_embedding_model

        if self._is_embedding_model:
            self._embedding_compiler = jmespath.compile(EMBEDDING_JMESPATH_EXPRESSION)
            return

        model_manifest = seq(self.get_jumpstart_sdk_manifest(self._sagemaker_session.boto_region_name)).find(
            lambda x: x.get(MODEL_ID, None) == jumpstart_model_id
        )
        util.require(model_manifest, f"Model {jumpstart_model_id} is not a valid JumpStart Model")

        model_spec = self.get_jumpstart_sdk_spec(
            model_manifest.get(SPEC_KEY, None),
            self._sagemaker_session.boto_region_name,
        )
        if DEFAULT_PAYLOADS not in model_spec:
            # Model spec contains alt configs, which should
            # be obtained through JumpStart util function.
            logger.info(
                "default_payloads not found as a top-level attribute of model spec"
                "Searching for default_payloads in inference configs instead."
            )
            model_spec = verify_model_region_and_return_specs(
                region=self._sagemaker_session.boto_region_name,
                model_id=self._model_id,
                version=self._model_version,
                model_type=self._model_type,
                scope=JumpStartScriptScope.INFERENCE,
                sagemaker_session=self._sagemaker_session,
            )
            configs = model_spec.inference_configs  # type: ignore[attr-defined]
            util.require(configs, f"JumpStart Model: {jumpstart_model_id} is not supported at this time")
            default_payloads = configs.get_top_config_from_ranking().resolved_metadata_config[DEFAULT_PAYLOADS]
        else:
            # Continue to extract default payloads by manually parsing the spec json object.
            # TODO: update this code when the `default_payloads` attribute of JumpStartModelSpecs
            #  returns the full data, including fields like generated_text.
            default_payloads = model_spec[DEFAULT_PAYLOADS]

        util.require(default_payloads, f"JumpStart Model: {jumpstart_model_id} is not supported at this time")

        try:
            output_jmespath_expressions = jmespath.compile(GENERATED_TEXT_JMESPATH_EXPRESSION).search(default_payloads)
            input_log_probs_jmespath_expressions = jmespath.compile(INPUT_LOG_PROBS_JMESPATH_EXPRESSION).search(
                default_payloads
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
        self._output_jmespath_compiler = None
        self._log_prob_compiler = None
        try:
            self._output_jmespath_compiler = jmespath.compile(output_jmespath_expressions[0])
            if input_log_probs_jmespath_expressions:  # pragma: no branch
                self._log_prob_compiler = jmespath.compile(input_log_probs_jmespath_expressions[0])
        except (TypeError, JMESPathError) as e:
            if not self._output_jmespath_compiler:
                raise EvalAlgorithmInternalError(
                    f"Output jmespath expression found for Jumpstart model {jumpstart_model_id} is not valid jmespath. "
                    f"Please provide output jmespath to the JumpStartModelRunner"
                ) from e
            else:
                raise EvalAlgorithmInternalError(
                    f"Input log probability jmespath expression found for Jumpstart model {jumpstart_model_id} is not valid jmespath. "
                    f"Please provide correct input log probability jmespath to the JumpStartModelRunner"
                ) from e

    def extract_log_probability(self, data: Union[List, Dict], num_records: int = 1) -> float:
        """
        Extracts the log probability from the JumpStartModel response. This value is not provided by all JS text models.

        :param data: The model response from the JumpStart Model
        :param num_records: The number of records in the model response. Must be 1.
        """
        assert num_records == 1, "Jumpstart extractor does not support batch requests"
        util.require(
            self._log_prob_compiler, f"Model {self._model_id} does not include input log probabilities in it's response"
        )
        try:
            assert (
                self._log_prob_compiler
            ), f"Model {self._model_id} does not include input log probabilities in it's response"
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
            assert (
                self._output_jmespath_compiler
            ), f"Unable to extract generated text from Jumpstart model: {self._model_id}"
            output = self._output_jmespath_compiler.search(data)
            if output is None and not isinstance(output, str):
                raise EvalAlgorithmClientError(f"Unable to extract output from Jumpstart model: {self._model_id}")
            return output
        except ValueError as e:
            raise EvalAlgorithmClientError(f"Unable to extract output from Jumpstart model: {self._model_id}", e)

    def extract_embedding(self, data: Union[List, Dict], num_records: int = 1) -> List[float]:
        """
        Extracts the embedding from the JumpStartModel response. This only supported for text embedding models.

        :param data: The model response from the JumpStart Model
        :param num_records: The number of records in the model response. Must be 1.
        """
        assert num_records == 1, "Jumpstart extractor does not support batch requests"
        try:
            assert self._embedding_compiler, f"Unable to extract embedding from Jumpstart model: {self._model_id}"
            output = self._embedding_compiler.search(data)
            if output is None and not isinstance(output, str):
                raise EvalAlgorithmClientError(f"Unable to extract embedding from Jumpstart model: {self._model_id}")
            return output
        except ValueError as e:
            raise EvalAlgorithmClientError(f"Unable to extract embedding from Jumpstart model: {self._model_id}", e)

    @staticmethod
    def get_jumpstart_sdk_manifest(region: str) -> List[Dict]:
        jumpstart_bucket_base_url = os.environ.get(
            JUMPSTART_BUCKET_BASE_URL_FORMAT_ENV_VAR, JUMPSTART_BUCKET_BASE_URL_FORMAT
        ).format(region, region)
        open_source_url = "{}/{}".format(jumpstart_bucket_base_url, SDK_MANIFEST_FILE)
        proprietary_url = "{}/{}".format(jumpstart_bucket_base_url, PROPRIETARY_SDK_MANIFEST_FILE)

        with request.urlopen(open_source_url) as f:
            oss_models_manifest = f.read().decode("utf-8")
        with request.urlopen(proprietary_url) as f:
            proprietary_models_manifest = f.read().decode("utf-8")

        return json.loads(oss_models_manifest) + json.loads(proprietary_models_manifest)

    @staticmethod
    def get_jumpstart_sdk_spec(key: str, region: str) -> Dict:
        jumpstart_bucket_base_url = os.environ.get(
            JUMPSTART_BUCKET_BASE_URL_FORMAT_ENV_VAR, JUMPSTART_BUCKET_BASE_URL_FORMAT
        ).format(region, region)
        url = "{}/{}".format(jumpstart_bucket_base_url, key)
        with request.urlopen(url) as f:
            model_spec = f.read().decode("utf-8")
        return json.loads(model_spec)
