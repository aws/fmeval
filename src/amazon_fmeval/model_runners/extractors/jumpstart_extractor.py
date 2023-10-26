import json
from typing import Union, List, Dict, Optional
from urllib import request

from functional import seq
from sagemaker import Session

from amazon_fmeval import util
from amazon_fmeval.constants import (
    MODEL_ID,
    SPEC_KEY,
    GENERATED_TEXT_JMESPATH_EXPRESSION,
    SDK_MANIFEST_FILE,
    JUMPSTART_BUCKET_BASE_URL,
    DEFAULT_PAYLOADS,
)
from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.data_loaders.jmespath_util import compile_jmespath
from amazon_fmeval.model_runners.extractors.extractor import Extractor

# The expected model response location for Jumpstart that do produce the log probabilities
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
        self._sagemaker_session = sagemaker_session
        self._log_prob_compiler = compile_jmespath(JS_LOG_PROB_JMESPATH)

        model_spec_key = self.get_jumpstart_sdk_spec(
            seq(self.get_jumpstart_sdk_manifest()).find(lambda x: x[MODEL_ID] == jumpstart_model_id)[SPEC_KEY]
        )
        util.require(DEFAULT_PAYLOADS in model_spec_key, f"Model: {jumpstart_model_id} is not supported at this time")
        output_jmespath_expressions = compile_jmespath(GENERATED_TEXT_JMESPATH_EXPRESSION).search(
            model_spec_key[DEFAULT_PAYLOADS]
        )
        util.require(
            output_jmespath_expressions,
            f"Jumpstart model {jumpstart_model_id} is likely not supported. If you know it is generates text, "
            f"please provide output and log_probability jmespaths to the JumpStartModelRunner",
        )
        self._output_jmespath_compiler = compile_jmespath(output_jmespath_expressions[0])

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
                raise EvalAlgorithmClientError(f"Unable to extract output from Jumpstart model: {self._model_id}")
            return sum(log_probs)
        except ValueError as e:
            raise EvalAlgorithmClientError(f"Unable to extract output from Jumpstart model: {self._model_id}", e)

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
    def get_jumpstart_sdk_manifest() -> Dict:
        url = "{}/{}".format(JUMPSTART_BUCKET_BASE_URL, SDK_MANIFEST_FILE)
        with request.urlopen(url) as f:
            models_manifest = f.read().decode("utf-8")
        return json.loads(models_manifest)

    @staticmethod
    def get_jumpstart_sdk_spec(key) -> Dict:
        url = "{}/{}".format(JUMPSTART_BUCKET_BASE_URL, key)
        with request.urlopen(url) as f:
            model_spec = f.read().decode("utf-8")
        return json.loads(model_spec)
