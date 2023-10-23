from typing import Union, List, Dict, Optional

from sagemaker import Session
from sagemaker.jumpstart.payload_utils import _extract_generated_text_from_response

from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.data_loaders.jmespath_util import compile_jmespath
from amazon_fmeval.model_runners.extractors.extractor import Extractor

JS_LOG_PROB_JMESPATH = "details.prefill[*].logprob"


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

    def extract_log_probability(self, data: Union[List, Dict], num_records: int = 1) -> Optional[float]:
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

    def extract_output(self, data: Union[List, Dict], num_records: int = 1) -> Optional[str]:
        """
        Extracts the output string from the JumpStartModel response. This value is provided by all JS text models, but
        not all JS FM models. This only supported for text-to-text models.

        :param data: The model response from the JumpStart Model
        :param num_records: The number of records in the model response. Must be 1.
        """
        assert num_records == 1, "Jumpstart extractor does not support batch requests"
        try:
            return _extract_generated_text_from_response(
                response=data,
                model_id=self._model_id,
                model_version=self._model_version,
                sagemaker_session=self._sagemaker_session,
                tolerate_vulnerable_model=True,
                tolerate_deprecated_model=True,
            )
        except ValueError as e:
            raise EvalAlgorithmClientError(f"Unable to extract output from Jumpstart model: {self._model_id}", e)
