"""
Module to manage model runners for SageMaker Endpoints with JumpStart LLMs.
"""
import logging

import sagemaker
from typing import Optional, Tuple

import fmeval.util as util
from fmeval.constants import MIME_TYPE_JSON
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.model_runners.util import get_sagemaker_session, is_endpoint_in_service

logger = logging.getLogger(__name__)


class JumpStartModelRunner(ModelRunner):
    """
    A class to manage the creation and deletion of a SageMaker Jumpstart model runner
    when the user provides an endpoint name corresponding to a SageMaker Endpoint
    for a JumpStart LLM.
    """

    def __init__(
        self,
        endpoint_name: str,
        model_id: str,
        content_template: Optional[str] = None,
        model_version: Optional[str] = "*",
        custom_attributes: Optional[str] = None,
        output: Optional[str] = None,
        log_probability: Optional[str] = None,
        component_name: Optional[str] = None,
    ):
        """
        :param endpoint_name: Name of the SageMaker endpoint to be used for model predictions
        :param model_id: Identifier of the SageMaker Jumpstart model
        :param content_template: String template to compose the model input from the prompt
        :param model_version: Version of the SageMaker Jumpstart model
        :param custom_attributes: String that contains the custom attributes to be passed to
                                  SageMaker endpoint invocation
        :param output: JMESPath expression of output in the model output
        :param log_probability: JMESPath expression of log probability in the model output
        :param component_name: Name of the Amazon SageMaker inference component corresponding
                            the predictor
        """
        super().__init__(
            content_template=content_template,
            output=output,
            log_probability=log_probability,
            content_type=MIME_TYPE_JSON,
            accept_type=MIME_TYPE_JSON,
            jumpstart_model_id=model_id,
            jumpstart_model_version=model_version,
        )
        self._endpoint_name = endpoint_name
        self._model_id = model_id
        self._content_template = content_template
        self._model_version = model_version
        self._custom_attributes = custom_attributes
        self._output = output
        self._log_probability = log_probability
        self._component_name = component_name
        sagemaker_session = get_sagemaker_session()
        util.require(
            is_endpoint_in_service(sagemaker_session, self._endpoint_name),
            f"Endpoint {self._endpoint_name} is not in service",
        )
        predictor = sagemaker.predictor.retrieve_default(
            endpoint_name=self._endpoint_name,
            model_id=self._model_id,
            model_version=self._model_version,
            sagemaker_session=sagemaker_session,
        )
        util.require(predictor.accept == MIME_TYPE_JSON, f"Model accept type `{predictor.accept}` is not supported.")
        self._predictor = predictor

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Invoke the SageMaker endpoint and parse the model response.
        :param prompt: Input data for which you want the model to provide inference.
        """
        composed_data = self._composer.compose(prompt)
        model_output = self._predictor.predict(
            data=composed_data,
            custom_attributes=self._custom_attributes,
            component_name=self._component_name,
        )
        # expect output from all model responses in JS
        output = self._extractor.extract_output(data=model_output, num_records=1)
        log_probability = None
        try:
            log_probability = self._extractor.extract_log_probability(data=model_output, num_records=1)
        except EvalAlgorithmClientError as e:
            # log_probability may be missing
            logger.warning(f"Unable to fetch log_probability from model response: {e}")
        return output, log_probability

    def __reduce__(self):
        """
        Custom serializer method used by Ray when it serializes instances of this
        class in eval_algorithms.util.generate_model_predict_response_for_dataset.
        """
        serialized_data = (
            self._endpoint_name,
            self._model_id,
            self._content_template,
            self._model_version,
            self._custom_attributes,
            self._output,
            self._log_probability,
            self._component_name,
        )
        return self.__class__, serialized_data
