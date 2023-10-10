"""
Module to manage model runners for SageMaker Jumpstart endpoints.
"""
import logging
import sagemaker
from typing import Optional, Tuple

import util
from constants import MIME_TYPE_JSON
from model_runners.model_runner import ModelRunner
from model_runners.util import get_sagemaker_session, is_endpoint_in_service

logger = logging.getLogger(__name__)


class JumpStartModelRunner(ModelRunner):
    """
    A class to manage the creation and deletion of SageMaker Jumpstart model runner when user provides
    a SageMaker Jumpstart endpoint name from a SageMaker Jumpstart model.
    """

    def __init__(
        self,
        endpoint_name: str,
        model_id: str,
        content_template: str,
        model_version: Optional[str] = "*",
        custom_attributes: Optional[str] = None,
        output: Optional[str] = None,
        log_probability: Optional[str] = None,
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
        """
        super().__init__(content_template, output, log_probability, MIME_TYPE_JSON, MIME_TYPE_JSON)
        self._sagemaker_session: sagemaker.session.Session = get_sagemaker_session()
        self._model_id: str = model_id
        self._model_version: Optional[str] = model_version
        self._custom_attributes: Optional[str] = custom_attributes

        util.require(
            is_endpoint_in_service(self._sagemaker_session, endpoint_name),
            "Endpoint is not in service",
        )

        self._predictor = sagemaker.predictor.retrieve_default(
            endpoint_name=endpoint_name,
            model_id=self._model_id,
            model_version=self._model_version,
            sagemaker_session=self._sagemaker_session,
        )
        util.require(
            self._predictor.accept == MIME_TYPE_JSON, f"Model accept type `{self._predictor.accept}` is not supported."
        )
        util.require(
            self._predictor.content_type == MIME_TYPE_JSON,
            f"Model content type `{self._predictor.content_type}` is not supported.",
        )

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Invoke the SageMaker endpoint and parse the model response.
        :param prompt: Input data for which you want the model to provide inference.
        """
        composed_data = self._composer.compose(prompt)
        model_output = self._predictor.predict(data=composed_data, custom_attributes=self._custom_attributes)
        output = self._extractor.extract_output(data=model_output, num_records=1)
        log_probability = self._extractor.extract_log_probability(data=model_output, num_records=1)
        return output, log_probability
