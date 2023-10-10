"""
Module to manage model runners for SageMaker endpoints.
"""
import logging
import sagemaker
import amazon_fmeval.util as util
from typing import Optional, Tuple
from amazon_fmeval.constants import MIME_TYPE_JSON
from amazon_fmeval.model_runners.model_runner import ModelRunner
from amazon_fmeval.model_runners.util import get_sagemaker_session, is_endpoint_in_service

logger = logging.getLogger(__name__)


class SageMakerModelRunner(ModelRunner):
    """
    A class to manage the creation and deletion of SageMaker model runner when user provides
    a SageMaker endpoint name from a SageMaker model.
    """

    def __init__(
        self,
        endpoint_name: str,
        content_template: str,
        custom_attributes: Optional[str] = None,
        output: Optional[str] = None,
        log_probability: Optional[str] = None,
        content_type: str = MIME_TYPE_JSON,
        accept_type: str = MIME_TYPE_JSON,
    ):
        """
        :param endpoint_name: Name of the SageMaker endpoint to be used for model predictions
        :param content_template: String template to compose the model input from the prompt
        :param custom_attributes: String that contains the custom attributes to be passed to
                                  SageMaker endpoint invocation
        :param output: JMESPath expression of output in the model output
        :param log_probability: JMESPath expression of log probability in the model output
        :param content_type: The content type of the request sent to the model for inference
        :param accept_type: Name of the SageMaker endpoint to be used for model predictions
        """
        super().__init__(content_template, output, log_probability, content_type, accept_type)
        self._sagemaker_session: sagemaker.session.Session = get_sagemaker_session()
        self._custom_attributes: Optional[str] = custom_attributes

        util.require(
            is_endpoint_in_service(self._sagemaker_session, endpoint_name),
            "Endpoint is not in service",
        )
        self._predictor = sagemaker.predictor.Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=self._sagemaker_session,
            # we only support JSON format model input/output currently
            serializer=sagemaker.serializers.JSONSerializer(),
            deserializer=sagemaker.deserializers.JSONDeserializer(),
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
