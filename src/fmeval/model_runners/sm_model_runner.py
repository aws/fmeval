"""
Module to manage model runners for SageMaker endpoints.
"""
import logging
import sagemaker
import fmeval.util as util
from typing import Optional, Tuple, Union, List
from fmeval.constants import MIME_TYPE_JSON
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.model_runners.util import get_sagemaker_session, is_endpoint_in_service

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
        embedding: Optional[str] = None,
        content_type: str = MIME_TYPE_JSON,
        accept_type: str = MIME_TYPE_JSON,
        component_name: Optional[str] = None,
    ):
        """
        :param endpoint_name: Name of the SageMaker endpoint to be used for model predictions
        :param content_template: String template to compose the model input from the prompt
        :param custom_attributes: String that contains the custom attributes to be passed to
                                  SageMaker endpoint invocation
        :param output: JMESPath expression of output in the model output
        :param log_probability: JMESPath expression of log probability in the model output
        :param embedding: JMESPath expression of embedding in the model output
        :param content_type: The content type of the request sent to the model for inference
        :param accept_type: The accept type of the request sent to the model for inference
        :param component_name: Name of the Amazon SageMaker inference component corresponding
                            the predictor
        """
        super().__init__(content_template, output, log_probability, embedding, content_type, accept_type)
        self._endpoint_name = endpoint_name
        self._content_template = content_template
        self._custom_attributes = custom_attributes
        self._output = output
        self._log_probability = log_probability
        self._embedding = embedding
        self._content_type = content_type
        self._accept_type = accept_type
        self._component_name = component_name

        sagemaker_session = get_sagemaker_session()
        util.require(
            is_endpoint_in_service(sagemaker_session, self._endpoint_name),
            "Endpoint {endpoint_name} is not in service",
        )
        self._predictor = sagemaker.predictor.Predictor(
            endpoint_name=self._endpoint_name,
            sagemaker_session=sagemaker_session,
            # we only support JSON format model input/output currently
            serializer=sagemaker.serializers.JSONSerializer(),
            deserializer=sagemaker.deserializers.JSONDeserializer(),
        )

    def predict(self, prompt: str) -> Union[Tuple[Optional[str], Optional[float]], List[float]]:
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

        embedding = (
            self._extractor.extract_embedding(data=model_output, num_records=1)
            if self._extractor.embedding_jmespath_expression
            else None
        )
        if embedding:
            return embedding

        output = (
            self._extractor.extract_output(data=model_output, num_records=1)
            if self._extractor.output_jmespath_expression
            else None
        )
        log_probability = (
            self._extractor.extract_log_probability(data=model_output, num_records=1)
            if self._extractor.log_probability_jmespath_expression
            else None
        )
        return output, log_probability

    def __reduce__(self):
        """
        Custom serializer method used by Ray when it serializes instances of this
        class in eval_algorithms.util.generate_model_predict_response_for_dataset.
        """
        serialized_data = (
            self._endpoint_name,
            self._content_template,
            self._custom_attributes,
            self._output,
            self._log_probability,
            self._embedding,
            self._content_type,
            self._accept_type,
            self._component_name,
        )
        return self.__class__, serialized_data
