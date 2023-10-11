"""
Module to manage model runners for Bedrock models.
"""
import json
import logging
from amazon_fmeval.util import require
from typing import Optional, Tuple
from amazon_fmeval.constants import MIME_TYPE_JSON
from amazon_fmeval.model_runners.model_runner import ModelRunner
from amazon_fmeval.model_runners.util import get_bedrock_runtime_client

logger = logging.getLogger(__name__)


class BedrockModelRunner(ModelRunner):
    """
    A class to manage the creation and deletion of Bedrock model runner when user provides
    a Bedrock model id.
    """

    def __init__(
        self,
        model_id: str,
        content_template: str,
        output: Optional[str] = None,
        log_probability: Optional[str] = None,
        content_type: str = MIME_TYPE_JSON,
        accept_type: str = MIME_TYPE_JSON,
    ):
        """
        :param model_id: Id of the Bedrock model to be used for model predictions
        :param content_template: String template to compose the model input from the prompt
        :param output: JMESPath expression of output in the model output
        :param log_probability: JMESPath expression of log probability in the model output
        :param content_type: The content type of the request sent to the model for inference
        :param accept_type: The accept type of the request sent to the model for inference
        """
        super().__init__(content_template, output, log_probability, content_type, accept_type)
        self._bedrock_runtime_client = get_bedrock_runtime_client()
        self._model_id = model_id
        self._content_type = content_type
        self._accept_type = accept_type

        require(self._accept_type == MIME_TYPE_JSON, f"Model accept type `{self._accept_type}` is not supported.")
        require(
            self._content_type == MIME_TYPE_JSON,
            f"Model content type `{self._content_type}` is not supported.",
        )

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Invoke the Bedrock model and parse the model response.
        :param prompt: Input data for which you want the model to provide inference.
        """
        composed_data = self._composer.compose(prompt)
        response = self._bedrock_runtime_client.invoke_model(
            body=composed_data, modelId=self._model_id, accept=self._accept_type, contentType=self._content_type
        )
        model_output = json.loads(response.get("body").read())
        output = self._extractor.extract_output(data=model_output, num_records=1)
        log_probability = self._extractor.extract_log_probability(data=model_output, num_records=1)
        return output, log_probability
