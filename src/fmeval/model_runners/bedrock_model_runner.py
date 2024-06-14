"""
Module to manage model runners for Bedrock models.
"""
import json
import logging
from fmeval.util import require
from typing import Optional, Tuple, List, Union
from fmeval.constants import MIME_TYPE_JSON
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.model_runners.util import get_bedrock_runtime_client

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
        embedding: Optional[str] = None,
        content_type: str = MIME_TYPE_JSON,
        accept_type: str = MIME_TYPE_JSON,
    ):
        """
        :param model_id: Id of the Bedrock model to be used for model predictions
        :param content_template: String template to compose the model input from the prompt
        :param output: JMESPath expression of output in the model output
        :param log_probability: JMESPath expression of log probability in the model output
        :param embedding: JMESPath expression of embedding in the model output
        :param content_type: The content type of the request sent to the model for inference
        :param accept_type: The accept type of the request sent to the model for inference
        """
        super().__init__(content_template, output, log_probability, embedding, content_type, accept_type)
        self._model_id = model_id
        self._content_template = content_template
        self._output = output
        self._log_probability = log_probability
        self._content_type = content_type
        self._accept_type = accept_type
        self._embedding = embedding

        require(
            output is not None or log_probability is not None or embedding is not None,
            "One of output jmespath expression, log probability or embedding jmespath expression must be provided",
        )
        require(self._accept_type == MIME_TYPE_JSON, f"Model accept type `{self._accept_type}` is not supported.")
        require(
            self._content_type == MIME_TYPE_JSON,
            f"Model content type `{self._content_type}` is not supported.",
        )
        self._bedrock_runtime_client = get_bedrock_runtime_client()

    def predict(self, prompt: str) -> Union[Tuple[Optional[str], Optional[float]], List[float]]:
        """
        Invoke the Bedrock model and parse the model response.
        :param prompt: Input data for which you want the model to provide inference.
        """
        composed_data = self._composer.compose(prompt)
        body = json.dumps(composed_data)
        response = self._bedrock_runtime_client.invoke_model(
            body=body, modelId=self._model_id, accept=self._accept_type, contentType=self._content_type
        )
        model_output = json.loads(response.get("body").read())

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
            self._model_id,
            self._content_template,
            self._output,
            self._log_probability,
            self._embedding,
            self._content_type,
            self._accept_type,
        )
        return self.__class__, serialized_data
