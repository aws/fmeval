from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union

from fmeval.constants import MIME_TYPE_JSON
from fmeval.model_runners.composers import create_content_composer
from fmeval.model_runners.extractors import create_extractor


class ModelRunner(ABC):
    """
    This class is responsible for running the model and extracting the model output.

    It handles everything related to the model, including: model deployment, payload construction for invocations,
    and making sense of the model output.
    """

    def __init__(
        self,
        content_template: Optional[str] = None,
        output: Optional[str] = None,
        log_probability: Optional[str] = None,
        embedding: Optional[str] = None,
        content_type: str = MIME_TYPE_JSON,
        accept_type: str = MIME_TYPE_JSON,
        **kwargs
    ):
        """
        :param content_template: String template to compose the model input from the prompt
        :param output: JMESPath expression of output in the model output
        :param log_probability: JMESPath expression of log probability in the model output
        :param embedding: JMESPath expression of embedding in the model output
        :param content_type: The content type of the request sent to the model for inference
        :param accept_type: The accept type of the request sent to the model for inference
        """
        self._composer = create_content_composer(content_type=content_type, template=content_template, **kwargs)
        self._extractor = create_extractor(
            model_accept_type=accept_type,
            output_location=output,
            log_probability_location=log_probability,
            embedding_location=embedding,
            **kwargs,
        )

    @abstractmethod
    def predict(self, prompt: str) -> Union[Tuple[Optional[str], Optional[float]], List[float]]:
        """
        Runs the model on the given prompt. This includes updating the prompt to fit the request format that the model
        expects, and extracting the output and log probability from the model response. The response of the ModelRunner
        will be a tuple of (output, log_probability) or embedding

        :param prompt: the prompt
        :return: the tuple containing model output string and the log probability, or embedding
        """
