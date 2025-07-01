import logging

from typing import Dict, List, Optional, Union

import fmeval.util as util
from fmeval.data_loaders.jmespath_util import compile_jmespath
from fmeval.model_runners.extractors.extractor import Extractor

logger = logging.getLogger(__name__)


class JsonExtractor(Extractor):
    """
    JSON model response extractor
    """

    def __init__(
        self,
        output_jmespath_expression: Optional[str] = None,
        log_probability_jmespath_expression: Optional[str] = None,
        embedding_jmespath_expression: Optional[str] = None,
    ):
        """
        Creates an instance of Json extractor that can extract the output and log probability from the JSON model
        response.

        :param output_jmespath_expression: JMESPath expression of the output string
        """
        self.log_probability_jmespath_expression = log_probability_jmespath_expression
        self.log_probability_jmespath = (
            compile_jmespath(log_probability_jmespath_expression) if log_probability_jmespath_expression else None
        )
        self.output_jmespath_expression = output_jmespath_expression
        self.output_jmespath = compile_jmespath(output_jmespath_expression) if output_jmespath_expression else None
        self.embedding_jmespath_expression = embedding_jmespath_expression
        self.embedding_jmespath = (
            compile_jmespath(embedding_jmespath_expression) if embedding_jmespath_expression else None
        )

    def extract_log_probability(self, data: Union[List, Dict], num_records: int) -> Union[List[float], float]:
        """
        Extract log probability from model response.

        :param data: Model response. The log_probability_jmespath_expression is used to extract the log probabilities
                     of the input tokens. Each record in the extracted probabilities will be a float or list of floats.
                     Examples for the extracted probabilities:
                        - data: 0.1, num_records: 1, num tokens: 1 (or probabilities already summed up)
                        - data: [0.1], num_records: 1, num tokens: 1 (or probabilities already summed up)
                        - data: [0.1, 0.2], num_records: 1, num tokens: 2
        :param num_records: number of inference records in the model output
        :return: float or list of float where each float is sum of log probabilities.
        """
        assert num_records == 1, "JSON extractor does not support batch requests"
        util.require(
            self.log_probability_jmespath_expression,
            "Extractor cannot extract log_probability as log_probability_jmespath_expression is not provided",
        )
        log_probs = self.log_probability_jmespath.search(data)
        util.require(
            log_probs is not None, f"JMESpath {self.log_probability_jmespath_expression} could not find any data"
        )
        if isinstance(log_probs, float):
            return log_probs
        util.require(
            isinstance(log_probs, List) and all(isinstance(value, float) for value in log_probs),
            f"Extractor found: {log_probs} which does not match expected {float} or list of {float}",
        )
        return sum(log_probs)

    def extract_output(self, data: Union[List, Dict], num_records: int) -> Union[List[str], str]:
        """
        Extract output from JSON model output

        :param data: Model response. The output_jmespath_expression is used to extract the predicted output. The
                     predicted output must be a string
        :param num_records: number of inference records in the model output
        :return: model output
        """
        assert num_records == 1, "JSON extractor does not support batch requests"
        util.require(
            self.output_jmespath_expression,
            "Extractor cannot extract output as output_jmespath_expression is not provided",
        )
        outputs = self.output_jmespath.search(data)
        util.require(outputs is not None, f"JMESpath {self.output_jmespath_expression} could not find any data")
        util.require(isinstance(outputs, str), f"Extractor found: {outputs} which does not match expected type {str}")
        return outputs

    def extract_embedding(self, data: Union[List, Dict], num_records: int) -> Union[List[float]]:
        """
        Extract embedding from JSON model output

        :param data: Model response. The embedding_jmespath_expression is used to extract the predicted output. The
                     predicted output must be a string
        :param num_records: number of inference records in the model output
        :return: model output
        """
        assert num_records == 1, "JSON extractor does not support batch requests"
        util.require(
            self.embedding_jmespath_expression,
            "Extractor cannot extract embedding as embedding_jmespath_expression is not provided",
        )
        embedding = self.embedding_jmespath.search(data)
        util.require(embedding is not None, f"JMESpath {self.embedding_jmespath_expression} could not find any data")
        util.require(
            isinstance(embedding, List), f"Extractor found: {embedding} which does not match expected type {List}"
        )
        return embedding
