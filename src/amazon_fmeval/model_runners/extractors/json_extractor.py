import logging

from typing import Any, Dict, List, Union, Type, Optional

import amazon_fmeval.util as util
from amazon_fmeval.data_loaders.jmespath_util import compile_jmespath
from amazon_fmeval.model_runners.extractors.extractor import Extractor

logger = logging.getLogger(__name__)


class JsonExtractor(Extractor):
    """
    JSON model response extractor
    """

    def __init__(
        self,
        output_jmespath_expression: Optional[str] = None,
        log_probability_jmespath_expression: Optional[str] = None,
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

    def extract_log_probability(self, data: Union[List, Dict], num_records: int) -> Union[List[float], float]:
        """
        Extract log probability from model response.

        :param data: Model response. The probability_jmespath_expression is used to extract the predicted softmax
                     probabilities. Each record in the extracted probabilities will be a list of list of floats.
                     Examples for the extracted probabilities:
                        - data: [[0.6],[0.3],[0.1]], num_records: 1
                        - data: [ [[0.6],[0.3],[0.1]] ], num_records: 1
                        - data: [ [[0.6]], [[0.1]] ], num_records: 2
        :param num_records: number of inference records in the model output
        :return: list of a list, each element is a list of probabilities.
        """
        util.require(
            self.log_probability_jmespath,
            "Extractor cannot extract log_probability as log_probability_jmespath_expression is not provided",
        )
        probabilities = self.log_probability_jmespath.search(data)
        self._validate_types(
            values=probabilities,
            type=float,
            num_records=num_records,
            jmespath_expression=self.log_probability_jmespath_expression,  # type: ignore
        )

        return probabilities

    def extract_output(self, data: Union[List, Dict], num_records: int) -> Union[List[str], str]:
        """
        Extract output from JSON model output

        :param data: Model response. The output_jmespath_expression is used to extract the predicted output. The
                     predicted output must be a string
        :param num_records: number of inference records in the model output
        :return: model output
        """
        util.require(
            self.output_jmespath_expression,
            "Extractor cannot extract output as output_jmespath_expression is not provided",
        )
        outputs = self.output_jmespath.search(data)
        self._validate_types(
            values=outputs, type=str, num_records=num_records, jmespath_expression=self.output_jmespath_expression  # type: ignore
        )

        return outputs

    @staticmethod
    def _validate_types(values: Any, type: Type, num_records: int, jmespath_expression: str):
        util.require(values is not None, f"JMESpath {jmespath_expression} could not find any data")
        if num_records == 1:
            util.require(
                isinstance(values, type), f"Extractor found: {values} which does not match expected type {type}"
            )
        else:
            util.require(
                isinstance(values, List)
                and len(values) == num_records
                and all(isinstance(value, type) for value in values),
                f"Extractor found: {values} which does not match expected list of {type}",
            )
