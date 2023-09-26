import logging
import numpy as np

from typing import List, Union, Dict, Any, Optional

from data_loaders.utils.util import (
    get_nested_list_levels,
    has_only_valid_types_in_nested_list,
    validate_list_with_same_size_item,
    is_nested_list,
    has_only_valid_types,
)
from data_loaders.utils.jmespath_util import compile_jmespath
from model_runners.extractors.extractor_base import Extractor


logger = logging.getLogger(__name__)


class JsonExtractor(Extractor):
    """
    JSON model response extractor
    """

    def __init__(self, probability_jmespath_expression: str):
        """
        :param probability_jmespath_expression: JMESPath expression of probability
        """
        self.probability_jmespath_expression = probability_jmespath_expression
        self.probability_jmespath = compile_jmespath(probability_jmespath_expression)
        self.output_jmespath_expression = output_jmespath_expression
        self.output_jmespath = compile_jmespath(output_jmespath_expression)
        self.label_headers_jmespath_expression = label_headers_jmespath_expression
        self.label_headers_jmespath = compile_jmespath(label_headers_jmespath_expression)

    def extract_probability(self, data: Union[List, Dict], num_records: int) -> List:
        """
        Extract probabilities from JSON model output

        :param data: Model response. The probability_jmespath_expression is used to extract the predicted softmax
                     probabilities. Each record in the extracted probabilities will be a list of list of floats.
                     Examples for the extracted probabilities:
                        - data: [[0.6],[0.3],[0.1]], num_records: 1
                        - data: [ [[0.6],[0.3],[0.1]] ], num_records: 1
                        - data: [ [[0.6]], [[0.1]] ], num_records: 2
        :param num_records: number of inference records in the model output
        :return: list of a list, each element is a list of probabilities.
        """
        probabilities = self.probability_jmespath.search(data)
        self._validate_probability(data, probabilities, num_records)

        return probabilities

    def _validate_probability(self, data: Union[List, Dict], extracted: Any, num_records: int) -> None:
        """
        Validate whether the extracted probability follows the expected probability format

        :param extracted: extracted probability
        :param data: model response
        :param num_records: number of inference records in the model output
        """
        valid_types = {float}

        levels = get_nested_list_levels(extracted)
        # Case: only one record, e.g. [[0.1, 0.9], [0.2, 0.8]]
        if levels == 2:
            require(
                has_only_valid_types_in_nested_list(extracted, valid_types),
                "Invalid probability value types in '{}' in model prediction '{}'".format(extracted, data),
            )
        # Case: multiple records, e.g. [[[0.1, 0.9], [0.2, 0.8]], [[0.3, 0.7], [0.2, 0.8]]]
        elif levels == 3:
            for probs in extracted:
                require(
                    has_only_valid_types_in_nested_list(probs, valid_types),
                    "Invalid probability value types in '{}' in model prediction '{}'".format(probs, data),
                )
        else:
            raise UserError(
                "Invalid format of probability '{}' extracted by '{}' in model prediction '{}'".format(
                    extracted, self.probability_jmespath_expression, data
                )
            )

        actual_num_records = 1 if levels == 2 else len(extracted)
        # validate number of records in the extracted probability
        require(
            actual_num_records == num_records,
            "Unable to derive the predicted probabilities from model prediction `{}` "
            "because prediction record number {} doesn't match num_records {}.".format(
                data, actual_num_records, num_records
            ),
        )

    def extract_output(self, data: Union[List, Dict], num_records: int) -> Union[List, str, float]:
        """
        Extract output from JSON model output

        :param data: Model response. The output_jmespath_expression is used to
          extract the predicted output. The predicted output can be one of these,
          - str - for tasks like Summarization, Q&A etc Union[str, List[str]]  and (len(List[str]) == 1)
          - float for regression. Union[float, List[float]] and (len(List[float]) == 1)
          - List[float] - for Classification. In this use case, we have to infer the label based on the highest probability. List[float] and len(List[float]) > 1
        :param num_records: number of inference records in the model output
        :return: model output
        """
        output = self.output_jmespath.search(data)
        self._validate_output(data, output, num_records)

        if self.label_headers_extractor:
            # case: classification
            output = self._get_classification_result(output, data, num_records)

        return output

    def _validate_output(self, data: Union[List, Dict], extracted: Any, num_records: int):
        """
        Validate extracted output

        :param extracted: extracted output
        :param data: model response
        :param num_records: number of inference records in the model output
        """
        if self.label_headers_extractor is None:
            # case: non classification
            is_valid = self._validate_text(extracted, num_records) or self._validate_non_classification_output_float(
                extracted, num_records
            )
        else:
            # case: classification
            is_valid = self._validate_classification_output_float(extracted, num_records)

        require(
            is_valid,
            "Invalid output '{}' extracted by '{}' in model prediction '{}' with records number {}".format(
                extracted, self.output_jmespath_expression, data, num_records
            ),
        )

    @staticmethod
    def _validate_text(extracted: Any, num_records: int) -> bool:
        """
        Validate if the extracted data is valid text output

        :param extracted: extracted output
        :param num_records: number of inference records in the model output
        :return: whether extracted output is valid
        """
        valid_types = {str}
        # case: str
        if isinstance(extracted, str):
            return num_records == 1
        # case: [str]
        elif isinstance(extracted, list) and has_only_valid_types(extracted, valid_types):
            return num_records == len(extracted)
        # case: [[str1], [str2]]
        elif (
            is_nested_list(extracted)
            and has_only_valid_types_in_nested_list(extracted, valid_types)
            # check invalid case: [["text1"], ["text2", "text3"]]
            and validate_list_with_same_size_item(extracted, 1)
        ):
            return num_records == len(extracted)

        return False

    @staticmethod
    def _validate_classification_output_float(extracted: Any, num_records: int) -> bool:
        """
        Validate if the extracted data is valid classification output float

        :param extracted: extracted output
        :param num_records: number of inference records in the model output
        :return: whether extracted output is valid
        """
        valid_types = {float}
        # case: [float1, float2]
        if isinstance(extracted, list) and has_only_valid_types(extracted, valid_types) and len(extracted) > 1:
            return num_records == 1
        # case: [[float1, float2], [float3, float4]]
        elif (
            is_nested_list(extracted)
            and has_only_valid_types_in_nested_list(extracted, valid_types)
            and validate_list_with_same_size_item(extracted)
            and len(extracted[0]) > 1
        ):
            return num_records == len(extracted)

        return False

    @staticmethod
    def _validate_non_classification_output_float(extracted: Any, num_records: int) -> bool:
        """
        Validate if the extracted data is valid none classification output float

        :param extracted: extracted output
        :param num_records: number of inference records in the model output
        :return: whether extracted output is valid
        """
        valid_types = {float}
        # case: float
        if isinstance(extracted, float):
            return num_records == 1
        # case: [float1, float2]
        elif isinstance(extracted, list) and has_only_valid_types(extracted, valid_types):
            return num_records == len(extracted)
        # case: [[float1], [float2]]
        elif (
            is_nested_list(extracted)
            and has_only_valid_types_in_nested_list(extracted, valid_types)
            and len(extracted[0]) == 1
        ):
            return num_records == len(extracted)

        return False

    def _get_classification_result(self, output: List, data: Union[List, Dict], num_records: int) -> Union[List, str]:
        """
        create the classificaiton result with output and header labels extraction

        :param output: output extracted from model response
        :param data: model response
        :param num_records: number of inference records in the model outpu
        :return: classification result
        """
        output_levels = get_nested_list_levels(output)
        label_headers = self.label_headers_extractor.extract(data, num_records)  # type: ignore
        headers_levels = get_nested_list_levels(label_headers)

        require(
            output_levels == headers_levels,
            "Failed to get classification result: output '{}' and headers '{}' are not in the same format".format(
                output, label_headers
            ),
        )

        result = None
        # case: output [0.1, 0.9], label_headers ["cat", "dog"]
        if output_levels == 1 and len(output) == len(label_headers):
            max_indices = np.argmax(output)
            result = label_headers[max_indices]
        # case: output [[0.1, 0.9], [0.2, 0.8]], label_headers [["cat", "dog"], ["dog", "bird"]]
        elif output_levels == 2 and len(output[0]) == len(label_headers[0]):
            max_indices = np.argmax(output, axis=1)
            result = [label_headers[row][max_index] for row, max_index in enumerate(max_indices)]  # type: ignore

        if not result:
            raise UserError(
                "Failed to get classification result: output '{}' is not complied with label headers '{}'".format(
                    output, label_headers
                )
            )

        return result

    def extract_label_headers(self, data: Union[List, Dict], num_records: int) -> List[str]:
        """
        Extract output from JSON model output

        :param data: Model response. The label_headers_jmespath_expression is used to
          extract the predicted classification label headers. The extracted probabilities will be a list of strings and list size is > 1.
        :param num_records: number of inference records in the model output
        :return: list of labels
        """
        label_headers = self.label_headers_jmespath.search(data)
        self._validate_label_headers(data, label_headers, num_records)
        return label_headers

    def _validate_label_headers(self, data: Union[List, Dict], extracted: Any, num_records: int):
        """
        Validate if the extracted label headers

        :param extracted: extracted label headers
        :param data: model response
        :param num_records: number of inference records in the model output
        """
        valid = False
        valid_types = {str}

        if (
            isinstance(extracted, list)
            and has_only_valid_types(extracted, valid_types)
            and len(extracted) > 1
            and num_records == 1  # prediction records should match number of records
        ):
            valid = True
        elif (
            is_nested_list(extracted)
            and has_only_valid_types_in_nested_list(extracted, valid_types)
            and validate_list_with_same_size_item(extracted)
            and len(extracted[0]) > 1
            and len(extracted) == num_records  # prediction records should match number of records
        ):
            valid = True

        require(
            valid,
            "Invalid label headers '{}' extracted by '{}' in model prediction '{}' with num_records '{}'".format(
                extracted, self.label_headers_jmespath_expression, data, num_records
            ),
        )
