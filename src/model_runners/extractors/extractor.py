import logging
from typing import List, Union, Dict, Any, Optional
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)

class Extractor(ABC):
    """
    Interface class for model response extractors.
    """

    @abstractmethod
    def extract(self, data: Union[List, Dict], num_records: int):
        """
        Extract data from model response.

        :param data: Model response.
        :param num_records: number of inference records in the model output
        """
        raise NotImplementedError

    @abstractmethod
    def extract_probability(self, data: Union[List, Dict], num_records: int) -> Optional[List]:
        """
        Extract probability from model response.

        :param data: Model response.
        :return: A list of lists, where each element is a list of probabilities.
        """

    @abstractmethod
    def extract_output(self, data: Union[List, Dict], num_records: int) -> Optional[Union[List, str, float]]:
        """
        Extract output from model response.

        :param data: Model response.
        :return: model output
        """

    @abstractmethod
    def _validate(self, data: Union[List, Dict], extracted: Any, num_records: int):
        """
        Validate if the target data is properly extracted.

        :param extracted: extracted data
        :param data: model response
        :param num_records: number of inference records in the model output
        """
        raise NotImplementedError
