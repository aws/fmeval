import logging
from typing import Dict, List, Union
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class Extractor(ABC):
    """
    Interface class for model response extractors.
    """

    @abstractmethod
    def extract_log_probability(self, data: Union[List, Dict], num_records: int) -> Union[List[float], float]:
        """
        Extract log probability from model response.

        :param data: Model response.
        :return: A list of lists, where each element is a list of probabilities.
        """

    @abstractmethod
    def extract_output(self, data: Union[List, Dict], num_records: int) -> Union[List[str], str]:
        """
        Extract output from model response.

        :param data: Model response.
        :return: model output
        """

    @abstractmethod
    def extract_embedding(self, data: Union[List, Dict], num_records: int) -> Union[List[List[float]], List[float]]:
        """
        Extract embedding from model response.

        :param data: Model response.
        :return: model output
        """
