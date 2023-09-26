from abc import ABC, abstractmethod
from typing import Dict, Tuple, List


class ModelRunner(ABC):
    """
    This class is responsible for running the model and extracting the model output.

    It handles everything related to the model, including: model deployment, payload construction for invocations,
    and making sense of the model output.
    """

    @abstractmethod
    def predict(self, prompt: str) -> Tuple[str, float]:
        """
        Runs the model on the given data. Output and log-probability will both be written
        to the `prediction_column_name` column as dictionary. It returns a tuple of:

        ```
        (
            "output": str
            "input_log_probability": float
        )
        ```
        The output values are extracted using the `output` parameter provided in the model settings.
        Similarly, probability values are extracted from the model response using the `probability` parameter.

        :param data: the dataset where each instance is a row in the dataset.
        :param prompt: the prompt
        :return: the dictionary
        """

    @abstractmethod
    def batch_predict(self, prompts: List[str]) -> List[Tuple[str, float]]:
        """

        :param prompts:
        :return:
        """
