from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


class BaseHelperModel(ABC):
    """
    Base class for 3P helper model invoker. Note: These Helper models are inherently
    Machine learning models being used by Evaluation algorithms.
    """

    @abstractmethod
    def invoke(self, text_input: str) -> Any:
        """
        Method to invoke helper model
        :param text_input: model text input
        :returns: model output
        """


class ToxigenHelperModel(BaseHelperModel):
    """
    Helper model for toxigen model: https://huggingface.co/tomh/toxigen_roberta/tree/main
    """

    def __init__(self, model_path: str):
        """
        Constructor to locally load the helper model for inference.
        :param model_path: local path of model artifacts
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        self._pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def invoke(self, text_input: str) -> List[Dict[str, Union[str, float]]]:
        """
        Method to invoke helper model
        :param text_input: model text input
        :returns: model output
        """
        return self._pipeline(text_input)
