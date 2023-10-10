from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import evaluate as hf_evaluate

from amazon_fmeval.util import singleton


class BaseHelperModel(ABC):
    """
    Base class for 3P helper model invoker. Note: These Helper models are inherently
    Machine learning models being used by Evaluation algorithms.
    """

    @abstractmethod
    def get_helper_score(self, text_input: str) -> Any:
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

    def get_helper_score(self, text_input: str) -> List[Dict[str, Union[str, float]]]:
        """
        Method to invoke helper model
        :param text_input: model text input
        :returns: model output
        """
        return self._pipeline(text_input)


@singleton
class BertscoreHelperModel(BaseHelperModel):
    """
    BERTscore is a similarity-based metric that compares the embedding of the prediction and target sentences
    under a (learned) model, typically, from the BERT family.
    This score may lead to increased flexibility compared to rouge and METEOR in terms of rephrasing since
    semantically similar sentences are (typically) embedded similarly.

    https://huggingface.co/spaces/evaluate-metric/bertscore
    """

    def __init__(self, model_type: str):
        """
        Default constructor

        :param model_type: Model type to be used for bertscore
        """
        self._bertscore = hf_evaluate.load("bertscore")
        self._model_type = model_type

        # Dummy call to download the model within constructor
        self._bertscore.compute(
            predictions=["dummy_prediction"],
            references=["dummy_reference"],
            model_type=self._model_type,
        )

    def get_helper_score(self, target_output: str, model_output: str) -> float:  # type: ignore[override]
        """
        Method to invoke the concerned model and get bertscore
        :param target_output: Reference text
        :model_output: Model prediction text
        """
        return self._bertscore.compute(
            predictions=[model_output],
            references=[target_output],
            model_type=self._model_type,
        )["f1"][0]
