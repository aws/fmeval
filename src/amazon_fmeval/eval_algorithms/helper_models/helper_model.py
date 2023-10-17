from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import evaluate as hf_evaluate

from amazon_fmeval.util import singleton


class BaseHelperModel(ABC):
    """
    Base class for 3P helper model invoker. Note: These Helper models are inherently
    Machine learning models being used by Evaluation algorithms.
    """

    @abstractmethod
    def get_helper_scores(self, text_input: str) -> Any:
        """
        Method to invoke helper model
        :param text_input: model text input
        :returns: model output
        """


class ToxigenHelperModel(BaseHelperModel):
    """
    Helper model for toxigen model: https://huggingface.co/tomh/toxigen_roberta/tree/main
    """
    TOXIGEN_MODEL_NAME = "tomh/toxigen_roberta"
    SCORE_NAME = "toxicity"

    def __init__(self):
        """
        Constructor to locally load the helper model for inference.
        """
        self._pipeline = pipeline("text-classification", model=self.TOXIGEN_MODEL_NAME)

    def get_helper_scores(self, text_input: List[str]) -> Dict[str, np.ndarray]:
        """
        Method to get scores from ToxigenHelper
        :param text_input: list of text inputs for the model
        :returns: dict with key as score name and value being list of scores for text inputs

        Note: Toxigen scores are for label: LABEL_1
        """
        inference_output = self._pipeline(text_input)
        result = {
            self.SCORE_NAME: np.array([x["score"] if x["label"] == "LABEL_1" else 1.0 - x["score"]
                                       for x in inference_output])
        }
        return result

    def __call__(self, batch: Dict[str, np.ndarray], column_name: str) -> Dict[str, np.ndarray]:
        """
        Call method to allow using this helper as a ray actor.

        :param batch: batch of data to be scored.
        :param column_name: Column name for input texts to helper model
        :return: batch with scores added to it.
        """
        scores = self.get_helper_scores(batch[column_name].tolist())

        batch.update(scores)
        return batch


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

    def get_helper_scores(self, target_output: str, model_output: str) -> float:  # type: ignore[override]
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
