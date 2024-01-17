import ray
import numpy as np
import evaluate as hf_evaluate

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from detoxify import Detoxify
from transformers import pipeline
from fmeval.constants import ColumnNames

TOXIGEN_SCORE_NAME = "toxicity"

DETOXIFY_SCORE_TOXICITY = "toxicity"
DETOXIFY_SCORE_SEVERE_TOXICITY = "severe_toxicity"
DETOXIFY_SCORE_OBSCENE = "obscene"
DETOXIFY_SCORE_IDENTITY_ATTACK = "identity_attack"
DETOXIFY_SCORE_INSULT = "insult"
DETOXIFY_SCORE_THREAT = "threat"
DETOXIFY_SCORE_SEXUAL_EXPLICIT = "sexual_explicit"
DETOXIFY_SCORE_NAMES = [
    DETOXIFY_SCORE_TOXICITY,
    DETOXIFY_SCORE_SEVERE_TOXICITY,
    DETOXIFY_SCORE_OBSCENE,
    DETOXIFY_SCORE_IDENTITY_ATTACK,
    DETOXIFY_SCORE_INSULT,
    DETOXIFY_SCORE_THREAT,
    DETOXIFY_SCORE_SEXUAL_EXPLICIT,
]


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
    COLUMN_NAME = ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value

    def __init__(self, column_name: str = COLUMN_NAME):
        """
        Constructor to locally load the helper model for inference.

        :param column_name: column name used to fetch input texts in __call__ method
        """
        self._model = pipeline("text-classification", model=self.TOXIGEN_MODEL_NAME)
        self._column_name = column_name

    def get_helper_scores(self, text_input: List[str]) -> Dict[str, List[float]]:  # type: ignore[override]
        """
        Method to get scores from ToxigenHelper
        :param text_input: list of text inputs for the model
        :returns: dict with key as score name and value being list of scores for text inputs

        Note: Toxigen scores are for label: LABEL_1
        """
        inference_output = self._model(text_input)
        result = {
            TOXIGEN_SCORE_NAME: [x["score"] if x["label"] == "LABEL_1" else 1.0 - x["score"] for x in inference_output]
        }
        return result

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Call method to allow using this helper as a ray actor.

        :param batch: batch of data to be scored.
        :return: batch with scores added to it.
        """
        scores = self.get_helper_scores(batch[self._column_name].tolist())

        for key, value in scores.items():
            batch.update({key: np.array(value)})
        return batch

    @staticmethod
    def get_score_names() -> List[str]:
        """
        Util method to return name of scores generated by helper model
        :returns: List of score names
        """
        return [TOXIGEN_SCORE_NAME]


class DetoxifyHelperModel(BaseHelperModel):
    """
    Helper model for Detoxify: https://github.com/unitaryai/detoxify

    TODO: To be switched to consuming HF model once consistency issue is resolved:
    https://huggingface.co/unitary/unbiased-toxic-roberta. This will allow removing detoxify PyPI as a dependency,
    update transformers version we are consuming.
    """

    DETOXIFY_MODEL_TYPE = "unbiased"
    COLUMN_NAME = ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value

    def __init__(self, column_name: str = COLUMN_NAME):
        """
        Constructor to locally load the helper model for inference.

        :param column_name: column name used to fetch input texts in __call__ method
        """
        self._model = Detoxify(model_type="unbiased").predict
        self._column_name = column_name

    def get_helper_scores(self, text_input: List[str]) -> Dict[str, List[float]]:  # type: ignore[override]
        """
        Method to get scores from DetoxifyHelper
        :param text_input: list of text inputs for the model
        :returns: dict with keys as score name and value being list of scores for text inputs
        """
        return self._model(text_input)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Call method to allow using this helper as a ray actor.

        :param batch: batch of data to be scored.
        :return: batch with scores added to it.
        """
        scores = self.get_helper_scores(batch[self._column_name].tolist())

        for key, value in scores.items():
            batch.update({key: np.array(value)})
        return batch

    @staticmethod
    def get_score_names() -> List[str]:
        """
        Util method to return name of scores generated by helper model
        :returns: List of score names
        """
        return DETOXIFY_SCORE_NAMES


@ray.remote(num_cpus=1)
class BertscoreHelperModel(BaseHelperModel):
    """
    BERTscore is a similarity-based metric that compares the embedding of the prediction and target sentences
    under a (learned) model, typically, from the BERT family.
    This score may lead to increased flexibility compared to rouge and METEOR in terms of rephrasing since
    semantically similar sentences are (typically) embedded similarly.

    https://huggingface.co/spaces/evaluate-metric/bertscore

    Note: we specify that this Ray actor requires num_cpus=1 in order to limit the number of concurrently
    running tasks or actors to avoid out of memory issues.
    See https://docs.ray.io/en/latest/ray-core/patterns/limit-running-tasks.html#core-patterns-limit-running-tasks
    for a detailed explanation.
    """

    def __init__(self, model_type: str):  # pragma: no cover
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
        # Note: the following code is covered by unit tests,
        # but since it gets executed by Ray, Mypy marks it
        # as not covered.
        return self._bertscore.compute(  # pragma: no cover
            predictions=[model_output],
            references=[target_output],
            model_type=self._model_type,
        )["f1"][0]
