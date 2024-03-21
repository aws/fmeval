import torch
import transformers
import evaluate as hf_evaluate

from enum import Enum
from typing import Dict, List
from transformers import pipeline, AutoConfig


class ToxigenModel:
    """Toxigen helper model.

    See https://huggingface.co/tomh/toxigen_roberta/tree/main
    """

    MODEL_NAME = "tomh/toxigen_roberta"
    SCORE_NAME = "toxicity"

    def __init__(self):
        """Load the helper model into memory."""
        self._model = pipeline("text-classification", model=ToxigenModel.MODEL_NAME)

    def invoke_model(self, text_inputs: List[str]) -> Dict[str, List[float]]:
        """Get Toxigen scores by invoking self._model on a list of text inputs.

        Note: Toxigen scores are for the label "LABEL_1".

        :param text_inputs: A list of text inputs for the model.
        :returns: A dict mapping score name to a list of scores for each of the text inputs.
        """
        inference_output = self._model(text_inputs)
        return {
            ToxigenModel.SCORE_NAME: [
                x["score"] if x["label"] == "LABEL_1" else 1.0 - x["score"] for x in inference_output
            ]
        }

    def __reduce__(self):
        """Serializer method."""
        return self.__class__, ()


class DetoxifyModel:
    """Detoxify helper model.

    See https://github.com/unitaryai/detoxify

    Note: we load the unbiased model directly from the state dict due to dependency conflicts between detoxify and
    transformers libraries.

    TODO: To be switched to consuming HF model once consistency issue is resolved:
    https://huggingface.co/unitary/unbiased-toxic-roberta. This will allow removing detoxify PyPI as a dependency,
    update transformers version we are consuming.
    """

    UNBIASED_MODEL_URL = (
        "https://github.com/unitaryai/detoxify/releases/download/v0.3-alpha/toxic_debiased-c7548aa0.ckpt"
    )
    TOXICITY_SCORE = "toxicity"
    SEVERE_TOXICITY_SCORE = "severe_toxicity"
    OBSCENE_SCORE = "obscene"
    IDENTITY_ATTACK_SCORE = "identity_attack"
    INSULT_SCORE = "insult"
    THREAT_SCORE = "threat"
    SEXUAL_EXPLICIT_SCORE = "sexual_explicit"
    SCORE_NAMES = [
        TOXICITY_SCORE,
        SEVERE_TOXICITY_SCORE,
        OBSCENE_SCORE,
        IDENTITY_ATTACK_SCORE,
        INSULT_SCORE,
        THREAT_SCORE,
        SEXUAL_EXPLICIT_SCORE,
    ]

    def __init__(self):
        """Load the helper model into memory."""
        state_dict = torch.hub.load_state_dict_from_url(DetoxifyModel.UNBIASED_MODEL_URL, map_location="cpu")
        config = state_dict["config"]["arch"]["args"]
        self._model = (
            getattr(transformers, config["model_name"])
            .from_pretrained(
                pretrained_model_name_or_path=None,
                config=AutoConfig.from_pretrained(config["model_type"], num_labels=config["num_classes"]),
                state_dict=state_dict["state_dict"],
                local_files_only=False,
            )
            .to("cpu")
        )
        self._tokenizer = getattr(transformers, config["tokenizer_name"]).from_pretrained(config["model_type"])

    def invoke_model(self, text_inputs: List[str]) -> Dict[str, List[float]]:
        """Get Detoxify scores by invoking self._model on a list of text inputs.

        :param text_inputs: A list of text inputs for the model.
        :returns: A dict mapping score name to a list of scores for each of the text inputs.
        """
        inputs = self._tokenizer(text_inputs, return_tensors="pt", truncation=True, padding=True).to(self._model.device)
        scores = torch.sigmoid(self._model(**inputs)[0]).cpu().detach().numpy()
        return {
            score_name: [score[i].tolist() for score in scores]
            for i, score_name in enumerate(DetoxifyModel.SCORE_NAMES)
        }

    def __reduce__(self):
        """Serializer method."""
        return self.__class__, ()


class BertscoreModel:
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

    def __init__(self, model_type: str):
        """Load the HuggingFace bertscore metric.

        :param model_type: Model type to be used for bertscore
        """
        self._bertscore = hf_evaluate.load("bertscore")
        self._model_type = model_type

    def invoke_model(self, target_output: str, model_output: str) -> float:
        """Invoke the helper model to obtain a BERT score.

        :param target_output: Reference text.
        :param model_output: Model output text.
        :returns: The computed BERT score.
        """
        return self._bertscore.compute(
            predictions=[model_output],
            references=[target_output],
            model_type=self._model_type,
        )["f1"][0]

    def __reduce__(self):
        """Serializer method."""
        return self.__class__, (self._model_type,)


class BertscoreModelTypes(Enum):
    """This class holds the names of all the allowed models for computing the BERTScore."""

    MICROSOFT_DEBERTA_MODEL = "microsoft/deberta-xlarge-mnli"
    ROBERTA_MODEL = "roberta-large-mnli"

    @classmethod
    def model_is_allowed(cls, model_name: str) -> bool:
        """
        Given a model name like 'roberta-large-mnli', check if this is an allowed model for computing BERTScore.
        """
        return any(elem.value == model_name for elem in iter(cls))

    @classmethod
    def model_list(cls) -> List[str]:
        """
        Return a list of all the allowed models for computing BERTScore.
        """
        return [elem.value for elem in iter(cls)]
