import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Type

from functional import seq


@dataclass(frozen=True)
class EvalScore:
    """
    The class that contains the aggregated scores computed for different eval offerings

    :param name: The name of the eval score offering
    :param value: The aggregated score computed for the given eval offering
    """

    name: str
    value: float

    def __eq__(self, other: Type["EvalScore"]):  # type: ignore[override]
        try:
            assert self.name == other.name
            assert math.isclose(self.value, other.value)
            return True
        except AssertionError:
            return False


class EvalAlgorithm(Enum):
    """The evaluation types supported by Amazon Foundation Model Evaluations.

    The evaluation types are used to determine the evaluation metrics for the
    model.
    """

    PROMPT_STEREOTYPING = "prompt_stereotyping"
    FACTUAL_KNOWLEDGE = "factual_knowledge"
    TOXICITY = "toxicity"
    SEMANTIC_ROBUSTNESS = "semantic_robustness"
    ACCURACY = "accuracy"
    QA_ACCURACY = "qa_accuracy"
    SUMMARIZATION_ACCURACY = "summarization_accuracy"

    def __str__(self):
        """
        Returns a prettified name
        """
        return self.name.replace("_", " ")


@dataclass(frozen=True)
class CategoryScore:
    """The class that contains the aggregated scores computed across specific categories in the dataset.

    :param name: The name of the category.
    :param scores: The aggregated score computed for the given category.
    """

    name: str
    scores: List[EvalScore]

    def __eq__(self, other: Type["CategoryScore"]):  # type: ignore[override]
        try:
            assert self.name == other.name
            assert len(self.scores) == len(other.scores)
            assert seq(self.scores).sorted(key=lambda score: score.name).zip(
                seq(other.scores).sorted(key=lambda score: score.name)
            ).filter(lambda item: item[0] == item[1]).len() == len(self.scores)
            return True
        except AssertionError:
            return False


@dataclass(frozen=True)
class EvalOutput:
    """
    The class that contains evaluation scores from `EvalAlgorithmInterface`.

    :param eval_name: The name of the evaluation
    :param dataset_name: The name of dataset used by eval_algo
    :param prompt_template: A template used to compose prompts, only consumed if model_output is not provided in dataset
    :param dataset_scores: The aggregated score computed across the whole dataset.
    :param category_scores: A list of CategoryScore object that contain the scores for each category in the dataset.
    :param output_path: Local path of eval output on dataset. This output contains prompt-response with
    record wise eval scores
    """

    eval_name: str
    dataset_name: str
    dataset_scores: List[EvalScore]
    prompt_template: Optional[str] = None
    category_scores: Optional[List[CategoryScore]] = None
    output_path: Optional[str] = None

    def __post_init__(self):  # pragma: no cover
        """Post initialisation validations for EvalOutput"""
        if not self.category_scores:
            return

        dataset_score_names = [eval_score.name for eval_score in self.dataset_scores]
        if self.category_scores:
            for category_score in self.category_scores:
                assert len(category_score.scores) == len(self.dataset_scores)
                assert dataset_score_names == [
                    category_eval_score.name for category_eval_score in category_score.scores
                ]

    def __eq__(self, other: Type["EvalOutput"]):  # type: ignore[override]
        try:
            assert self.eval_name == other.eval_name
            assert self.dataset_name == other.dataset_name
            assert self.prompt_template == other.prompt_template
            assert self.dataset_scores if other.dataset_scores else not self.dataset_scores
            assert len(self.dataset_scores) == len(other.dataset_scores)
            assert seq(self.dataset_scores).sorted(key=lambda x: x.name).zip(
                seq(other.dataset_scores).sorted(key=lambda x: x.name)
            ).filter(lambda x: x[0] == x[1]).len() == len(self.dataset_scores)
            assert self.category_scores if other.category_scores else not self.category_scores
            if self.category_scores:
                assert seq(self.category_scores).sorted(key=lambda cat_score: cat_score.name).zip(
                    seq(other.category_scores).sorted(key=lambda cat_score: cat_score.name)
                ).filter(lambda item: item[0] == item[1]).len() == len(self.category_scores)
            return True
        except AssertionError:
            return False
