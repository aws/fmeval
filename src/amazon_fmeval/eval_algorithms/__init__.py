import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Type, Dict, Tuple

from functional import seq

from amazon_fmeval.constants import MIME_TYPE_JSONLINES
from amazon_fmeval.data_loaders.data_config import DataConfig


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
    GENERAL_SEMANTIC_ROBUSTNESS = "general_semantic_robustness"
    ACCURACY = "accuracy"
    QA_ACCURACY = "qa_accuracy"
    QA_ACCURACY_SEMANTIC_ROBUSTNESS = "qa_accuracy_semantic_robustness"
    SUMMARIZATION_ACCURACY = "summarization_accuracy"
    CLASSIFICATION_ACCURACY = "classification_accuracy"
    SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS = "summarization_accuracy_semantic_robustness"

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
    :param error: A string error message for a failed evaluation.
    """

    eval_name: str
    dataset_name: str
    dataset_scores: Optional[List[EvalScore]] = None
    prompt_template: Optional[str] = None
    category_scores: Optional[List[CategoryScore]] = None
    output_path: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self):  # pragma: no cover
        """Post initialisation validations for EvalOutput"""
        assert self.dataset_scores or self.error

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
            assert self.error == other.error
            assert self.dataset_scores if other.dataset_scores else not self.dataset_scores
            if self.dataset_scores:  # pragma: no branch
                assert self.dataset_scores and other.dataset_scores
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


class ModelTask(Enum):
    """The different types of tasks that are supported by the evaluations.

    The model tasks are used to determine the evaluation metrics for the
    model.
    """

    NO_TASK = "no_task"
    CLASSIFICATION = "classification"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"


# These mappings are not to be consumed for any use cases and is for representational purposes.
# NO_TASK should have all keys from EvalAlgorithm
MODEL_TASK_EVALUATION_MAP = {
    ModelTask.NO_TASK: [
        EvalAlgorithm.PROMPT_STEREOTYPING,
        EvalAlgorithm.FACTUAL_KNOWLEDGE,
        EvalAlgorithm.TOXICITY,
        EvalAlgorithm.GENERAL_SEMANTIC_ROBUSTNESS,
    ],
    ModelTask.CLASSIFICATION: [
        EvalAlgorithm.CLASSIFICATION_ACCURACY,
    ],
    ModelTask.QUESTION_ANSWERING: [
        EvalAlgorithm.TOXICITY,
        EvalAlgorithm.QA_ACCURACY,
        EvalAlgorithm.QA_ACCURACY_SEMANTIC_ROBUSTNESS,
    ],
    ModelTask.SUMMARIZATION: [
        EvalAlgorithm.TOXICITY,
        EvalAlgorithm.SUMMARIZATION_ACCURACY,
        EvalAlgorithm.SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS,
    ],
}

# Constants for Built-in dataset names
TREX = "trex"
BOOLQ = "boolq"
TRIVIA_QA = "trivia_qa"
NATURAL_QUESTIONS = "natural_questions"
CROW_PAIRS = "crow-pairs"
CNN_DAILY_MAIL = "cnn_daily_mail"
XSUM = "xsum"
IMDB_MOVIE_REVIEWS = "imdb_movie_reviews"
WOMENS_CLOTHING_ECOMMERCE_REVIEWS = "womens_clothing_ecommerce_reviews"
BOLD = "bold"
WIKITEXT2 = "wikitext2"
REAL_TOXICITY_PROMPTS = "real_toxicity_prompts"
REAL_TOXICITY_PROMPTS_CHALLENGING = "real_toxicity_prompts_challenging"

# Mapping of Eval algorithms and corresponding Built-in datasets
EVAL_DATASETS: Dict[str, List[str]] = {
    EvalAlgorithm.FACTUAL_KNOWLEDGE.value: [TREX],
    EvalAlgorithm.QA_ACCURACY.value: [BOOLQ, TRIVIA_QA, NATURAL_QUESTIONS],
    EvalAlgorithm.QA_ACCURACY_SEMANTIC_ROBUSTNESS.value: [BOOLQ, TRIVIA_QA, NATURAL_QUESTIONS],
    EvalAlgorithm.PROMPT_STEREOTYPING.value: [CROW_PAIRS],
    EvalAlgorithm.SUMMARIZATION_ACCURACY.value: [CNN_DAILY_MAIL, XSUM],
    EvalAlgorithm.GENERAL_SEMANTIC_ROBUSTNESS.value: [BOLD, TREX, WIKITEXT2],
    EvalAlgorithm.CLASSIFICATION_ACCURACY.value: [IMDB_MOVIE_REVIEWS],  # WOMENS_CLOTHING_ECOMMERCE_REVIEWS
    EvalAlgorithm.SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS.value: [CNN_DAILY_MAIL, XSUM],
    EvalAlgorithm.TOXICITY.value: [BOLD, REAL_TOXICITY_PROMPTS, REAL_TOXICITY_PROMPTS_CHALLENGING],
}

# Mapping of Default Prompt Template corresponding to eval, built-in dataset pair
# TODO: To be correctly populated
EVAL_PROMPT_TEMPLATES: Dict[Tuple[str, str], str] = {
    (EvalAlgorithm.FACTUAL_KNOWLEDGE.value, TREX): "Answer: $feature",
    (EvalAlgorithm.QA_ACCURACY.value, BOOLQ): "$feature",
    (EvalAlgorithm.QA_ACCURACY.value, TRIVIA_QA): "$feature",
    (EvalAlgorithm.QA_ACCURACY.value, NATURAL_QUESTIONS): "$feature",
    (EvalAlgorithm.QA_ACCURACY_SEMANTIC_ROBUSTNESS.value, BOOLQ): "$feature",
    (EvalAlgorithm.QA_ACCURACY_SEMANTIC_ROBUSTNESS.value, TRIVIA_QA): "$feature",
    (EvalAlgorithm.QA_ACCURACY_SEMANTIC_ROBUSTNESS.value, NATURAL_QUESTIONS): "$feature",
    (EvalAlgorithm.PROMPT_STEREOTYPING.value, CROW_PAIRS): "$feature",
    (EvalAlgorithm.SUMMARIZATION_ACCURACY.value, CNN_DAILY_MAIL): "Summarise: $feature",
    (EvalAlgorithm.SUMMARIZATION_ACCURACY.value, XSUM): "Summarise: $feature",
    (EvalAlgorithm.CLASSIFICATION_ACCURACY.value, IMDB_MOVIE_REVIEWS): "$feature",
    (EvalAlgorithm.CLASSIFICATION_ACCURACY.value, WOMENS_CLOTHING_ECOMMERCE_REVIEWS): "$feature",
    (EvalAlgorithm.GENERAL_SEMANTIC_ROBUSTNESS.value, BOLD): "$feature",
    (EvalAlgorithm.GENERAL_SEMANTIC_ROBUSTNESS.value, TREX): "$feature",
    (EvalAlgorithm.GENERAL_SEMANTIC_ROBUSTNESS.value, WIKITEXT2): "$feature",
    (EvalAlgorithm.SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS.value, CNN_DAILY_MAIL): "Summarise: $feature",
    (EvalAlgorithm.SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS.value, XSUM): "Summarise: $feature",
    (EvalAlgorithm.TOXICITY.value, BOLD): "$feature",
    (EvalAlgorithm.TOXICITY.value, REAL_TOXICITY_PROMPTS): "$feature",
    (EvalAlgorithm.TOXICITY.value, REAL_TOXICITY_PROMPTS_CHALLENGING): "$feature",
}

# Mapping of Built-in dataset names and their DataConfigs
# TODO: To be updated once datasets are uploaded in S3, update Configs accordingly
DATASET_CONFIGS: Dict[str, DataConfig] = {
    TREX: DataConfig(
        dataset_name=TREX,
        dataset_uri="s3://amazon-fmeval/datasets/trex/trex.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="question",
        target_output_location="answers",
        category_location="knowledge_category",
    ),
    BOOLQ: DataConfig(
        dataset_name=BOOLQ,
        dataset_uri="s3://amazon-fmeval/datasets/boolq/boolq.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="question",
        target_output_location="answer",
    ),
    TRIVIA_QA: DataConfig(
        dataset_name=TRIVIA_QA,
        dataset_uri="s3://amazon-fmeval/datasets/triviaQA/triviaQA.json",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="question",
        target_output_location="answer",
    ),
    NATURAL_QUESTIONS: DataConfig(
        dataset_name=NATURAL_QUESTIONS,
        dataset_uri="s3://amazon-fmeval/datasets/natural_questions/natural_questions.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="question",
        target_output_location="answer",
    ),
    CROW_PAIRS: DataConfig(
        dataset_name=CROW_PAIRS,
        dataset_uri="s3://amazon-fmeval/datasets/crow-pairs/crow-pairs.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        sent_more_input_location="sent_more",
        sent_less_input_location="sent_less",
        category_location="bias_type",
    ),
    CNN_DAILY_MAIL: DataConfig(
        dataset_name=CNN_DAILY_MAIL,
        dataset_uri="s3://amazon-fmeval/datasets/cnn_dailymail/cnn_dailymail.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="document",
        target_output_location="summary",
    ),
    XSUM: DataConfig(
        dataset_name=XSUM,
        dataset_uri="s3://amazon-fmeval/datasets/xsum/xsum.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="document",
        target_output_location="summary",
    ),
    IMDB_MOVIE_REVIEWS: DataConfig(
        dataset_name=IMDB_MOVIE_REVIEWS,
        dataset_uri="s3://amazon-fmeval/datasets/imdb_reviews/imdb_movie_reviews.json",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="text",
        target_output_location="sentiment",
    ),
    # TODO replace dummy link
    WOMENS_CLOTHING_ECOMMERCE_REVIEWS: DataConfig(
        dataset_name=WOMENS_CLOTHING_ECOMMERCE_REVIEWS,
        dataset_uri="dummy link",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="Review Text",
        target_output_location="Recommended IND",
        model_output_location=None,
        category_location="Class Name",
    ),
    # TODO to be populated
    BOLD: DataConfig(
        dataset_name=BOLD,
        dataset_uri="dummy link",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="tba",
        target_output_location="tba",
        model_output_location="tba",
        category_location="tba",
    ),
    # TODO to be populated
    WIKITEXT2: DataConfig(
        dataset_name=WIKITEXT2,
        dataset_uri="dummy link",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="tba",
        target_output_location="tba",
        model_output_location="tba",
        category_location="tba",
    ),
    # TODO to be populated
    REAL_TOXICITY_PROMPTS: DataConfig(
        dataset_name=REAL_TOXICITY_PROMPTS,
        dataset_uri="dummy link",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="tba",
        target_output_location="tba",
        model_output_location="tba",
        category_location="tba",
    ),
    # TODO to be populated
    REAL_TOXICITY_PROMPTS_CHALLENGING: DataConfig(
        dataset_name=REAL_TOXICITY_PROMPTS_CHALLENGING,
        dataset_uri="dummy link",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="tba",
        target_output_location="tba",
        model_output_location="tba",
        category_location="tba",
    ),
}
