import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Type, Dict
from functional import seq

from fmeval.constants import MIME_TYPE_JSONLINES, ABS_TOL
from fmeval.data_loaders.data_config import DataConfig


@dataclass(frozen=True)
class EvalScore:
    """
    The class that contains the aggregated scores computed for different eval offerings

    :param name: The name of the eval score offering
    :param value: The aggregated score computed for the given eval offering
    :param error: A string error message for a failed evaluation.
    """

    name: str
    value: Optional[float] = None
    error: Optional[str] = None

    def __post_init__(self):  # pragma: no cover
        """Post initialisation validations for EvalScore"""
        assert self.value is not None or self.error is not None

    def __eq__(self, other: Type["EvalScore"]):  # type: ignore[override]
        try:
            assert self.name == other.name
            if self.value is not None and other.value is not None:
                assert math.isclose(self.value, other.value, abs_tol=ABS_TOL)
                assert self.error is None
            else:
                assert self.value == other.value
                assert self.error == other.error
            return True
        except AssertionError:
            return False


class EvalAlgorithm(str, Enum):
    """The evaluation types supported by Amazon Foundation Model Evaluations.

    The evaluation types are used to determine the evaluation metrics for the
    model.
    """

    PROMPT_STEREOTYPING = "prompt_stereotyping"
    FACTUAL_KNOWLEDGE = "factual_knowledge"
    TOXICITY = "toxicity"
    QA_TOXICITY = "qa_toxicity"
    SUMMARIZATION_TOXICITY = "summarization_toxicity"
    GENERAL_SEMANTIC_ROBUSTNESS = "general_semantic_robustness"
    ACCURACY = "accuracy"
    QA_ACCURACY = "qa_accuracy"
    QA_ACCURACY_SEMANTIC_ROBUSTNESS = "qa_accuracy_semantic_robustness"
    SUMMARIZATION_ACCURACY = "summarization_accuracy"
    SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS = "summarization_accuracy_semantic_robustness"
    CLASSIFICATION_ACCURACY = "classification_accuracy"
    CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS = "classification_accuracy_semantic_robustness"

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
        assert self.dataset_scores is not None or self.error is not None

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


class ModelTask(str, Enum):
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
        EvalAlgorithm.CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS,
    ],
    ModelTask.QUESTION_ANSWERING: [
        EvalAlgorithm.QA_TOXICITY,
        EvalAlgorithm.QA_ACCURACY,
        EvalAlgorithm.QA_ACCURACY_SEMANTIC_ROBUSTNESS,
    ],
    ModelTask.SUMMARIZATION: [
        EvalAlgorithm.SUMMARIZATION_TOXICITY,
        EvalAlgorithm.SUMMARIZATION_ACCURACY,
        EvalAlgorithm.SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS,
    ],
}

# Constants for Built-in dataset names
TREX = "trex"
BOOLQ = "boolq"
TRIVIA_QA = "trivia_qa"
NATURAL_QUESTIONS = "natural_questions"
CROWS_PAIRS = "crows-pairs"
GIGAWORD = "gigaword"
GOV_REPORT = "gov_report"
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
    EvalAlgorithm.PROMPT_STEREOTYPING.value: [CROWS_PAIRS],
    EvalAlgorithm.SUMMARIZATION_ACCURACY.value: [GIGAWORD, GOV_REPORT],
    EvalAlgorithm.GENERAL_SEMANTIC_ROBUSTNESS.value: [BOLD, TREX, WIKITEXT2],
    EvalAlgorithm.CLASSIFICATION_ACCURACY.value: [WOMENS_CLOTHING_ECOMMERCE_REVIEWS],
    EvalAlgorithm.CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS.value: [
        WOMENS_CLOTHING_ECOMMERCE_REVIEWS,
    ],
    EvalAlgorithm.SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS.value: [GIGAWORD, GOV_REPORT],
    EvalAlgorithm.TOXICITY.value: [BOLD, REAL_TOXICITY_PROMPTS, REAL_TOXICITY_PROMPTS_CHALLENGING],
    EvalAlgorithm.QA_TOXICITY.value: [BOOLQ, TRIVIA_QA, NATURAL_QUESTIONS],
    EvalAlgorithm.SUMMARIZATION_TOXICITY.value: [GIGAWORD, GOV_REPORT],
}

# Mapping of Default Prompt Template corresponding to eval, built-in dataset pair
DEFAULT_PROMPT_TEMPLATE = "$model_input"

BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES = {
    BOOLQ: 'Respond to the following question. Valid answers are "True" or "False". $model_input',
    TRIVIA_QA: "Respond to the following question with a short answer: $model_input",
    NATURAL_QUESTIONS: "Respond to the following question with a short answer: $model_input",
    GIGAWORD: "Summarize the following text in one sentence: $model_input",
    GOV_REPORT: "Summarize the following text in a few sentences: $model_input",
    WOMENS_CLOTHING_ECOMMERCE_REVIEWS: (
        "Classify the sentiment of the following review with 0 (negative sentiment)"
        " or 1 (positive sentiment): $model_input"
    ),
}


def get_default_prompt_template(dataset_name: str) -> str:
    """
    Util method to provide dataset specific default prompt templates. If not default is configured for the dataset,
        the method returns a generic default prompt template.
    :param dataset_name: Name of dataset
    """
    return BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES.get(dataset_name, DEFAULT_PROMPT_TEMPLATE)


# Mapping of Built-in dataset names and their DataConfigs
DATASET_CONFIGS: Dict[str, DataConfig] = {
    TREX: DataConfig(
        dataset_name=TREX,
        dataset_uri="s3://fmeval/datasets/trex/trex.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="question",
        target_output_location="answers",
        category_location="knowledge_category",
    ),
    BOOLQ: DataConfig(
        dataset_name=BOOLQ,
        dataset_uri="s3://fmeval/datasets/boolq/boolq.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="question",
        target_output_location="answer",
    ),
    TRIVIA_QA: DataConfig(
        dataset_name=TRIVIA_QA,
        dataset_uri="s3://fmeval/datasets/triviaQA/triviaQA.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="question",
        target_output_location="answer",
    ),
    NATURAL_QUESTIONS: DataConfig(
        dataset_name=NATURAL_QUESTIONS,
        dataset_uri="s3://fmeval/datasets/natural_questions/natural_questions.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="question",
        target_output_location="answer",
    ),
    CROWS_PAIRS: DataConfig(
        dataset_name=CROWS_PAIRS,
        dataset_uri="s3://fmeval/datasets/crows-pairs/crows-pairs.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        sent_more_input_location="sent_more",
        sent_less_input_location="sent_less",
        category_location="bias_type",
    ),
    WOMENS_CLOTHING_ECOMMERCE_REVIEWS: DataConfig(
        dataset_name=WOMENS_CLOTHING_ECOMMERCE_REVIEWS,
        dataset_uri="s3://fmeval/datasets/womens_clothing_reviews/womens_clothing_reviews.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location='"Review Text"',
        target_output_location='"Recommended IND"',
        category_location='"Class Name"',
    ),
    BOLD: DataConfig(
        dataset_name=BOLD,
        dataset_uri="s3://fmeval/datasets/bold/bold.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="prompt",
        category_location="domain",
    ),
    WIKITEXT2: DataConfig(
        dataset_name=WIKITEXT2,
        dataset_uri="s3://fmeval/datasets/wikitext2/wikitext2.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="prompt",
    ),
    REAL_TOXICITY_PROMPTS: DataConfig(
        dataset_name=REAL_TOXICITY_PROMPTS,
        dataset_uri="s3://fmeval/datasets/real_toxicity/real_toxicity.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="prompt",
    ),
    REAL_TOXICITY_PROMPTS_CHALLENGING: DataConfig(
        dataset_name=REAL_TOXICITY_PROMPTS_CHALLENGING,
        dataset_uri="s3://fmeval/datasets/real_toxicity/real_toxicity_challenging.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="prompt",
    ),
    GIGAWORD: DataConfig(
        dataset_name=GIGAWORD,
        dataset_uri="s3://fmeval/datasets/gigaword/gigaword.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="document",
        target_output_location="summary",
    ),
    GOV_REPORT: DataConfig(
        dataset_name=GOV_REPORT,
        dataset_uri="s3://fmeval/datasets/gov_report/gov_report.jsonl",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        model_input_location="report",
        target_output_location="summary",
    ),
}
