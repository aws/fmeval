from enum import Enum
from typing import NamedTuple, Tuple, List

from amazon_fmeval.eval_algorithms.factual_knowledge import FACTUAL_KNOWLEDGE
from amazon_fmeval.eval_algorithms.prompt_stereotyping import PROMPT_STEREOTYPING, LOG_PROBABILITY_DIFFERENCE
from amazon_fmeval.eval_algorithms.qa_accuracy import F1_SCORE, EXACT_MATCH_SCORE, QUASI_EXACT_MATCH_SCORE
from amazon_fmeval.eval_algorithms.summarization_accuracy import METEOR_SCORE, BERT_SCORE, ROUGE_SCORE
from amazon_fmeval.eval_algorithms.classification_accuracy import (
    CLASSIFICATION_ACCURACY_SCORE,
    BALANCED_ACCURACY_SCORE,
    PRECISION_SCORE,
    RECALL_SCORE,
)
from amazon_fmeval.eval_algorithms.classification_accuracy_semantic_robustness import (
    DELTA_CLASSIFICATION_ACCURACY_SCORE,
)
from amazon_fmeval.eval_algorithms.qa_accuracy_semantic_robustness import (
    DELTA_F1_SCORE,
    DELTA_EXACT_MATCH_SCORE,
    DELTA_QUASI_EXACT_MATCH_SCORE,
)
from amazon_fmeval.eval_algorithms.summarization_accuracy_semantic_robustness import (
    DELTA_ROUGE_SCORE,
    DELTA_BERT_SCORE,
    DELTA_METEOR_SCORE,
)
from amazon_fmeval.eval_algorithms.general_semantic_robustness import WER_SCORE
from amazon_fmeval.eval_algorithms import (
    TREX,
    BOOLQ,
    TRIVIA_QA,
    NATURAL_QUESTIONS,
    CROWS_PAIRS,
    CNN_DAILY_MAIL,
    XSUM,
    WOMENS_CLOTHING_ECOMMERCE_REVIEWS,
    BOLD,
    WIKITEXT2,
    REAL_TOXICITY_PROMPTS,
    REAL_TOXICITY_PROMPTS_CHALLENGING,
)
from amazon_fmeval.eval_algorithms import EvalAlgorithm
from amazon_fmeval.eval_algorithms.helper_models.helper_model import (
    TOXIGEN_SCORE_NAME,
    DETOXIFY_SCORE_TOXICITY,
    DETOXIFY_SCORE_SEVERE_TOXICITY,
    DETOXIFY_SCORE_OBSCENE,
    DETOXIFY_SCORE_IDENTITY_ATTACK,
    DETOXIFY_SCORE_INSULT,
    DETOXIFY_SCORE_THREAT,
    DETOXIFY_SCORE_SEXUAL_EXPLICIT,
)

# For general HTML alignment
CENTER = "center"
LEFT = "left"
RIGHT = "right"


class ListType(Enum):
    BULLETED = "bulleted"
    NUMBERED = "numbered"


# For general use in Markdown-related code
SINGLE_NEWLINE = "  \n"
DOUBLE_NEWLINE = "  \n\n"

# For tables and bar plots
NUM_SAMPLES_TO_DISPLAY_IN_TABLE = 5
CATEGORY_BAR_COLOR = "steelblue"
OVERALL_BAR_COLOR = "coral"
MAX_CHAR = 200

# Extensions used by the markdown library to convert markdown to HTML
MARKDOWN_EXTENSIONS = ["tables", "md_in_html"]

# Dataset score label used in category bar plot
DATASET_SCORE_LABEL = "Overall"

# Scores that are not per sample
AGGREGATE_ONLY_SCORES = [BALANCED_ACCURACY_SCORE, PRECISION_SCORE, RECALL_SCORE]

# For string formatting in eval names/score names
GENERAL_STRING_REPLACEMENTS: List[Tuple[str, str]] = [("qa", "Q&A"), ("f1", "F1")]
SCORE_STRING_REPLACEMENTS: List[Tuple[str, str]] = [
    ("prompt stereotyping", "is_biased"),
    ("meteor", "METEOR"),
    ("bertscore", "BERTScore"),
    ("rouge", "ROUGE"),
    ("F1 score", "F1 over words"),
]
EVAL_NAME_STRING_REPLACEMENTS: List[Tuple[str, str]] = [
    (EvalAlgorithm.QA_ACCURACY.value, EvalAlgorithm.ACCURACY.value),
    (EvalAlgorithm.SUMMARIZATION_ACCURACY.value, EvalAlgorithm.ACCURACY.value),
    (EvalAlgorithm.CLASSIFICATION_ACCURACY.value, EvalAlgorithm.ACCURACY.value),
    (EvalAlgorithm.GENERAL_SEMANTIC_ROBUSTNESS.value, "semantic_robustness"),
    (EvalAlgorithm.QA_ACCURACY_SEMANTIC_ROBUSTNESS.value, "semantic_robustness"),
    (EvalAlgorithm.CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS.value, "semantic_robustness"),
    (EvalAlgorithm.SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS.value, "semantic_robustness"),
    (EvalAlgorithm.QA_ACCURACY.value, EvalAlgorithm.TOXICITY.value),
    (EvalAlgorithm.SUMMARIZATION_TOXICITY.value, EvalAlgorithm.TOXICITY.value),
    (EvalAlgorithm.CLASSIFICATION_ACCURACY.value, EvalAlgorithm.TOXICITY.value),
]
PLOT_TITLE_STRING_REPLACEMENTS: List[Tuple[str, str]] = [("prompt_stereotyping", "is_biased score")]
COLUMN_NAME_STRING_REPLACEMENTS: List[Tuple[str, str]] = [
    ("sent_more", "stereotypical"),
    ("sent_less", "anti-stereotypical"),
    ("prob_", "probability_"),
    ("prompt_stereotyping", "Log Probability Difference"),
    ("word_error_rate", "Average WER"),
    ("classification_accuracy", "accuracy"),
    ("f1_score", "f1 over words"),
    ("meteor", "METEOR"),
    ("bertscore", "BERTScore"),
    ("rouge", "ROUGE"),
]

# Dataset types
BUILT_IN_DATASET = "Built-in Dataset"
CUSTOM_DATASET = "Custom Dataset"

PROMPT_COLUMN_NAME = "prompt"

TOXICITY_EVAL_NAMES = [
    EvalAlgorithm.TOXICITY.value,
    EvalAlgorithm.QA_TOXICITY.value,
    EvalAlgorithm.SUMMARIZATION_TOXICITY.value,
]

# Toxicity detector names
TOXIGEN_NAME = "Toxigen-roberta"
DETOXIFY_NAME = "UnitaryAI Detoxify-unbiased"
# Example table descriptions
TABLE_DESCRIPTION = "Below are a few examples of the highest and lowest-scoring examples across all categories. Some text may be truncated due to length constraints. To view the full prompts, please go to the S3 job output location that you specified when configuring the job. "
WER_TABLE_DESCRIPTION = "Below are a few examples of the highest and lowest-scoring examples across all categories. The lower the word error rate, the better the model performs. Some text may be truncated due to length constraints. To view the full prompts, please go to the S3 job output location that you specified when configuring the job."
STEREOTYPING_TABLE_DESCRIPTION = "For each sentence pair, we report the log probability difference, a value ranging -&#8734; to &#8734;, indicating how much the model stereotypes. Below are a few example of the most and least stereotypical prompts. Some text may be truncated due to length constraints. To view the full prompts, please go to the S3 job output location that you specified when configuring the job."
FACTUAL_KNOWLEDGE_TABLE_DESCRIPTION = "Below are a few examples of correct and incorrect model responses. Some text may be truncated due to length constraints. To view the full prompts, please go to the S3 job output location that you specified when configuring the job. "

# Score descriptions for the report
SCORE_DESCRIPTIONS = {
    # Factual knowledge
    FACTUAL_KNOWLEDGE: "A factual knowledge score is a value between 0 and 1, indicating the percentage of correctly retrieved real-world facts across different knowledge categories.",
    # Prompt stereotyping
    PROMPT_STEREOTYPING: "The is_biased prompt stereotyping score is a value between 0 and 1. Non-stereotyping models score around 0.5. Both 0 and 1 indicate stereotyping model behaviour.",
    LOG_PROBABILITY_DIFFERENCE: "For each sentence pair, we report the log probability difference, a value ranging -&#8734; to &#8734;, indicating how much the model stereotypes. ",
    # QA accuracy
    F1_SCORE: "Numerical score between 0 (worst) and 1 (best). F1-score is the harmonic mean of precision and recall. It is computed as follows:  precision = true positives / (true positives + false positives) and recall = true positives / (true positives + false negatives). Then F1 = 2 (precision * recall)/(precision + recall) .",
    EXACT_MATCH_SCORE: "An exact match score is a binary score where 1 indicates the model output and answer match exactly and 0 indicates otherwise.",
    QUASI_EXACT_MATCH_SCORE: "Similar as above, but both model output and answer are normalised first by removing any articles and punctuation. E.g., 1 also for predicted answers “Antarctica.” or “the Antarctica” .",
    # Summarization accuracy
    ROUGE_SCORE: "A ROUGE-N score computes the N-gram (sequences of n words) word overlaps between the reference and model summary, with the value ranging between 0 (no match) to 1 (perfect match).",
    METEOR_SCORE: "Meteor is similar to ROUGE-N, but it also accounts for rephrasing by using traditional NLP techniques such as stemming (e.g. matching “singing” to “sing”,“sings” etc.) and synonym lists.",
    BERT_SCORE: "BERTScore uses a second ML model (from the BERT family) to compute sentence embeddings and compare their similarity.",
    # Classification accuracy
    CLASSIFICATION_ACCURACY_SCORE: "The classification accuracy is `predicted_label == true_label`, reported as the mean accuracy over all datapoints.",
    PRECISION_SCORE: "The precision score is computed as `true positives / (true positives + false positives)`. ",
    RECALL_SCORE: "The recall score is computed as `true positives / (true positives + false negatives)`",
    BALANCED_ACCURACY_SCORE: "The balanced accuracy score is the same as accuracy in the binary case, otherwise averaged recall per class.",
    # General semantic robustness
    WER_SCORE: "Word error rate (WER) is a value between 0 and 1, and measures the difference between the model output on the unperturbed input and the output(s) on one or more perturbed versions of the same input. For more details on how word error rate is computed, see the [HuggingFace Article on Word Error Rate](https://huggingface.co/spaces/evaluate-metric/wer).",
    # Summarization semantic robustness
    DELTA_ROUGE_SCORE: "The performance change of the ROUGE-N score is measured.",
    DELTA_METEOR_SCORE: "The performance change of the METEOR score is measured.",
    DELTA_BERT_SCORE: "The performance change of the BERTscore is measured.",
    # QA semantic robustness
    DELTA_EXACT_MATCH_SCORE: "The performance change of the Exact Match score is measured.",
    DELTA_QUASI_EXACT_MATCH_SCORE: "The performance change of the Quasi Exact Match score is measured.",
    DELTA_F1_SCORE: "The performance change of the F1 over Words score is measured.",
    # Classification semantic robustness
    DELTA_CLASSIFICATION_ACCURACY_SCORE: "The score is the binary indicator on whether or not the model answer is correct.",
    # Toxicity
    TOXIGEN_SCORE_NAME: "A binary score where 0 indicates no toxicity detected and 1 indicates toxicity detected.",
    DETOXIFY_SCORE_TOXICITY: "A binary score from 0 (no toxicity detected) to 1 (toxicity detected) for the class: **toxicity**",
    DETOXIFY_SCORE_SEVERE_TOXICITY: "A binary score from 0 (no toxicity detected) to 1 (toxicity detected) for the class: **severe_toxicity**",
    DETOXIFY_SCORE_OBSCENE: "A binary score from 0 (no toxicity detected) to 1 (toxicity detected) for the class: **obscene**",
    DETOXIFY_SCORE_IDENTITY_ATTACK: "A binary score from 0 (no toxicity detected) to 1 (toxicity detected) for the class **identity_attack**",
    DETOXIFY_SCORE_INSULT: "A binary score from 0 (no toxicity detected) to 1 (toxicity detected) for the class: **insult**",
    DETOXIFY_SCORE_THREAT: "A binary score from 0 (no toxicity detected) to 1 (toxicity detected) for the class: **threat**",
    DETOXIFY_SCORE_SEXUAL_EXPLICIT: "A binary score from 0 (no toxicity detected) to 1 (toxicity detected) for the class: **sexual_explicit**",
}


class DatasetDetails(NamedTuple):
    name: str
    url: str
    description: str
    size: int


# Dataset details with the formatted names, URLs, descriptions and size
DATASET_DETAILS = {
    TREX: DatasetDetails(
        name="T-REx",
        url="https://hadyelsahar.github.io/t-rex/",
        description="A dataset which consists of knowledge triplets extracted from Wikipedia. The triplets take the form (subject, predicate, object), for instance, (Berlin, capital of, Germany) or (Tata Motors, subsidiary of, Tata Group). ",
        size=32260,
    ),
    BOOLQ: DatasetDetails(
        name="BoolQ",
        url="https://github.com/google-research-datasets/boolean-questions",
        description="A dataset consisting of question-passage-answer triplets. The question can be answered with yes/no, and the answer is contained in the passage. The questions are provided anonymously and unsolicited by users of the Google search engine, and afterwards paired with a paragraph from a Wikipedia article containing the answer.",
        size=12697,
    ),
    TRIVIA_QA: DatasetDetails(
        name="TriviaQA",
        url="http://nlp.cs.washington.edu/triviaqa/",
        description="A dataset consisting of 95K question-answer pairs with with on average six supporting evidence documents per question, leading to ~650K question-passage-answer triplets. The questions are authored by trivia enthusiasts and the evidence documents are independently gathered. ",
        size=156328,
    ),
    NATURAL_QUESTIONS: DatasetDetails(
        name="Natural Questions",
        url="https://github.com/google-research-datasets/natural-questions",
        description="A dataset consisting of ~320K question-passage-answer triplets. The questions are factual naturally-occurring questions. The passages are extracts from wikipedia articles (referred to as “long answers” in the original dataset). As before, providing the passage is optional depending on whether the open-book or closed-book case should be evaluated.",
        size=4289,
    ),
    CROWS_PAIRS: DatasetDetails(
        name="CrowS-Pairs",
        url="https://github.com/nyu-mll/crows-pairs",
        description="This dataset provides crowdsourced sentence pairs for the different categories along which stereotyping is to be measured.",
        size=1508,
    ),
    CNN_DAILY_MAIL: DatasetDetails(
        name="CNN/DailyMail",
        url="https://huggingface.co/datasets/cnn_dailymail",
        description="A dataset consisting of newspaper articles and their reference summaries. The reference summaries consist of highlights from the original article and are usually 2-4 sentences long.",
        size=287113,
    ),
    XSUM: DatasetDetails(
        name="XSUM",
        url="https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset",
        description="A dataset consisting of newspaper articles from the BBC and their reference summaries. The reference summaries consist of a single sentence: the boldfaced sentence at the begininning of each BBC article, provided by article’s authors.",
        size=204045,
    ),
    WOMENS_CLOTHING_ECOMMERCE_REVIEWS: DatasetDetails(
        name="Women's E-commerce Clothing Reviews",
        url="https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews",
        description="This dataset consists of clothing reviews, both as a text and numerical scores.",
        size=23486,
    ),
    BOLD: DatasetDetails(
        name="BOLD",
        url="https://github.com/amazon-science/bold",
        description="A large-scale dataset that consists of English prompts aimed at testing bias and toxicity generation across five domains: profession, gender, race, religion, and political ideology.",
        size=23679,
    ),
    WIKITEXT2: DatasetDetails(
        name="WikiText2",
        url="https://huggingface.co/datasets/wikitext",
        description="A dataset which consists of Good and Featured articles from Wikipedia. To create prompts, we broke each article down into sentences and extracted first 6 tokens from each sentence as the prompt.",
        size=86007,
    ),
    REAL_TOXICITY_PROMPTS: DatasetDetails(
        name="Real Toxicity Prompts",
        url="https://github.com/allenai/real-toxicity-prompts",
        description="A dataset of truncated sentence snippets from the web. ",
        size=98243,
    ),
    REAL_TOXICITY_PROMPTS_CHALLENGING: DatasetDetails(
        name="Real Toxicity Prompts Challenging",
        url="https://github.com/allenai/real-toxicity-prompts",
        description="A dataset of truncated sentence snippets from the web. Prompts marked as “challenging” have been found by the authors to consistently lead to generation of toxic continuation by tested models (i.e., GPT-1, GPT-2, GPT-3, CTRL, CTRL-WIKI).",
        size=1199,
    ),
}

TREX_DESCRIPTION_EXAMPLES = "We convert these predicates to prompts, e.g., Berlin is the capital of ___ (expected answer: Germany) and Tata Motors is a subsidiary of ___ (expected answer: Tata Group)."
