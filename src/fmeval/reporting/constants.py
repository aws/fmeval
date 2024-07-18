from enum import Enum
from typing import NamedTuple, Tuple, List

from fmeval.eval_algorithms.factual_knowledge import FACTUAL_KNOWLEDGE, FACTUAL_KNOWLEDGE_QUASI_EXACT
from fmeval.eval_algorithms.prompt_stereotyping import PROMPT_STEREOTYPING, LOG_PROBABILITY_DIFFERENCE
from fmeval.eval_algorithms.qa_accuracy import (
    F1_SCORE,
    EXACT_MATCH_SCORE,
    QUASI_EXACT_MATCH_SCORE,
    PRECISION_OVER_WORDS,
    RECALL_OVER_WORDS,
)
from fmeval.eval_algorithms.summarization_accuracy import METEOR_SCORE, BERT_SCORE, ROUGE_SCORE
from fmeval.eval_algorithms.classification_accuracy import (
    CLASSIFICATION_ACCURACY_SCORE,
    BALANCED_ACCURACY_SCORE,
    PRECISION_SCORE,
    RECALL_SCORE,
)
from fmeval.eval_algorithms.classification_accuracy_semantic_robustness import (
    DELTA_CLASSIFICATION_ACCURACY_SCORE,
)
from fmeval.eval_algorithms.qa_accuracy_semantic_robustness import (
    DELTA_F1_SCORE,
    DELTA_EXACT_MATCH_SCORE,
    DELTA_QUASI_EXACT_MATCH_SCORE,
    DELTA_PRECISION_OVER_WORDS,
    DELTA_RECALL_OVER_WORDS,
)
from fmeval.eval_algorithms.summarization_accuracy_semantic_robustness import (
    DELTA_ROUGE_SCORE,
    DELTA_BERT_SCORE,
    DELTA_METEOR_SCORE,
)
from fmeval.eval_algorithms.general_semantic_robustness import WER_SCORE, BERT_SCORE_DISSIMILARITY
from fmeval.eval_algorithms import (
    TREX,
    BOOLQ,
    TRIVIA_QA,
    NATURAL_QUESTIONS,
    CROWS_PAIRS,
    WOMENS_CLOTHING_ECOMMERCE_REVIEWS,
    BOLD,
    WIKITEXT2,
    REAL_TOXICITY_PROMPTS,
    REAL_TOXICITY_PROMPTS_CHALLENGING,
    GIGAWORD,
    GOV_REPORT,
)
from fmeval.eval_algorithms import EvalAlgorithm
from fmeval.eval_algorithms.helper_models.helper_model import (
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


class ListType(str, Enum):
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
GENERAL_STRING_REPLACEMENTS: List[Tuple[str, str]] = [("qa", "Q&A"), ("f1", "F1"), ("sagemaker", "SageMaker")]
SCORE_STRING_REPLACEMENTS: List[Tuple[str, str]] = [
    ("prompt stereotyping", "is_biased"),
    ("meteor", "METEOR"),
    ("bertscore", "BERTScore"),
    ("rouge", "ROUGE"),
    ("F1 score", "F1 over words"),
    ("obscene", "Obscenity"),
    ("sexual explicit", "Sexual Explicitness"),
]
EVAL_NAME_STRING_REPLACEMENTS: List[Tuple[str, str]] = [
    (EvalAlgorithm.QA_ACCURACY.value, EvalAlgorithm.ACCURACY.value),
    (EvalAlgorithm.SUMMARIZATION_ACCURACY.value, EvalAlgorithm.ACCURACY.value),
    (EvalAlgorithm.CLASSIFICATION_ACCURACY.value, EvalAlgorithm.ACCURACY.value),
    (EvalAlgorithm.GENERAL_SEMANTIC_ROBUSTNESS.value, "semantic_robustness"),
    ("accuracy_semantic_robustness", "semantic_robustness"),
    (EvalAlgorithm.QA_ACCURACY.value, EvalAlgorithm.TOXICITY.value),
    (EvalAlgorithm.SUMMARIZATION_TOXICITY.value, EvalAlgorithm.TOXICITY.value),
    (EvalAlgorithm.CLASSIFICATION_ACCURACY.value, EvalAlgorithm.TOXICITY.value),
]
PLOT_TITLE_STRING_REPLACEMENTS: List[Tuple[str, str]] = [("prompt_stereotyping", "is_biased score")]
COLUMN_NAME_STRING_REPLACEMENTS: List[Tuple[str, str]] = [
    ("sent_more", "s_more"),
    ("s_more_input", "<math>S<sub>more</sub></math>"),
    ("sent_less", "s_less"),
    ("s_less_input", "<math>S<sub>less</sub></math>"),
    ("prob_", "probability_"),
    ("word_error_rate", "Average WER"),
    ("classification_accuracy", "accuracy"),
    ("f1_score", "f1 over words"),
    ("meteor", "METEOR"),
    ("bertscore", "BERTScore"),
    ("rouge", "ROUGE"),
]
AVOID_REMOVE_UNDERSCORE = ["sent_more_input", "sent_less_input", "is_biased"]
ACCURACY_SEMANTIC_ROBUSTNESS_ALGOS = [
    EvalAlgorithm.SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS.value,
    EvalAlgorithm.QA_ACCURACY_SEMANTIC_ROBUSTNESS,
    EvalAlgorithm.CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS,
]
ACCURACY_SEMANTIC_ROBUSTNESS_SCORES = [
    CLASSIFICATION_ACCURACY_SCORE,
    METEOR_SCORE,
    BERT_SCORE,
    ROUGE_SCORE,
    F1_SCORE,
    EXACT_MATCH_SCORE,
    QUASI_EXACT_MATCH_SCORE,
]
# Dataset types
BUILT_IN_DATASET = "Built-in Dataset"
CUSTOM_DATASET = "Custom Dataset"

TOXICITY_EVAL_NAMES = [
    EvalAlgorithm.TOXICITY.value,
    EvalAlgorithm.QA_TOXICITY.value,
    EvalAlgorithm.SUMMARIZATION_TOXICITY.value,
]

# Prompt stereotyping table column name
PROBABILITY_RATIO = "<math><box>p(S<sub>more</sub>)/p(S<sub>less</sub>)</box></math>"
IS_BIASED = "is_biased"

# Toxicity detector names
TOXIGEN_NAME = "Toxigen-roberta"
DETOXIFY_NAME = "UnitaryAI Detoxify-unbiased"
TOXIGEN_URI = "https://github.com/microsoft/TOXIGEN"
DETOXIFY_URI = "https://github.com/unitaryai/detoxify"
# Example table descriptions
TABLE_DESCRIPTION = "Below are a few examples of the highest and lowest-scoring examples across all categories. Some text may be truncated due to length constraints. To view the full prompts, please go to the S3 job output location that you specified when configuring the job. "
WER_TABLE_DESCRIPTION = "Below are a few examples of the highest and lowest-scoring examples across all categories. The lower the word error rate, the better the model performs. Some text may be truncated due to length constraints. To view the full prompts, please go to the S3 job output location that you specified when configuring the job."
STEREOTYPING_TABLE_DESCRIPTION = "**Probability ratio:** \n\nFor each sentence pair, we additionally report by how much the model stereotypes. This is given by the ratio <math><box>p(S<sub>more</sub>)/p(S<sub>less</sub>)</box></math>. This ratio will be between 0 and ∞. A ratio of 200, for example, means that the model assigns 200x the probability to the more stereotypical sentence than the less stereotypical sentence."
FACTUAL_KNOWLEDGE_TABLE_DESCRIPTION = "Below are a few examples of correct and incorrect model responses. Some text may be truncated due to length constraints. To view the full prompts, please go to the S3 job output location that you specified when configuring the job. "

# Score descriptions for the report
SCORE_DESCRIPTIONS = {
    # Factual knowledge
    FACTUAL_KNOWLEDGE: "Exact inclusion score is a binary score where 1 indicates that the target output is contained in the model output, and 0 indicates otherwise. The average over the whole dataset is reported, such that the final score indicates the percentage of correctly retrieved real-world facts.",
    FACTUAL_KNOWLEDGE_QUASI_EXACT: "Quasi-exact inclusion is a binary score that represents whether a slightly modified (removing excess whitespace and punctuation) version of the target output is contained in the model output. If yes, the score is 1, otherwise 0. The average over the whole dataset is reported, such that the final score indicates the percentage of correctly retrieved real-world facts after removing excess whitespace and punctuation.",
    # Prompt stereotyping
    PROMPT_STEREOTYPING: "The is_biased prompt stereotyping score measures whether the model assigns higher probability to the more stereotypical sentence than the less stereotypical one, i.e., <math><box>p(S<sub>more</sub>) > p(S<sub>less</sub>)</box></math>. It is a value between 0 and 1, where 1 indicates that the model always prefers the more stereotypical sentence while 0 means that it never prefers the more stereotypical sentence. An unbiased model prefers more and less stereotypical sentences at equal rates, corresponding to a score of 0.5",
    LOG_PROBABILITY_DIFFERENCE: "For each sentence pair, we report the log probability difference, a value ranging -&#8734; to &#8734;, indicating how much the model stereotypes. ",
    # QA accuracy
    F1_SCORE: "Numerical score between 0 (worst) and 1 (best). F1-score is the harmonic mean of precision and recall. It is computed as follows:  precision = true positives / (true positives + false positives) and recall = true positives / (true positives + false negatives). Then F1 = 2 (precision * recall)/(precision + recall) .",
    EXACT_MATCH_SCORE: "An exact match score is a binary score where 1 indicates the model output and answer match exactly and 0 indicates otherwise.",
    QUASI_EXACT_MATCH_SCORE: "Similar as above, but both model output and answer are normalised first by removing any articles and punctuation. E.g., 1 also for predicted answers “Antarctica.” or “the Antarctica” .",
    PRECISION_OVER_WORDS: "The precision score is the fraction of words in the model output that are also found in the target output.",
    RECALL_OVER_WORDS: "The recall score is the fraction of words in the target output that are also found in the model output.`",
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
    BERT_SCORE_DISSIMILARITY: "BERTScore Dissimilarity is computed as 1 - BERTScore and measures semantic differences between the original and perturbed versions of the same input.",
    # Summarization semantic robustness
    DELTA_ROUGE_SCORE: "Delta ROUGE-N score measures the change in Rouge between the original and perturbed versions of the same input.",
    DELTA_METEOR_SCORE: "Delta Meteor score measures the change in Meteor between the original and perturbed versions of the same input.",
    DELTA_BERT_SCORE: "Delta BERTscore measures the change in BERTscore between the original and perturbed versions of the same input.",
    # QA semantic robustness
    DELTA_EXACT_MATCH_SCORE: "Delta Exact Match score measures the change in Exact Match score between the original and perturbed versions of the same input.",
    DELTA_QUASI_EXACT_MATCH_SCORE: "Delta Quasi Exact Match score measures the change in Quasi Exact Match score between the original and perturbed versions of the same input.",
    DELTA_F1_SCORE: "Delta F1 score measures the change in F1 score between the original and perturbed versions of the same input.",
    DELTA_PRECISION_OVER_WORDS: "Delta Precision measures the change in Precision between the original and perturbed versions of the same input.",
    DELTA_RECALL_OVER_WORDS: "Delta Recall measures the change in Recall between the original and perturbed versions of the same input.",
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
    GIGAWORD: DatasetDetails(
        name="Gigaword",
        url="https://huggingface.co/datasets/gigaword",
        description="A dataset with around 4 million news article headlines. Our dataset consists of 190k entries from its validation set.",
        size=189651,
    ),
    GOV_REPORT: DatasetDetails(
        name="Government Report",
        url="https://gov-report-data.github.io/",
        description="A dataset including a long-form summarization benchmark. It contains significantly longer documents (9.4k words) and summaries (553 words) than most existing datasets.",
        size=7238,
    ),
}

TREX_DESCRIPTION_EXAMPLES = "We convert these predicates to prompts, e.g., Berlin is the capital of ___ (expected answer: Germany) and Tata Motors is a subsidiary of ___ (expected answer: Tata Group)."

CROWS_PAIRS_DISCLAIMER = "**Disclaimer**: 1) The crowdsourced CrowS dataset is noisy. While it gives a good indication of overall model performance, individual pairs may be invalid. 2) CrowS measures U.S.-typical stereotypes. Specifically, the bias categories are taken from the US Equal Employment Opportunities Commission’s list of protected categories and the sentence pairs are produced by Amazon Mechanical Turk workers in the United States."
