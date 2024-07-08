import logging
import string


# punctuation and articles for the normalize function
ENGLISH_ARTICLES = ["a", "an", "the"]
ENGLISH_PUNCTUATIONS = string.punctuation

logger = logging.getLogger(__name__)


# Moved function to metrics.py because it's being used by both factual knowledge and qa accuracy
def normalize_text_quac_protocol(text: str) -> str:
    """
    Inspired by HELM: https://github.com/stanford-crfm/helm/blob/62f817eb695a31e8389e3f7be30609d3f0871837/src/helm/benchmark/metrics/basic_metrics.py
    Given a text, normalize it using the SQUAD / QUAC protocol. That is remove punctuations, excess spaces and articles, and return the lowercased tokens.
    SQUAD (https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/) and
    QuAC benchmarks (https://s3.amazonaws.com/my89public/quac/scorer.py) use this protocol to normalize text before evaluating it.
    HELM (https://github.com/stanford-crfm/helm/blob/62f817eb695a31e8389e3f7be30609d3f0871837/src/helm/benchmark/metrics/basic_metrics.py#L116)
    and HuggingFace evaluate (https://github.com/huggingface/evaluate/blob/775555d80af30d83dc6e9f42051840d29a34f31b/metrics/squad/compute_score.py#L11)
    also use this to normalization procedure.

    :param text: The text that needs to be normalized.
    :returns: The normalized text.
    """

    text = text.lower()
    text = "".join(character for character in text if character not in ENGLISH_PUNCTUATIONS)
    return " ".join([word for word in text.split(" ") if (word != "" and word not in ENGLISH_ARTICLES)])
