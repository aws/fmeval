from typing import List, Tuple

from fmeval.eval_algorithms import EvalAlgorithm
from fmeval.reporting.constants import (
    DATASET_DETAILS,
    GENERAL_STRING_REPLACEMENTS,
    SCORE_STRING_REPLACEMENTS,
    EVAL_NAME_STRING_REPLACEMENTS,
    COLUMN_NAME_STRING_REPLACEMENTS,
    PLOT_TITLE_STRING_REPLACEMENTS,
    AVOID_REMOVE_UNDERSCORE,
)


def format_string(
    text: str,
    remove_underscore: bool = True,
    as_title: bool = False,
    as_score: bool = False,
    as_plot_title: bool = False,
    as_eval_name: bool = False,
    as_column_name: bool = False,
) -> str:
    """
    :param text: text, name of the score or eval.
    :param remove_underscore: Boolean indicating if underscores should be replaced with spaces.
    :param as_title: Boolean indicating if the text is a title, if set to True will capitalize each word.
    :param as_score: Boolean indicating if "score" should be appended to the text.
    :param as_plot_title: Boolean indicating if this is a plot title.
    :param as_eval_name: Boolean indicating if this is the name of an evaluation.
    :param as_column_name: Boolean indicating if this is the name of a table column.
    :return: formatted score name.
    """
    formatted_text = _replace_strings(text, GENERAL_STRING_REPLACEMENTS)
    if as_plot_title and EvalAlgorithm.PROMPT_STEREOTYPING.value in formatted_text:
        formatted_text = _replace_strings(formatted_text, PLOT_TITLE_STRING_REPLACEMENTS)
        remove_underscore = False
    if as_column_name:
        formatted_text = _replace_strings(formatted_text, COLUMN_NAME_STRING_REPLACEMENTS)
    if as_eval_name:
        formatted_text = _replace_strings(formatted_text, EVAL_NAME_STRING_REPLACEMENTS)
    if remove_underscore:
        if text not in AVOID_REMOVE_UNDERSCORE:  # pragma: no branch
            formatted_text = formatted_text.replace("_", " ")
    if as_score:
        formatted_text = _replace_strings(formatted_text, SCORE_STRING_REPLACEMENTS)
        formatted_text = formatted_text if "score" in formatted_text.lower() else f"{formatted_text} score"
    if as_title:
        # Capitalize each word while preserving original capitalization within words
        formatted_text = " ".join(w if w.isupper() else w[0].upper() + w[1:] for w in formatted_text.split())
    return formatted_text


def _replace_strings(text: str, replacements: List[Tuple[str, str]]) -> str:
    """
    :param text: The text which contains substrings that may be replaced.
    :param replacements: The tuples with format (original substring, replacement substring).
    :return: The text with the strings replaced if they exist.
    """
    for (old, new) in replacements:
        text = text.replace(old, new)
    return text


def format_dataset_name(dataset_name: str, hyperlink: bool = False, html: bool = True, color: str = "#006DAA") -> str:
    """
    :param dataset_name: The name of the dataset.
    :param hyperlink: Boolean indicating if hyperlink should be added to dataset name.
    :param html: Boolean indicating if hyperlink should be added in HTML format.
    :param color: The color of the text.
    :return: Properly capitalized dataset name.
    """
    if dataset_name not in DATASET_DETAILS:
        return dataset_name
    proper_dataset_name = DATASET_DETAILS[dataset_name].name
    if hyperlink:
        dataset_link = DATASET_DETAILS[dataset_name].url
        proper_dataset_name = add_hyperlink(proper_dataset_name, dataset_link, html, color)
    return proper_dataset_name


def add_hyperlink(text: str, link: str, html: bool = True, color: str = "#006DAA") -> str:
    """
    :param text: The text to add the hyperlink to.
    :param link: The URL to link to the text.
    :param html: Boolean indicating if hyperlink should be added in HTML format.
    :param color: The color of the text.
    """
    return f'<a style="color:{color};" href="{link}">{text}</a>' if html else f"[{text}]({link})"
