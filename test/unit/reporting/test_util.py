import pytest
from amazon_fmeval.reporting.util import format_string, format_dataset_name


@pytest.mark.parametrize(
    "original_string, kwargs, expected_string",
    [
        ("some general text", {}, "some general text"),
        ("text_with_underscores", {}, "text with underscores"),
        ("bERTScore plot", {"as_title": True}, "BERTScore Plot"),
        ("toxicity", {"as_score": True}, "toxicity score"),
        ("prompt_stereotyping", {"as_plot_title": True}, "is_biased score"),
        ("summarization_accuracy", {"as_eval_name": True}, "accuracy"),
        ("sent_more", {"as_column_name": True}, "s more"),
    ],
)
def test_format_string(original_string, kwargs, expected_string):
    """
    GIVEN valid parameters to format_string
    WHEN format_string is called
    THEN the correctly formatted string is returned
    """
    actual_string = format_string(text=original_string, **kwargs)
    assert actual_string == expected_string


@pytest.mark.parametrize(
    "dataset_name, hyperlink, html, expected_dataset_name",
    [
        ("custom dataset 1", True, True, "custom dataset 1"),
        ("crows-pairs", False, False, "CrowS-Pairs"),
        ("trex", True, True, '<a style="color:#006DAA;" href="https://hadyelsahar.github.io/t-rex/">T-REx</a>'),
        ("boolq", True, False, "[BoolQ](https://github.com/google-research-datasets/boolean-questions)"),
        ("trivia_qa", False, True, "TriviaQA"),
    ],
)
def test_format_dataset_name(dataset_name, hyperlink, html, expected_dataset_name):
    """
    GIVEN a built-in or custom dataset name
    WHEN format_dataset_name is called
    THEN the formatted dataset name is returned
    """
    actual_dataset_name = format_dataset_name(dataset_name=dataset_name, hyperlink=hyperlink, html=html)
    assert actual_dataset_name == expected_dataset_name
