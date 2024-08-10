import pytest
from unittest.mock import patch, call, Mock
from typing import NamedTuple, Optional
from ray.actor import ActorHandle

from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModel
from fmeval.transforms.summarization_accuracy_metrics import MeteorScore, RougeScore, BertScore, ROUGE_2


@pytest.mark.parametrize("load_modules", [True, False])
def test_meteor_score_init(load_modules):
    """
    GIVEN valid arguments to __init__.
    WHEN a MeteorScore is instantiated.
    THEN _load_modules is called if applicable.
    """
    with patch("fmeval.transforms.summarization_accuracy_metrics.MeteorScore._load_modules") as mock_load_modules:
        MeteorScore(
            target_output_keys=["target_output"],
            model_output_keys=["model_output"],
            output_keys=["meteor"],
            allow_duplicate_input_keys=False,
            load_modules=load_modules,
        )
        if load_modules:
            mock_load_modules.assert_called_once()
        else:
            mock_load_modules.assert_not_called()


def test_meteor_score_call():
    """
    GIVEN a MeteorScore instance.
    WHEN its __call__ method is invoked.
    THEN nltk.translate.meteor_score.single_meteor_score and
     nltk.word_tokenize are called with the correct arguments.

    Note: we don't validate the structure of the __call__ output since
    we already have @validate_call to handle that.
    """
    with patch(
        "fmeval.transforms.summarization_accuracy_metrics.meteor_score.single_meteor_score"
    ) as mock_meteor, patch("fmeval.transforms.summarization_accuracy_metrics.word_tokenize") as mock_word_tokenize:

        mock_word_tokenize.side_effect = ["tokenized_target_output", "tokenized_model_output"]
        ms = MeteorScore(
            target_output_keys=["target_output"],
            model_output_keys=["model_output"],
            output_keys=["meteor"],
            allow_duplicate_input_keys=False,
        )
        sample = {"target_output": "Hello there!", "model_output": "Hi"}
        ms(sample)
        mock_word_tokenize.assert_has_calls([call("Hello there!"), call("Hi")])
        mock_meteor.assert_called_once_with(reference="tokenized_target_output", hypothesis="tokenized_model_output")


def test_rouge_score_init():
    """
    GIVEN valid arguments to __init__.
    WHEN a RougeScore is instantiated.
    THEN hf_evaluate.load("rouge") is called.
    """
    with patch("fmeval.transforms.summarization_accuracy_metrics.hf_evaluate.load") as mock_load:
        RougeScore(
            target_output_keys=["target_output"],
            model_output_keys=["model_output"],
            output_keys=["rouge"],
            allow_duplicate_input_keys=False,
        )
        mock_load.assert_called_once_with("rouge")


def test_rouge_score_call():
    """
    GIVEN a RougeScore instance.
    WHEN its __call__ method is invoked.
    THEN self.rouge_metric.compute is called with the correct arguments.

    Note: we don't validate the structure of the __call__ output since
    we already have @validate_call to handle that.
    """
    with patch("fmeval.transforms.summarization_accuracy_metrics.hf_evaluate.load") as mock_hf_load:
        rouge_type = ROUGE_2
        mock_rouge_metric = Mock()
        mock_rouge_metric.compute = Mock()
        mock_rouge_metric.compute.return_value = {rouge_type: "blah"}
        mock_hf_load.return_value = mock_rouge_metric

        rs = RougeScore(
            target_output_keys=["target_output"],
            model_output_keys=["model_output"],
            output_keys=["rouge"],
            allow_duplicate_input_keys=False,
            rouge_type=rouge_type,
        )
        sample = {"target_output": "Hello there!", "model_output": "Hi"}
        rs(sample)
        mock_rouge_metric.compute.assert_called_once_with(
            predictions=["Hi"],
            references=["Hello there!"],
            use_stemmer=rs.use_stemmer,
            rouge_types=[rs.rouge_type],
        )


def test_bert_score_call_with_bertscore_model_object():
    """
    GIVEN a BertScore instance, where its `bertscore_model` is a BertscoreHelperModel object.
    WHEN its __call__ method is invoked.
    THEN self.bertscore_model is invoked with the correct arguments.

    Note: we don't validate the structure of the __call__ output since
    we already have @validate_call to handle that.
    """
    mock_bertscore_model = Mock(spec=BertscoreHelperModel)
    mock_bertscore_model.get_helper_scores = Mock()

    bs = BertScore(
        target_output_keys=["target_output"],
        model_output_keys=["model_output"],
        output_keys=["bertscore"],
        allow_duplicate_input_keys=False,
        bertscore_model=mock_bertscore_model,
    )
    sample = {"target_output": "Hello there!", "model_output": "Hi"}
    bs(sample)
    mock_bertscore_model.get_helper_scores.assert_called_once_with("Hello there!", "Hi")


def test_bert_score_call_with_target_output_keys_provider():
    """
    GIVEN a BertScore instance with a valid `target_output_keys provider`.
    WHEN its __call__ method is invoked.
    THEN self.bertscore_model is invoked with the correct arguments.

    Note: we don't validate the structure of the __call__ output since
    we already have @validate_call to handle that.
    """
    mock_bertscore_model = Mock(spec=BertscoreHelperModel)
    mock_bertscore_model.get_helper_scores = Mock()

    bs = BertScore(
        target_output_keys=None,
        model_output_keys=["model_output"],
        output_keys=["bertscore"],
        allow_duplicate_input_keys=False,
        target_output_keys_provider="target_output",
        bertscore_model=mock_bertscore_model,
    )
    sample = {"target_output": ["Hello there!"], "model_output": "Hi"}
    bs(sample)
    mock_bertscore_model.get_helper_scores.assert_called_once_with("Hello there!", "Hi")


def test_bertscore_multiple_targets_max_score():
    """
    GIVEN a BertScore instance with multiple possible target answers.
    WHEN its __call__ method is invoked.
    THEN the maximum score is returned.
    """
    mock_bertscore_model = Mock(spec=BertscoreHelperModel)
    mock_bertscore_model.get_helper_scores = Mock()

    output_scores = [0.2, 0.1, 0.3]
    mock_bertscore_model.get_helper_scores.side_effect = output_scores

    bs = BertScore(
        target_output_keys=None,
        model_output_keys=["model_output"],
        output_keys=["bertscore"],
        allow_duplicate_input_keys=False,
        target_output_keys_provider="target_output",
        bertscore_model=mock_bertscore_model,
    )
    sample = {"target_output": ["random output", "hello", "something"], "model_output": "hello"}
    output = bs(sample)

    mock_bertscore_model.get_helper_scores.assert_has_calls(
        [call("random output", "hello"), call("hello", "hello"), call("something", "hello")]
    )
    assert output["bertscore"] == pytest.approx(max(output_scores), rel=1e-5)


def test_bert_score_call_with_ray_actor_handle():
    """
    GIVEN a BertScore instance, where its `bertscore_model` is a Ray actor handle.
    WHEN its __call__ method is invoked.
    THEN the correct Ray APIs are called.

    Note: we don't validate the structure of the __call__ output since
    we already have @validate_call to handle that.
    """
    mock_bertscore_model = Mock(spec=ActorHandle)
    mock_bertscore_model.get_helper_scores = Mock()
    mock_bertscore_model.get_helper_scores.remote = Mock(return_value="remote invocation result")

    with patch("fmeval.transforms.summarization_accuracy_metrics.ray.get") as mock_ray_get:
        bs = BertScore(
            target_output_keys=["target_output"],
            model_output_keys=["model_output"],
            output_keys=["bertscore"],
            allow_duplicate_input_keys=False,
            bertscore_model=mock_bertscore_model,
        )
        sample = {"target_output": "Hello there!", "model_output": "Hi"}
        bs(sample)
        mock_bertscore_model.get_helper_scores.remote.assert_called_once_with("Hello there!", "Hi")
        mock_ray_get.assert_called_once_with("remote invocation result")


class TestCaseMetricNumericalValues(NamedTuple):
    model_output: str
    target_output: str
    expected_score: float
    rouge_type: Optional[str] = None


@pytest.mark.parametrize(
    "test_case",
    [
        TestCaseMetricNumericalValues(
            model_output="I like cake.",
            target_output="I like cake.",
            expected_score=0.9921875,
        ),
        TestCaseMetricNumericalValues(
            model_output="Berlin: Art, Heritage, Exhibitions Hub.",
            target_output="Berlin: an art metropolis.",
            expected_score=0.5009920634920636,
        ),
    ],
)
def test_meteor_numerical_values(test_case):
    ms = MeteorScore(
        target_output_keys=["target_output"],
        model_output_keys=["model_output"],
        output_keys=["meteor"],
        allow_duplicate_input_keys=False,
    )
    sample = {"target_output": test_case.target_output, "model_output": test_case.model_output}
    output = ms(sample)
    assert output["meteor"] == pytest.approx(test_case.expected_score, rel=1e-5)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCaseMetricNumericalValues(
            model_output="I like cake.", target_output="I like cake.", expected_score=1.0, rouge_type="rouge1"
        ),
        TestCaseMetricNumericalValues(
            model_output="Berlin: Art, Heritage, Exhibitions Hub.",
            target_output="Berlin: an art metropolis.",
            expected_score=0.4444444444444445,
            rouge_type="rouge1",
        ),
        TestCaseMetricNumericalValues(
            model_output="I like cake.", target_output="I like cake.", expected_score=1.0, rouge_type="rouge2"
        ),
        TestCaseMetricNumericalValues(
            model_output="Berlin: Art, Heritage, Exhibitions Hub.",
            target_output="Berlin: an art metropolis.",
            expected_score=0.0,
            rouge_type="rouge2",
        ),
        TestCaseMetricNumericalValues(
            model_output="I like cake.", target_output="I like cake.", expected_score=1.0, rouge_type="rougeL"
        ),
        TestCaseMetricNumericalValues(
            model_output="Berlin: Art, Heritage, Exhibitions Hub.",
            target_output="Berlin: an art metropolis.",
            expected_score=0.4444444444444445,
            rouge_type="rougeL",
        ),
    ],
)
def test_get_rouge_score(test_case):
    rs = RougeScore(
        target_output_keys=["target_output"],
        model_output_keys=["model_output"],
        output_keys=["rouge"],
        allow_duplicate_input_keys=False,
        rouge_type=test_case.rouge_type,
    )
    sample = {"target_output": test_case.target_output, "model_output": test_case.model_output}
    output = rs(sample)
    assert output["rouge"] == pytest.approx(test_case.expected_score, rel=1e-5)
