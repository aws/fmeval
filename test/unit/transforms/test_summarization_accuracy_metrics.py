import pytest
from unittest.mock import patch, call, Mock
from typing import List, NamedTuple, Optional

from ray import ObjectRef

from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.helper_models import BertscoreModel
from fmeval.transforms.summarization_accuracy_metrics import MeteorScore, RougeScore, BertScore, ROUGE_2


def test_meteor_score_init_load_modules():
    """
    GIVEN valid arguments to __init__.
    WHEN a MeteorScore is instantiated with load_meteor_modules=True.
    THEN the instance is created without errors, and _load_meteor_modules is called.
    """
    with patch("fmeval.transforms.summarization_accuracy_metrics._load_meteor_modules") as mock_load_modules:
        MeteorScore(
            input_keys=["target_output", "model_output"],
            output_keys=["meteor"],
            target_output_key="target_output",
            model_output_key="model_output",
            load_meteor_modules=True,
        )
        mock_load_modules.assert_called_once()


class TestCaseInitFailure(NamedTuple):
    input_keys: List[str]
    output_keys: List[str]
    err_msg: str


@pytest.mark.parametrize(
    "input_keys, output_keys, err_msg",
    [
        TestCaseInitFailure(
            input_keys=["not_target_output", "model_output"],
            output_keys=["meteor"],
            err_msg="input_keys to MeteorScore should be",
        ),
        TestCaseInitFailure(
            input_keys=["target_output", "not_model_output"],
            output_keys=["meteor"],
            err_msg="input_keys to MeteorScore should be",
        ),
        TestCaseInitFailure(
            input_keys=["target_output", "model_output"],
            output_keys=["key1", "key2"],
            err_msg="MeteorScore should only have a single output key.",
        ),
    ],
)
def test_meteor_score_init_failure(input_keys, output_keys, err_msg):
    """
    GIVEN invalid initializer arguments.
    WHEN a MeteorScore object is instantiated.
    THEN an EvalAlgorithmClientError with the correct error message is raised.
    """
    with pytest.raises(EvalAlgorithmClientError, match=err_msg):
        MeteorScore(
            input_keys=input_keys,
            output_keys=output_keys,
            target_output_key="target_output",
            model_output_key="model_output",
        )


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
            input_keys=["target_output", "model_output"],
            output_keys=["meteor"],
            target_output_key="target_output",
            model_output_key="model_output",
            load_meteor_modules=False,
        )
        sample = {"target_output": "Hello there!", "model_output": "Hi"}
        ms(sample)
        mock_word_tokenize.assert_has_calls([call("Hello there!"), call("Hi")])
        mock_meteor.assert_called_once_with(reference="tokenized_target_output", hypothesis="tokenized_model_output")


@pytest.mark.parametrize(
    "input_keys, output_keys, err_msg",
    [
        TestCaseInitFailure(
            input_keys=["not_target_output", "model_output"],
            output_keys=["rouge"],
            err_msg="input_keys to RougeScore should be",
        ),
        TestCaseInitFailure(
            input_keys=["target_output", "not_model_output"],
            output_keys=["rouge"],
            err_msg="input_keys to RougeScore should be",
        ),
        TestCaseInitFailure(
            input_keys=["target_output", "model_output"],
            output_keys=["key1", "key2"],
            err_msg="RougeScore should only have a single output key.",
        ),
    ],
)
def test_rouge_score_init_failure(input_keys, output_keys, err_msg):
    """
    GIVEN invalid initializer arguments.
    WHEN a RougeScore object is instantiated.
    THEN an EvalAlgorithmClientError with the correct error message is raised.
    """
    with pytest.raises(EvalAlgorithmClientError, match=err_msg):
        RougeScore(
            input_keys=input_keys,
            output_keys=output_keys,
            target_output_key="target_output",
            model_output_key="model_output",
        )


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
            input_keys=["target_output", "model_output"],
            output_keys=["rouge"],
            target_output_key="target_output",
            model_output_key="model_output",
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


@pytest.mark.parametrize(
    "input_keys, output_keys, err_msg",
    [
        TestCaseInitFailure(
            input_keys=["not_target_output", "model_output"],
            output_keys=["rouge"],
            err_msg="input_keys to BertScore should be",
        ),
        TestCaseInitFailure(
            input_keys=["target_output", "not_model_output"],
            output_keys=["rouge"],
            err_msg="input_keys to BertScore should be",
        ),
        TestCaseInitFailure(
            input_keys=["target_output", "model_output"],
            output_keys=["key1", "key2"],
            err_msg="BertScore should only have a single output key.",
        ),
    ],
)
def test_bert_score_init_failure(input_keys, output_keys, err_msg):
    """
    GIVEN invalid initializer arguments.
    WHEN a BertScore object is instantiated.
    THEN an EvalAlgorithmClientError with the correct error message is raised.
    """
    with pytest.raises(EvalAlgorithmClientError, match=err_msg):
        BertScore(
            input_keys=input_keys,
            output_keys=output_keys,
            target_output_key="target_output",
            model_output_key="model_output",
            bertscore_model=Mock(),
        )


def test_bert_score_call_with_bertscore_model_object():
    """
    GIVEN a BertScore instance, where its `bertscore_model` is a BertscoreModel object.
    WHEN its __call__ method is invoked.
    THEN self.bertscore_model is invoked with the correct arguments.

    Note: we don't validate the structure of the __call__ output since
    we already have @validate_call to handle that.
    """
    mock_bertscore_model = Mock(spec=BertscoreModel)
    mock_bertscore_model.invoke_model = Mock()

    bs = BertScore(
        input_keys=["target_output", "model_output"],
        output_keys=["rouge"],
        target_output_key="target_output",
        model_output_key="model_output",
        bertscore_model=mock_bertscore_model,
    )
    sample = {"target_output": "Hello there!", "model_output": "Hi"}
    bs(sample)
    mock_bertscore_model.invoke_model.assert_called_once_with("Hello there!", "Hi")


def test_bert_score_call_with_ray_actor_handle():
    """
    GIVEN a BertScore instance, where its `bertscore_model` is a Ray actor handle.
    WHEN its __call__ method is invoked.
    THEN the correct Ray APIs are called.

    Note: we don't validate the structure of the __call__ output since
    we already have @validate_call to handle that.
    """
    mock_bertscore_model = Mock(spec=ObjectRef)
    mock_bertscore_model.invoke_model = Mock()
    mock_bertscore_model.invoke_model.remote = Mock(return_value="remote invocation result")

    with patch("fmeval.transforms.summarization_accuracy_metrics.ray.get") as mock_ray_get:
        bs = BertScore(
            input_keys=["target_output", "model_output"],
            output_keys=["rouge"],
            target_output_key="target_output",
            model_output_key="model_output",
            bertscore_model=mock_bertscore_model,
        )
        sample = {"target_output": "Hello there!", "model_output": "Hi"}
        bs(sample)
        mock_bertscore_model.invoke_model.remote.assert_called_once_with("Hello there!", "Hi")
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
        input_keys=["target_output", "model_output"],
        output_keys=["meteor"],
        target_output_key="target_output",
        model_output_key="model_output",
        load_meteor_modules=False,
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
        input_keys=["target_output", "model_output"],
        output_keys=["rouge"],
        target_output_key="target_output",
        model_output_key="model_output",
        rouge_type=test_case.rouge_type,
    )
    sample = {"target_output": test_case.target_output, "model_output": test_case.model_output}
    output = rs(sample)
    assert output["rouge"] == pytest.approx(test_case.expected_score, rel=1e-5)
