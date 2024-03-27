import pytest
from unittest.mock import Mock, patch
from fmeval.util import EvalAlgorithmClientError
from fmeval.transforms.semantic_robustness_metrics import BertScoreDissimilarity, WER, MeanDeltaScores


def test_bertscore_dissimilarity_call():
    """
    GIVEN a BertScoreDissimilarity instance.
    WHEN its __call__ method is called.
    THEN the correct output is returned.
    """
    bsd = BertScoreDissimilarity(bert_score_keys=["a", "b", "c", "d"], output_key="bsd")
    sample = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    actual_bsd_score = bsd(sample)["bsd"]
    assert actual_bsd_score == 0.75  # 1 - mean(0.1, 0.2, 0.3, 0.4)


def test_wer_init_failure():
    """
    GIVEN prediction_keys and reference_keys arguments with mismatching lengths.
    WHEN a WER is initialized.
    THEN an exception with the correct error message is raised.
    """
    err_msg = (
        "prediction_keys and reference_keys should have the same number of elements. "
        "prediction_keys has 2 elements while reference_keys has "
        "3 elements."
    )
    with pytest.raises(EvalAlgorithmClientError, match=err_msg):
        WER(
            prediction_keys=["p1", "p2"],
            reference_keys=["r1", "r2", "r3"],
            output_key="wer",
        )


def test_wer_call():
    """
    GIVEN a WER instance.
    WHEN its __call__ method is called.
    THEN the huggingface wer metric is called with the correct arguments
        and the record is augmented with the output from calling the
        huggingface wer metric.
    """
    with patch("fmeval.transforms.summarization_accuracy_metrics.hf_evaluate.load") as mock_hf_load:
        mock_wer_metric = Mock()
        mock_wer_metric.compute = Mock()
        mock_wer_metric.compute.return_value = 0.123
        mock_hf_load.return_value = mock_wer_metric

        wer = WER(
            prediction_keys=["p1", "p2", "p3"],
            reference_keys=["r1", "r2", "r3"],
            output_key="wer",
        )

        sample = {"p1": "a", "p2": "b", "p3": "c", "r1": "d", "r2": "e", "r3": "f"}
        result = wer(sample)["wer"]
        mock_wer_metric.compute.assert_called_once_with(
            predictions=["a", "b", "c"],
            references=["d", "e", "f"],
        )
        assert result == 0.123


def test_mean_delta_scores_init():
    """
    GIVEN valid arguments.
    WHEN a MeanDeltaScores is initialized.
    THEN its input_keys and output_keys attributes are set correctly.
    """
    mds = MeanDeltaScores(
        key_mapping={
            "original_a": (["perturbed_a_1", "perturbed_a_2"], "mean_delta_a"),
            "original_b": (["perturbed_b_1", "perturbed_b_2"], "mean_delta_b"),
        }
    )
    assert mds.input_keys == [
        "original_a",
        "original_b",
        "perturbed_a_1",
        "perturbed_a_2",
        "perturbed_b_1",
        "perturbed_b_2",
    ]
    assert mds.output_keys == ["mean_delta_a", "mean_delta_b"]


def test_mean_delta_scores_call():
    """
    GIVEN a MeanDeltaScores instance.
    WHEN its __call__ method is called.
    THEN the correct scores are computed.
    """
    mds = MeanDeltaScores(
        key_mapping={
            "original_a": (["perturbed_a_1", "perturbed_a_2"], "mean_delta_a"),
            "original_b": (["perturbed_b_1", "perturbed_b_2"], "mean_delta_b"),
        }
    )
    sample = {
        "original_a": 162,
        "original_b": 189,
        "perturbed_a_1": 100,
        "perturbed_a_2": 200,
        "perturbed_b_1": 300,
        "perturbed_b_2": 400,
    }
    result = mds(sample)
    assert result["mean_delta_a"] == 50
    assert result["mean_delta_b"] == 161
