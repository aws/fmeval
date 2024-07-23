from unittest.mock import Mock, patch
from ray.actor import ActorHandle

from fmeval.transforms.qa_accuracy_metrics import BertScoreWithDelimiter

DUMMY_LIST = [0.0]  # BertScoreWithDelimiter takes the max score, so get must return a list of floats (scores)


# Added this test for coverage, will add more if necessary
def test_bert_score_with_delimiter_call_with_ray_actor_handle():
    """
    GIVEN a BertScoreWithDelimiter instance, where its `bertscore_model` is a Ray actor handle.
    WHEN its __call__ method is invoked.
    THEN the correct Ray APIs are called.

    Note: we don't validate the structure of the __call__ output since
    we already have @validate_call to handle that.
    """
    mock_bertscore_model = Mock(spec=ActorHandle)
    mock_bertscore_model.get_helper_scores = Mock()
    mock_bertscore_model.get_helper_scores.remote = Mock(return_value="remote invocation result")

    with patch("fmeval.transforms.qa_accuracy_metrics.ray.get") as mock_ray_get:
        mock_ray_get.return_value = DUMMY_LIST
        bs = BertScoreWithDelimiter(
            target_output_keys=["target_output"],
            model_output_keys=["model_output"],
            output_keys=["bertscore"],
            allow_duplicate_input_keys=False,
            bertscore_model=mock_bertscore_model,
        )
        sample = {"target_output": "Hello there!", "model_output": "Hi"}
        bs(sample)
        mock_bertscore_model.get_helper_scores.remote.assert_called_once_with("Hello there!", "Hi")
        mock_ray_get.assert_called_once_with(["remote invocation result"])  # this must be a list because ray.get
        # takes in possible_scores which is a list (corresponding to possible targets)
