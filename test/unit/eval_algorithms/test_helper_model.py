from unittest.mock import patch, PropertyMock

import numpy as np
import pytest

from amazon_fmeval.eval_algorithms.helper_models.helper_model import ToxigenHelperModel, BertscoreHelperModel


class TestHelperModel:
    @patch.object(ToxigenHelperModel, "TOXIGEN_MODEL_NAME", new_callable=PropertyMock)
    def test_toxigen_helper_model_get_helper_scores(self, mock_model_name):
        """
        Test helper model for Toxigen
        Using lightweight test model: https://huggingface.co/hf-internal-testing/tiny-random-roberta
        """
        mock_model_name.return_value = "hf-internal-testing/tiny-random-roberta"
        test_helper = ToxigenHelperModel()
        actual_response = test_helper.get_helper_scores(["My shitty text", "My good text"])
        assert "toxicity" in actual_response
        np.testing.assert_almost_equal(actual_response["toxicity"], np.array([0.5005707, 0.5005644]))

    @patch.object(ToxigenHelperModel, "TOXIGEN_MODEL_NAME", new_callable=PropertyMock)
    def test_toxigen_helper_model_call(self, mock_model_name):
        """
        Test helper model for Toxigen
        Using lightweight test model: https://huggingface.co/hf-internal-testing/tiny-random-roberta
        """
        mock_model_name.return_value = "hf-internal-testing/tiny-random-roberta"
        test_helper = ToxigenHelperModel()
        actual_response = test_helper({"prompt": np.array(["My shitty text", "My good text"])}, "prompt")
        expected_response = {
            "prompt": np.array(["My shitty text", "My good text"]),
            "toxicity": np.array([0.5005707, 0.5005644])
        }
        assert actual_response.keys() == expected_response.keys()
        np.testing.assert_array_equal(actual_response["prompt"], expected_response["prompt"])
        np.testing.assert_almost_equal(actual_response["toxicity"], expected_response["toxicity"])

    def test_bertscore_helper_model_roberta(self):
        """
        Test bertscore helper model
        """
        test_bertscore_1 = BertscoreHelperModel("distilbert-base-uncased")
        test_bertscore_2 = BertscoreHelperModel("distilbert-base-uncased")
        assert pytest.approx(test_bertscore_1) == pytest.approx(test_bertscore_2)
        assert test_bertscore_1.get_helper_scores("sample text reference", "sample text prediction") == pytest.approx(
            0.902793288230896
        )
