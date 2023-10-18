from unittest.mock import patch, PropertyMock

import numpy as np
import pytest

from amazon_fmeval.eval_algorithms.helper_models.helper_model import (
    ToxigenHelperModel,
    BertscoreHelperModel,
    TOXIGEN_SCORE_NAME,
    DETOXIFY_SCORE_NAMES,
    DetoxifyHelperModel,
    DETOXIFY_SCORE_TOXICITY,
    DETOXIFY_SCORE_SEVERE_TOXICITY,
    DETOXIFY_SCORE_OBSCENE,
    DETOXIFY_SCORE_IDENTITY_ATTACK,
    DETOXIFY_SCORE_INSULT,
    DETOXIFY_SCORE_THREAT,
    DETOXIFY_SCORE_SEXUAL_EXPLICIT,
)


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
        assert TOXIGEN_SCORE_NAME in actual_response
        assert actual_response[TOXIGEN_SCORE_NAME] == pytest.approx([0.5005707144737244, 0.5005643963813782], rel=1e-5)

    @patch.object(ToxigenHelperModel, "TOXIGEN_MODEL_NAME", new_callable=PropertyMock)
    def test_toxigen_helper_model_call(self, mock_model_name):
        """
        Test helper model for Toxigen
        Using lightweight test model: https://huggingface.co/hf-internal-testing/tiny-random-roberta
        """
        mock_model_name.return_value = "hf-internal-testing/tiny-random-roberta"
        test_helper = ToxigenHelperModel("prompt")
        actual_response = test_helper({"prompt": np.array(["My shitty text", "My good text"])})
        expected_response = {
            "prompt": np.array(["My shitty text", "My good text"]),
            TOXIGEN_SCORE_NAME: np.array([0.5005707, 0.5005644]),
        }
        assert actual_response.keys() == expected_response.keys()
        np.testing.assert_array_equal(actual_response["prompt"], expected_response["prompt"])
        np.testing.assert_almost_equal(actual_response[TOXIGEN_SCORE_NAME], expected_response["toxicity"])

    @patch.object(ToxigenHelperModel, "TOXIGEN_MODEL_NAME", new_callable=PropertyMock)
    def test_toxigen_helper_model_get_score_names(self, mock_model_name):
        """
        Test helper model for Toxigen
        Using lightweight test model: https://huggingface.co/hf-internal-testing/tiny-random-roberta
        """
        mock_model_name.return_value = "hf-internal-testing/tiny-random-roberta"
        test_helper = ToxigenHelperModel("prompt")
        assert test_helper.get_score_names() == [TOXIGEN_SCORE_NAME]

    def test_detoxify_helper_model_get_helper_scores(self):
        """
        Test helper model for Detoxify
        """
        test_helper = DetoxifyHelperModel()
        actual_response = test_helper.get_helper_scores(["My shitty text", "My good text"])
        expected_response = {
            "toxicity": [0.9817695021629333, 0.00045518550905399024],
            "severe_toxicity": [0.04576661065220833, 1.6480657905049156e-06],
            "obscene": [0.9683985114097595, 3.1544899684377015e-05],
            "identity_attack": [0.007208856288343668, 6.863904854981229e-05],
            "insult": [0.28945454955101013, 8.761371282162145e-05],
            "threat": [0.0014990373747423291, 2.826379204634577e-05],
            "sexual_explicit": [0.05178866535425186, 1.9261064153397456e-05],
        }
        assert list(actual_response.keys()) == DETOXIFY_SCORE_NAMES
        assert actual_response == pytest.approx(expected_response, rel=1e-5)

    def test_detoxify_helper_model_call(self):
        """
        Test helper model for Detoxify
        """
        test_helper = DetoxifyHelperModel()
        actual_response = test_helper({"model_output": np.array(["My shitty text", "My good text"])})
        expected_response = {
            "model_output": np.array(["My shitty text", "My good text"]),
            DETOXIFY_SCORE_TOXICITY: np.array([0.9817695021629333, 0.00045518550905399024]),
            DETOXIFY_SCORE_SEVERE_TOXICITY: np.array([0.04576661065220833, 1.6480657905049156e-06]),
            DETOXIFY_SCORE_OBSCENE: np.array([0.9683985114097595, 3.1544899684377015e-05]),
            DETOXIFY_SCORE_IDENTITY_ATTACK: np.array([0.007208856288343668, 6.863904854981229e-05]),
            DETOXIFY_SCORE_INSULT: np.array([0.28945454955101013, 8.761371282162145e-05]),
            DETOXIFY_SCORE_THREAT: np.array([0.0014990373747423291, 2.826379204634577e-05]),
            DETOXIFY_SCORE_SEXUAL_EXPLICIT: np.array([0.05178866535425186, 1.9261064153397456e-05]),
        }
        assert actual_response.keys() == expected_response.keys()
        np.testing.assert_array_equal(actual_response["model_output"], expected_response["model_output"])
        np.testing.assert_almost_equal(
            actual_response[DETOXIFY_SCORE_TOXICITY], expected_response[DETOXIFY_SCORE_TOXICITY]
        )
        np.testing.assert_almost_equal(
            actual_response[DETOXIFY_SCORE_SEVERE_TOXICITY], expected_response[DETOXIFY_SCORE_SEVERE_TOXICITY]
        )
        np.testing.assert_almost_equal(
            actual_response[DETOXIFY_SCORE_OBSCENE], expected_response[DETOXIFY_SCORE_OBSCENE]
        )
        np.testing.assert_almost_equal(
            actual_response[DETOXIFY_SCORE_IDENTITY_ATTACK], expected_response[DETOXIFY_SCORE_IDENTITY_ATTACK]
        )
        np.testing.assert_almost_equal(actual_response[DETOXIFY_SCORE_INSULT], expected_response[DETOXIFY_SCORE_INSULT])
        np.testing.assert_almost_equal(actual_response[DETOXIFY_SCORE_THREAT], expected_response[DETOXIFY_SCORE_THREAT])
        np.testing.assert_almost_equal(
            actual_response[DETOXIFY_SCORE_SEXUAL_EXPLICIT], expected_response[DETOXIFY_SCORE_SEXUAL_EXPLICIT]
        )

    def test_detoxify_helper_model_get_score_names(self):
        """
        Test helper model for Detoxify
        """
        test_helper = DetoxifyHelperModel()
        assert test_helper.get_score_names() == DETOXIFY_SCORE_NAMES

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
