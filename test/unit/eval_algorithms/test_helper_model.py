from unittest.mock import patch, PropertyMock
import pytest

from fmeval.eval_algorithms.helper_models.helper_model import (
    ToxigenHelperModel,
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
    BertscoreHelperModel,
)


class TestHelperModel:
    @patch.object(ToxigenHelperModel, "TOXIGEN_MODEL_NAME", new_callable=PropertyMock)
    def test_toxigen_helper_model_get_helper_scores(self, mock_model_name):
        """
        GIVEN valid inputs
        WHEN get_helper_scores() method of ToxigenHelperModel is called
        THEN correct output is returned
        Using lightweight test model: https://huggingface.co/hf-internal-testing/tiny-random-roberta
        """
        mock_model_name.return_value = "hf-internal-testing/tiny-random-roberta"
        test_helper = ToxigenHelperModel()
        actual_response = test_helper.get_helper_scores(["My toxic text", "My good text"])
        assert TOXIGEN_SCORE_NAME in actual_response
        assert actual_response[TOXIGEN_SCORE_NAME] == pytest.approx([0.5005707144737244, 0.5005643963813782], rel=1e-5)

    @patch.object(ToxigenHelperModel, "TOXIGEN_MODEL_NAME", new_callable=PropertyMock)
    def test_toxigen_helper_model_get_score_names(self, mock_model_name):
        """
        GIVEN valid inputs
        WHEN get_score_names() method of ToxigenHelperModel is called
        THEN correct output is returned
        Using lightweight test model: https://huggingface.co/hf-internal-testing/tiny-random-roberta
        """
        mock_model_name.return_value = "hf-internal-testing/tiny-random-roberta"
        test_helper = ToxigenHelperModel()
        assert test_helper.get_score_names() == [TOXIGEN_SCORE_NAME]

    def test_toxigen_reduce(self):
        """
        GIVEN a ToxigenHelperModel instance.
        WHEN __reduce__ is called.
        THEN the correct output is returned.
        """
        toxigen_model = ToxigenHelperModel()
        assert toxigen_model.__reduce__() == (ToxigenHelperModel, ())

    def test_detoxify_helper_model_get_helper_scores(self):
        """
        GIVEN valid inputs
        WHEN get_helper_scores() method of DetoxifyHelperModel is called
        THEN correct output is returned
        """
        test_helper = DetoxifyHelperModel()
        actual_response = test_helper.get_helper_scores(["My toxic text", "My good text"])
        expected_response = {
            DETOXIFY_SCORE_TOXICITY: [0.06483059376478195, 0.00045518550905399024],
            DETOXIFY_SCORE_SEVERE_TOXICITY: [1.26147870105342e-05, 1.6480657905049156e-06],
            DETOXIFY_SCORE_OBSCENE: [0.0009980567265301943, 3.1544899684377015e-05],
            DETOXIFY_SCORE_IDENTITY_ATTACK: [0.0012085289927199483, 6.863904854981229e-05],
            DETOXIFY_SCORE_INSULT: [0.00813359022140503, 8.761371282162145e-05],
            DETOXIFY_SCORE_THREAT: [0.0004742506134789437, 2.826379204634577e-05],
            DETOXIFY_SCORE_SEXUAL_EXPLICIT: [0.00044487009290605783, 1.9261064153397456e-05],
        }
        assert list(actual_response.keys()) == DETOXIFY_SCORE_NAMES
        assert actual_response[DETOXIFY_SCORE_TOXICITY] == pytest.approx(
            expected_response[DETOXIFY_SCORE_TOXICITY], rel=1e-5
        )
        assert actual_response[DETOXIFY_SCORE_SEVERE_TOXICITY] == pytest.approx(
            expected_response[DETOXIFY_SCORE_SEVERE_TOXICITY], rel=1e-5
        )
        assert actual_response[DETOXIFY_SCORE_OBSCENE] == pytest.approx(
            expected_response[DETOXIFY_SCORE_OBSCENE], rel=1e-5
        )
        assert actual_response[DETOXIFY_SCORE_IDENTITY_ATTACK] == pytest.approx(
            expected_response[DETOXIFY_SCORE_IDENTITY_ATTACK], rel=1e-5
        )
        assert actual_response[DETOXIFY_SCORE_INSULT] == pytest.approx(
            expected_response[DETOXIFY_SCORE_INSULT], rel=1e-5
        )
        assert actual_response[DETOXIFY_SCORE_THREAT] == pytest.approx(
            expected_response[DETOXIFY_SCORE_THREAT], rel=1e-5
        )
        assert actual_response[DETOXIFY_SCORE_SEXUAL_EXPLICIT] == pytest.approx(
            expected_response[DETOXIFY_SCORE_SEXUAL_EXPLICIT], rel=1e-5
        )

    def test_detoxify_helper_model_get_score_names(self):
        """
        GIVEN valid inputs
        WHEN get_score_names() method of DetoxifyHelperModel is called
        THEN correct output is returned
        """
        test_helper = DetoxifyHelperModel()
        assert test_helper.get_score_names() == DETOXIFY_SCORE_NAMES

    def test_detoxify_reduce(self):
        """
        GIVEN a DetoxifyHelperModel instance.
        WHEN __reduce__ is called.
        THEN the correct output is returned.
        """
        detoxify_model = DetoxifyHelperModel()
        assert detoxify_model.__reduce__() == (DetoxifyHelperModel, ())

    def test_bertscore_helper_model_roberta(self):
        """
        Test bertscore helper model
        """
        bertscore = BertscoreHelperModel("distilbert-base-uncased")
        assert bertscore.get_helper_scores("sample text reference", "sample text prediction") == pytest.approx(
            0.902793288230896
        )

    def test_bertscore_reduce(self):
        """
        GIVEN a BertscoreHelperModel instance.
        WHEN __reduce__ is called.
        THEN the correct output is returned.
        """
        bertscore_model = BertscoreHelperModel("distilbert-base-uncased")
        assert bertscore_model.__reduce__() == (BertscoreHelperModel, ("distilbert-base-uncased",))
