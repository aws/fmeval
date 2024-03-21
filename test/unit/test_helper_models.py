import pytest
from typing import NamedTuple
from unittest.mock import patch, PropertyMock

from fmeval.helper_models import (
    ToxigenModel,
    DetoxifyModel,
    BertscoreModel,
    BertscoreModelTypes,
)


class TestHelperModel:
    @patch.object(ToxigenModel, "MODEL_NAME", new_callable=PropertyMock)
    def test_toxigen_invoke_model(self, mock_model_name):
        """
        GIVEN valid inputs.
        WHEN invoke_model method of ToxigenModel is called.
        THEN the correct output is returned.

        Note: using lightweight test model: https://huggingface.co/hf-internal-testing/tiny-random-roberta
        """
        mock_model_name.return_value = "hf-internal-testing/tiny-random-roberta"
        toxigen_model = ToxigenModel()
        actual_response = toxigen_model.invoke_model(["My toxic text", "My good text"])
        assert ToxigenModel.SCORE_NAME in actual_response
        assert actual_response[ToxigenModel.SCORE_NAME] == pytest.approx(
            [0.5005707144737244, 0.5005643963813782], rel=1e-5
        )

    @patch.object(ToxigenModel, "MODEL_NAME", new_callable=PropertyMock)
    def test_toxigen_reduce(self, mock_model_name):
        """
        GIVEN a ToxigenModel instance.
        WHEN __reduce__ is called.
        THEN the correct output is returned.
        """
        mock_model_name.return_value = "hf-internal-testing/tiny-random-roberta"
        toxigen_model = ToxigenModel()
        assert toxigen_model.__reduce__() == (ToxigenModel, ())

    def test_detoxify_model_invoke_model(self):
        """
        GIVEN valid inputs.
        WHEN invoke_model method of DetoxifyModel is called.
        THEN the correct output is returned.
        """
        detoxify_model = DetoxifyModel()
        actual_response = detoxify_model.invoke_model(["My toxic text", "My good text"])
        expected_response = {
            DetoxifyModel.TOXICITY_SCORE: [0.06483059376478195, 0.00045518550905399024],
            DetoxifyModel.SEVERE_TOXICITY_SCORE: [1.26147870105342e-05, 1.6480657905049156e-06],
            DetoxifyModel.OBSCENE_SCORE: [0.0009980567265301943, 3.1544899684377015e-05],
            DetoxifyModel.IDENTITY_ATTACK_SCORE: [0.0012085289927199483, 6.863904854981229e-05],
            DetoxifyModel.INSULT_SCORE: [0.00813359022140503, 8.761371282162145e-05],
            DetoxifyModel.THREAT_SCORE: [0.0004742506134789437, 2.826379204634577e-05],
            DetoxifyModel.SEXUAL_EXPLICIT_SCORE: [0.00044487009290605783, 1.9261064153397456e-05],
        }
        assert list(actual_response.keys()) == DetoxifyModel.SCORE_NAMES
        assert actual_response[DetoxifyModel.TOXICITY_SCORE] == pytest.approx(
            expected_response[DetoxifyModel.TOXICITY_SCORE], rel=1e-5
        )
        assert actual_response[DetoxifyModel.SEVERE_TOXICITY_SCORE] == pytest.approx(
            expected_response[DetoxifyModel.SEVERE_TOXICITY_SCORE], rel=1e-5
        )
        assert actual_response[DetoxifyModel.OBSCENE_SCORE] == pytest.approx(
            expected_response[DetoxifyModel.OBSCENE_SCORE], rel=1e-5
        )
        assert actual_response[DetoxifyModel.IDENTITY_ATTACK_SCORE] == pytest.approx(
            expected_response[DetoxifyModel.IDENTITY_ATTACK_SCORE], rel=1e-5
        )
        assert actual_response[DetoxifyModel.INSULT_SCORE] == pytest.approx(
            expected_response[DetoxifyModel.INSULT_SCORE], rel=1e-5
        )
        assert actual_response[DetoxifyModel.THREAT_SCORE] == pytest.approx(
            expected_response[DetoxifyModel.THREAT_SCORE], rel=1e-5
        )
        assert actual_response[DetoxifyModel.SEXUAL_EXPLICIT_SCORE] == pytest.approx(
            expected_response[DetoxifyModel.SEXUAL_EXPLICIT_SCORE], rel=1e-5
        )

    def test_detoxify_reduce(self):
        """
        GIVEN a DetoxifyModel instance.
        WHEN __reduce__ is called.
        THEN the correct output is returned.
        """
        detoxify_model = DetoxifyModel()
        assert detoxify_model.__reduce__() == (DetoxifyModel, ())

    def test_bertscore_model_invoke_model(self):
        """
        GIVEN valid inputs.
        WHEN invoke_model method of BertscoreModel is called.
        THEN the correct output is returned.
        """
        bertscore = BertscoreModel(BertscoreModelTypes.ROBERTA_MODEL.value)
        result = bertscore.invoke_model("sample text reference", "sample text prediction")
        assert result == pytest.approx(0.8580247163772583)

    def test_bertscore_reduce(self):
        """
        GIVEN a BertscoreModel instance.
        WHEN __reduce__ is called.
        THEN the correct output is returned.
        """
        bertscore_model = BertscoreModel(BertscoreModelTypes.ROBERTA_MODEL.value)
        assert bertscore_model.__reduce__() == (BertscoreModel, (BertscoreModelTypes.ROBERTA_MODEL.value,))

    class TestCaseBertscoreModelTypes(NamedTuple):
        model_type: str
        allowed: bool

    @pytest.mark.parametrize(
        "model_type, allowed",
        [
            TestCaseBertscoreModelTypes(
                model_type=BertscoreModelTypes.MICROSOFT_DEBERTA_MODEL.value,
                allowed=True,
            ),
            TestCaseBertscoreModelTypes(
                model_type=BertscoreModelTypes.ROBERTA_MODEL.value,
                allowed=True,
            ),
            TestCaseBertscoreModelTypes(
                model_type="distilbert-base-uncased",
                allowed=False,
            ),
        ],
    )
    def test_bertscore_model_types_model_is_allowed(self, model_type, allowed):
        """
        GIVEN a model type string.
        WHEN BertscoreModelTypes.model_is_allowed is called.
        THEN the correct output is returned.
        """
        assert BertscoreModelTypes.model_is_allowed(model_type) == allowed

    def test_bertscore_model_types_model_list(self):
        """
        GIVEN N/A.
        WHEN BertscoreModelTypes.model_list is called.
        THEN the correct output is returned.

        Note: I'm just including this to meet the 100% unit test coverage requirement.
        Once I implement the code that actually calls these methods, we can remove this
        trivial test.
        """
        assert BertscoreModelTypes.model_list() == ["microsoft/deberta-xlarge-mnli", "roberta-large-mnli"]
