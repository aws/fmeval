import os

import pytest

from amazon_fmeval.eval_algorithms.helper_models.helper_model import ToxigenHelperModel, BertscoreHelperModel
from amazon_fmeval.util import project_root


class TestHelperModel:
    def test_toxigen_helper_model(self):
        """
        Test helper model for Toxigen
        Using test model: https://huggingface.co/hf-internal-testing/tiny-random-roberta downloaded at
        local path: "" for this.
        """
        test_helper = ToxigenHelperModel(
            os.path.join(project_root(__file__), "test", "resources", "test_roberta_model")
        )
        test_text_input = "My simple text"
        expected_response = [{"label": "LABEL_1", "score": pytest.approx(0.5005736947059631)}]
        assert test_helper.get_helper_score(test_text_input) == expected_response

    def test_bertscore_helper_model_roberta(self):
        """
        Test bertscore helper model
        """
        test_bertscore_1 = BertscoreHelperModel("distilbert-base-uncased")
        test_bertscore_2 = BertscoreHelperModel("distilbert-base-uncased")
        assert pytest.approx(test_bertscore_1) == pytest.approx(test_bertscore_2)
        assert test_bertscore_1.get_helper_score("sample text reference", "sample text prediction") == pytest.approx(
            0.902793288230896
        )
