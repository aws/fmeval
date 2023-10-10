import os

import pytest

from eval_algorithms.helper_models.helper_model import ToxigenHelperModel
from util import project_root


class TestHelperModel:
    def test_toxigen_helper_model(self):
        """
        Test invoke_detector method for ToxigenEvaluationDetectorInvoker
        Using test model: https://huggingface.co/hf-internal-testing/tiny-random-roberta downloaded at
        local path: "" for this.
        """
        test_invoker = ToxigenHelperModel(
            os.path.join(project_root(__file__), "test", "resources", "test_roberta_model")
        )
        test_text_input = "My simple text"
        expected_response = [{"label": "LABEL_1", "score": pytest.approx(0.5005736947059631)}]
        assert test_invoker.invoke(test_text_input) == expected_response
