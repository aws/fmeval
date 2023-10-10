import re

import pytest

from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.extractors.json_extractor import JsonExtractor


class TestJsonExtractor:
    valid_single_model_response = {"predictions": {"output": "Model response valid", "prob": 0.8}}
    valid_multi_model_response = {
        "predictions": [
            {"output": "Model response valid", "prob": 0.8},
            {"output": "Model response valid", "prob": 0.9},
        ]
    }

    def test_json_extractor_valid_single_record(self):
        json_extractor = JsonExtractor(
            output_jmespath_expression="predictions.output",
            log_probability_jmespath_expression="predictions.prob",
        )
        assert json_extractor.extract_output(self.valid_single_model_response, 1) == "Model response valid"
        assert json_extractor.extract_log_probability(self.valid_single_model_response, 1) == 0.8

    def test_json_extractor_valid_single_record_invalid_jmespath(self):
        json_extractor = JsonExtractor(
            output_jmespath_expression="predictions.invalid",
            log_probability_jmespath_expression="predictions.prob",
        )
        with pytest.raises(EvalAlgorithmClientError, match="JMESpath predictions.invalid could not find any data"):
            json_extractor.extract_output(self.valid_single_model_response, 1)

    def test_json_extractor_invalid_output_jmespath_single_record(self):
        json_extractor = JsonExtractor(
            output_jmespath_expression="predictions.prob", log_probability_jmespath_expression="predictions.prob"
        )
        with pytest.raises(
            EvalAlgorithmClientError, match="Extractor found: 0.8 which does not match expected type <class 'str'>"
        ):
            json_extractor.extract_output(self.valid_single_model_response, 1)

    def test_json_extractor_invalid_probability_jmespath_single_record(self):
        json_extractor = JsonExtractor(
            output_jmespath_expression="predictions.output",
            log_probability_jmespath_expression="predictions.output",
        )
        with pytest.raises(
            EvalAlgorithmClientError,
            match="Extractor found: Model response valid which does not match expected type <class 'float'>",
        ):
            json_extractor.extract_log_probability(self.valid_single_model_response, 1)

    def test_json_extractor_valid_multi_record(self):
        json_extractor = JsonExtractor(
            output_jmespath_expression="predictions[*].output",
            log_probability_jmespath_expression="predictions[*].prob",
        )
        assert json_extractor.extract_output(self.valid_multi_model_response, 2) == ["Model response valid"] * 2
        assert json_extractor.extract_log_probability(self.valid_multi_model_response, 2) == [0.8, 0.9]

    def test_json_extractor_invalid_output_jmespath_multi_record(self):
        json_extractor = JsonExtractor(
            output_jmespath_expression="predictions[*].prob",
            log_probability_jmespath_expression="predictions[*].prob",
        )
        with pytest.raises(
            EvalAlgorithmClientError,
            match=re.escape("Extractor found: [0.8, 0.9] which does not match expected list of <class 'str'>"),
        ):
            json_extractor.extract_output(self.valid_multi_model_response, 2)

    def test_json_extractor_invalid_probability_jmespath_multi_record(self):
        json_extractor = JsonExtractor(
            output_jmespath_expression="predictions[*].output",
            log_probability_jmespath_expression="predictions[*].output",
        )
        with pytest.raises(
            EvalAlgorithmClientError,
            match=re.escape(
                "Extractor found: ['Model response valid', 'Model response valid'] "
                "which does not match expected list of <class 'float'>"
            ),
        ):
            json_extractor.extract_log_probability(self.valid_multi_model_response, 2)
