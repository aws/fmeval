import pytest

from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.extractors.json_extractor import JsonExtractor


class TestJsonExtractor:
    valid_model_responses = [
        {"predictions": {"output": "Model response valid", "prob": 0.8}},
        {"predictions": {"output": "Model response valid", "prob": [0.8]}},
        {"predictions": {"output": "Model response valid", "prob": [0.8, 0.1]}},
    ]

    @pytest.mark.parametrize(
        "valid_model_response, expected_output, expected_log_prob",
        [
            (valid_model_responses[0], "Model response valid", 0.8),
            (valid_model_responses[1], "Model response valid", 0.8),
            (valid_model_responses[2], "Model response valid", 0.9),
        ],
    )
    def test_json_extractor_valid_single_record(self, valid_model_response, expected_output, expected_log_prob):
        json_extractor = JsonExtractor(
            output_jmespath_expression="predictions.output",
            log_probability_jmespath_expression="predictions.prob",
        )
        assert json_extractor.extract_output(valid_model_response, 1) == expected_output
        assert json_extractor.extract_log_probability(valid_model_response, 1) == pytest.approx(expected_log_prob)

    def test_json_extractor_valid_single_record_invalid_jmespath(self):
        json_extractor = JsonExtractor(
            output_jmespath_expression="predictions.invalid",
            log_probability_jmespath_expression="predictions.prob",
        )
        with pytest.raises(EvalAlgorithmClientError, match="JMESpath predictions.invalid could not find any data"):
            json_extractor.extract_output(self.valid_model_responses[0], 1)

    def test_json_extractor_invalid_output_jmespath_single_record(self):
        json_extractor = JsonExtractor(
            output_jmespath_expression="predictions.prob", log_probability_jmespath_expression="predictions.prob"
        )
        with pytest.raises(
            EvalAlgorithmClientError, match="Extractor found: 0.8 which does not match expected type <class 'str'>"
        ):
            json_extractor.extract_output(self.valid_model_responses[0], 1)

    def test_json_extractor_invalid_probability_jmespath_single_record(self):
        json_extractor = JsonExtractor(
            output_jmespath_expression="predictions.output",
            log_probability_jmespath_expression="predictions.output",
        )
        with pytest.raises(
            EvalAlgorithmClientError,
            match="Extractor found: Model response valid which does not match expected list of <class 'float'>",
        ):
            json_extractor.extract_log_probability(self.valid_model_responses[0], 1)
