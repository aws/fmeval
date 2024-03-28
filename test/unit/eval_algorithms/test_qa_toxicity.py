from typing import List, NamedTuple
from unittest.mock import patch, MagicMock

import pytest
import ray
from _pytest.fixtures import fixture

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON,
)
from fmeval.eval_algorithms import (
    EvalScore,
    EvalOutput,
    CategoryScore,
    NATURAL_QUESTIONS,
    BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES,
    TRIVIA_QA,
    BOOLQ,
    DEFAULT_PROMPT_TEMPLATE,
)
from fmeval.eval_algorithms.helper_models.helper_model import (
    TOXIGEN_SCORE_NAME,
    DETOXIFY_SCORE_TOXICITY,
    DETOXIFY_SCORE_SEVERE_TOXICITY,
    DETOXIFY_SCORE_OBSCENE,
    DETOXIFY_SCORE_IDENTITY_ATTACK,
    DETOXIFY_SCORE_INSULT,
    DETOXIFY_SCORE_THREAT,
    DETOXIFY_SCORE_SEXUAL_EXPLICIT,
    DETOXIFY_SCORE_NAMES,
)
from fmeval.eval_algorithms.qa_toxicity import QAToxicity
from fmeval.eval_algorithms.toxicity import (
    ToxicityConfig,
    TOXIGEN_MODEL,
    DETOXIFY_MODEL,
)

DATASET_TOXIGEN = ray.data.from_items(
    [
        {
            DatasetColumns.MODEL_INPUT.value.name: "sample model input",
            DatasetColumns.PROMPT.value.name: "sample model input",
            DatasetColumns.CATEGORY.value.name: "dummy_category_1",
            DatasetColumns.MODEL_OUTPUT.value.name: "sample_model_output",
            TOXIGEN_SCORE_NAME: 1.0,
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "sample model input",
            DatasetColumns.PROMPT.value.name: "sample model input",
            DatasetColumns.CATEGORY.value.name: "dummy_category_2",
            DatasetColumns.MODEL_OUTPUT.value.name: "sample_model_output",
            TOXIGEN_SCORE_NAME: 1.0,
        },
    ]
)

DATASET_DETOXIFY = ray.data.from_items(
    [
        {
            DatasetColumns.MODEL_INPUT.value.name: "sample model input",
            DatasetColumns.CATEGORY.value.name: "dummy_category_1",
            DatasetColumns.MODEL_OUTPUT.value.name: "sample_model_output",
            DETOXIFY_SCORE_TOXICITY: 0.5,
            DETOXIFY_SCORE_SEVERE_TOXICITY: 0.5,
            DETOXIFY_SCORE_OBSCENE: 0.5,
            DETOXIFY_SCORE_IDENTITY_ATTACK: 0.5,
            DETOXIFY_SCORE_INSULT: 0.5,
            DETOXIFY_SCORE_THREAT: 0.5,
            DETOXIFY_SCORE_SEXUAL_EXPLICIT: 0.5,
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "sample model input",
            DatasetColumns.CATEGORY.value.name: "dummy_category_1",
            DatasetColumns.MODEL_OUTPUT.value.name: "sample_model_output",
            DETOXIFY_SCORE_TOXICITY: 0.5,
            DETOXIFY_SCORE_SEVERE_TOXICITY: 0.5,
            DETOXIFY_SCORE_OBSCENE: 0.5,
            DETOXIFY_SCORE_IDENTITY_ATTACK: 0.5,
            DETOXIFY_SCORE_INSULT: 0.5,
            DETOXIFY_SCORE_THREAT: 0.5,
            DETOXIFY_SCORE_SEXUAL_EXPLICIT: 0.5,
        },
    ]
)

EVAL_RESULTS_PATH = "/tmp/eval_results/"


class TestQAToxicity:
    @fixture(scope="module")
    def config(self) -> ToxicityConfig:
        return ToxicityConfig(model_type=TOXIGEN_MODEL)

    @staticmethod
    def get_toxigen_mock():
        instance = MagicMock()
        instance.get_helper_scores.return_value = {TOXIGEN_SCORE_NAME: [0.5]}
        instance.get_score_names.return_value = [TOXIGEN_SCORE_NAME]
        toxigen_mock = MagicMock()
        toxigen_mock.return_value = instance
        return toxigen_mock

    @staticmethod
    def get_detoxify_mock():
        instance = MagicMock()
        instance.get_helper_scores.return_value = {
            DETOXIFY_SCORE_TOXICITY: [0.1],
            DETOXIFY_SCORE_SEVERE_TOXICITY: [0.1],
            DETOXIFY_SCORE_OBSCENE: [0.1],
            DETOXIFY_SCORE_IDENTITY_ATTACK: [0.1],
            DETOXIFY_SCORE_INSULT: [0.1],
            DETOXIFY_SCORE_THREAT: [0.1],
            DETOXIFY_SCORE_SEXUAL_EXPLICIT: [0.1],
        }
        instance.get_score_names.return_value = DETOXIFY_SCORE_NAMES
        detoxify_mock = MagicMock()
        detoxify_mock.return_value = instance
        return detoxify_mock

    class TestCaseToxicityEvaluateSample(NamedTuple):
        model_output: str
        expected_response: List[EvalScore]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseToxicityEvaluateSample(
                model_output="Dummy toxic model output",
                expected_response=[EvalScore(name=TOXIGEN_SCORE_NAME, value=0.5)],
            )
        ],
    )
    @patch.dict("fmeval.eval_algorithms.qa_toxicity.TOXICITY_HELPER_MODEL_MAPPING", {TOXIGEN_MODEL: get_toxigen_mock()})
    def test_toxicity_evaluate_sample_toxigen(self, test_case, config):
        """
        GIVEN valid inputs
        WHEN QAToxicity.evaluate_sample with toxigen model_type is called
        THEN correct List of EvalScores is returned
        """
        eval_algorithm = QAToxicity(config)
        assert eval_algorithm.evaluate_sample(test_case.model_output) == test_case.expected_response

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseToxicityEvaluateSample(
                model_output="Dummy toxic model output",
                expected_response=[
                    EvalScore(name=DETOXIFY_SCORE_TOXICITY, value=0.1),
                    EvalScore(name=DETOXIFY_SCORE_SEVERE_TOXICITY, value=0.1),
                    EvalScore(name=DETOXIFY_SCORE_OBSCENE, value=0.1),
                    EvalScore(name=DETOXIFY_SCORE_IDENTITY_ATTACK, value=0.1),
                    EvalScore(name=DETOXIFY_SCORE_INSULT, value=0.1),
                    EvalScore(name=DETOXIFY_SCORE_THREAT, value=0.1),
                    EvalScore(name=DETOXIFY_SCORE_SEXUAL_EXPLICIT, value=0.1),
                ],
            )
        ],
    )
    @patch.dict(
        "fmeval.eval_algorithms.qa_toxicity.TOXICITY_HELPER_MODEL_MAPPING", {DETOXIFY_MODEL: get_detoxify_mock()}
    )
    def test_toxicity_evaluate_sample_detoxify(self, test_case):
        """
        GIVEN valid inputs
        WHEN QAToxicity.evaluate_sample with detoxify model_type is called
        THEN correct List of EvalScores is returned
        """
        config = ToxicityConfig()
        eval_algorithm = QAToxicity(config)
        assert eval_algorithm.evaluate_sample(test_case.model_output) == test_case.expected_response
