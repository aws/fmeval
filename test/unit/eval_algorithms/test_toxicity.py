import re
from typing import NamedTuple, List, Optional
from unittest.mock import patch, MagicMock, Mock, PropertyMock

import numpy as np
import pytest
import ray
from _pytest.fixtures import fixture
from ray.data import Dataset

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON,
    MEAN,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms import EvalOutput, EvalScore
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
    DetoxifyHelperModel,
    ToxigenHelperModel,
)
from fmeval.eval_algorithms.toxicity import (
    ToxicityConfig,
    Toxicity,
    TOXIGEN_MODEL,
    DETOXIFY_MODEL,
    ToxicityScores,
)
from fmeval.exceptions import EvalAlgorithmClientError

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


class TestToxicity:
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
    @patch.dict("fmeval.eval_algorithms.toxicity.TOXICITY_HELPER_MODEL_MAPPING", {TOXIGEN_MODEL: get_toxigen_mock()})
    def test_toxicity_evaluate_sample_toxigen(self, test_case, config):
        """
        GIVEN valid inputs
        WHEN Toxicity.evaluate_sample with toxigen model_type is called
        THEN correct List of EvalScores is returned
        """
        eval_algorithm = Toxicity(config)
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
    @patch.dict("fmeval.eval_algorithms.toxicity.TOXICITY_HELPER_MODEL_MAPPING", {DETOXIFY_MODEL: get_detoxify_mock()})
    def test_toxicity_evaluate_sample_detoxify(self, test_case):
        """
        GIVEN valid inputs
        WHEN Toxicity.evaluate_sample with detoxify model_type is called
        THEN correct List of EvalScores is returned
        """
        config = ToxicityConfig()
        eval_algorithm = Toxicity(config)
        assert eval_algorithm.evaluate_sample(test_case.model_output) == test_case.expected_response

    def test_toxicity_invalid_config(self):
        """
        GIVEN invalid inputs
        WHEN ToxicityConfig is initialised
        THEN expected error is raised
        """
        expected_error_message = (
            "Invalid model_type: my_model requested in ToxicityConfig, please choose from "
            "acceptable values: ['toxigen', 'detoxify']"
        )
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(expected_error_message)):
            ToxicityConfig(model_type="my_model")

    class TestCaseToxicityEvaluate(NamedTuple):
        input_dataset: Dataset
        prompt_template: Optional[str]
        dataset_config: Optional[DataConfig]
        input_dataset_with_generated_model_output: Optional[Dataset]
        expected_response: List[EvalOutput]
        dataset_with_scores: Dataset

    class TestCaseEvaluate(NamedTuple):
        user_provided_prompt_template: Optional[str]
        dataset_prompt_template: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseEvaluate(
                user_provided_prompt_template="Do something with $model_input.",
                dataset_prompt_template="Do something with $model_input.",
            ),
            TestCaseEvaluate(
                user_provided_prompt_template=None,
                dataset_prompt_template=None,
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.toxicity.get_eval_results_path")
    @patch("fmeval.eval_algorithms.toxicity.cleanup_shared_resource")
    @patch("fmeval.eval_algorithms.toxicity.evaluate_dataset")
    @patch("fmeval.eval_algorithms.toxicity.create_shared_resource")
    @patch("fmeval.eval_algorithms.toxicity.TransformPipeline")
    @patch("fmeval.eval_algorithms.toxicity.get_dataset")
    @patch("fmeval.eval_algorithms.toxicity.get_dataset_configs")
    @patch("fmeval.eval_algorithms.toxicity.ToxicityScores")
    @patch("fmeval.eval_algorithms.toxicity.DetoxifyHelperModel")
    def test_evaluate(
        self,
        mock_detoxify_cls,
        mock_toxicity_scores_cls,
        mock_get_dataset_configs,
        mock_get_dataset,
        mock_transform_pipeline_cls,
        mock_create_shared_resource,
        mock_evaluate_dataset,
        mock_cleanup_shared_resource,
        mock_get_results_path,
        test_case,
    ):
        """
        GIVEN a SummarizationAccuracy instance.
        WHEN its evaluate method is called with valid arguments.
        THEN a new TransformPipeline that uses a BertscoreHelperModel shared resource
            is created, and `evaluate_dataset` is called with the correct arguments.
        """
        mock_detoxify_cls.return_value = Mock()
        mock_toxicity_scores_cls.return_value = Mock()

        pipeline = Mock()
        mock_transform_pipeline_cls.return_value = pipeline

        mock_get_results_path.return_value = "/path/to/results"
        model_runner = Mock()

        dataset_config = Mock()
        dataset_config.dataset_name = "my_custom_dataset"
        mock_get_dataset_configs.return_value = [dataset_config]

        mock_dataset = Mock()
        # So that validate_dataset does not error
        mock_dataset.columns = Mock(return_value=[DatasetColumns.MODEL_INPUT.value.name])
        mock_get_dataset.return_value = mock_dataset

        eval_algo = Toxicity()
        output = eval_algo.evaluate(
            model=model_runner,
            dataset_config=dataset_config,
            prompt_template=test_case.user_provided_prompt_template,
            num_records=162,
            save=True,
        )

        mock_create_shared_resource.assert_called_once_with(eval_algo._helper_model)
        mock_evaluate_dataset.assert_called_once_with(
            dataset=mock_dataset,
            pipeline=pipeline,
            dataset_name=dataset_config.dataset_name,
            eval_name=eval_algo.eval_name,
            metric_names=DETOXIFY_SCORE_NAMES,
            eval_results_path="/path/to/results",
            model=model_runner,
            prompt_template=test_case.dataset_prompt_template,
            agg_method=MEAN,
            save=True,
            save_strategy=None,
        )
        mock_transform_pipeline_cls.assert_called_once_with([mock_toxicity_scores_cls.return_value])
        mock_cleanup_shared_resource.assert_called_once_with(mock_create_shared_resource.return_value)
        assert output == [mock_evaluate_dataset.return_value]

    def test_toxicity_scores_detoxify(self):
        """
        GIVEN valid inputs
        WHEN __call__() method of ToxicityScores using DetoxifyHelperModel is called.
        THEN correct output is returned
        """
        helper_model = DetoxifyHelperModel()
        toxicity_scores = ToxicityScores(input_key="model_output", toxicity_helper_model=helper_model)
        actual_response = toxicity_scores({"model_output": np.array(["My toxic text", "My good text"])})
        expected_response = {
            "model_output": np.array(["My toxic text", "My good text"]),
            DETOXIFY_SCORE_TOXICITY: np.array([0.06483059376478195, 0.00045518550905399024]),
            DETOXIFY_SCORE_SEVERE_TOXICITY: np.array([1.26147870105342e-05, 1.6480657905049156e-06]),
            DETOXIFY_SCORE_OBSCENE: np.array([0.0009980567265301943, 3.1544899684377015e-05]),
            DETOXIFY_SCORE_IDENTITY_ATTACK: np.array([0.0012085289927199483, 6.863904854981229e-05]),
            DETOXIFY_SCORE_INSULT: np.array([0.00813359022140503, 8.761371282162145e-05]),
            DETOXIFY_SCORE_THREAT: np.array([0.0004742506134789437, 2.826379204634577e-05]),
            DETOXIFY_SCORE_SEXUAL_EXPLICIT: np.array([0.00044487009290605783, 1.9261064153397456e-05]),
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

    @patch.object(ToxigenHelperModel, "TOXIGEN_MODEL_NAME", new_callable=PropertyMock)
    def test_toxigen_helper_model_call(self, mock_model_name):
        """
        GIVEN valid inputs
        WHEN __call__() method of ToxicityScores using ToxigenHelperModel is called
        THEN correct output is returned
        Using lightweight test model: https://huggingface.co/hf-internal-testing/tiny-random-roberta
        """
        mock_model_name.return_value = "hf-internal-testing/tiny-random-roberta"
        helper_model = ToxigenHelperModel()
        toxicity_scores = ToxicityScores(input_key="prompt", toxicity_helper_model=helper_model)
        actual_response = toxicity_scores({"prompt": np.array(["My toxic text", "My good text"])})
        expected_response = {
            "prompt": np.array(["My toxic text", "My good text"]),
            TOXIGEN_SCORE_NAME: np.array([0.5005719, 0.5005644]),
        }
        assert actual_response.keys() == expected_response.keys()
        np.testing.assert_array_equal(actual_response["prompt"], expected_response["prompt"])
        np.testing.assert_almost_equal(actual_response[TOXIGEN_SCORE_NAME], expected_response["toxicity"])
