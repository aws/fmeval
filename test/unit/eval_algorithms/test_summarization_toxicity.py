import re
from typing import NamedTuple, List, Optional
from unittest.mock import patch, MagicMock

import pytest
import ray
from _pytest.fixtures import fixture
from ray.data import Dataset

from amazon_fmeval.constants import (
    MODEL_INPUT_COLUMN_NAME,
    CATEGORY_COLUMN_NAME,
    MODEL_OUTPUT_COLUMN_NAME,
    MIME_TYPE_JSON,
)
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.eval_algorithms import EvalScore, EvalOutput, CategoryScore
from amazon_fmeval.eval_algorithms.helper_models.helper_model import (
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
from amazon_fmeval.eval_algorithms.summarization_toxicity import SummarizationToxicity
from amazon_fmeval.eval_algorithms.toxicity import (
    ToxicityConfig,
    TOXIGEN_MODEL,
    DETOXIFY_MODEL,
    PROMPT_COLUMN_NAME,
)
from amazon_fmeval.exceptions import EvalAlgorithmClientError

DATASET_TOXIGEN = ray.data.from_items(
    [
        {
            MODEL_INPUT_COLUMN_NAME: "sample model input",
            PROMPT_COLUMN_NAME: "Summarize: sample model input",
            CATEGORY_COLUMN_NAME: "dummy_category_1",
            MODEL_OUTPUT_COLUMN_NAME: "sample_model_output",
            TOXIGEN_SCORE_NAME: 1.0,
        },
        {
            MODEL_INPUT_COLUMN_NAME: "sample model input",
            PROMPT_COLUMN_NAME: "Summarize: sample model input",
            CATEGORY_COLUMN_NAME: "dummy_category_2",
            MODEL_OUTPUT_COLUMN_NAME: "sample_model_output",
            TOXIGEN_SCORE_NAME: 1.0,
        },
    ]
)

DATASET_DETOXIFY = ray.data.from_items(
    [
        {
            MODEL_INPUT_COLUMN_NAME: "sample model input",
            PROMPT_COLUMN_NAME: "Summarize: sample model input",
            CATEGORY_COLUMN_NAME: "dummy_category_1",
            MODEL_OUTPUT_COLUMN_NAME: "sample_model_output",
            DETOXIFY_SCORE_TOXICITY: 0.5,
            DETOXIFY_SCORE_SEVERE_TOXICITY: 0.5,
            DETOXIFY_SCORE_OBSCENE: 0.5,
            DETOXIFY_SCORE_IDENTITY_ATTACK: 0.5,
            DETOXIFY_SCORE_INSULT: 0.5,
            DETOXIFY_SCORE_THREAT: 0.5,
            DETOXIFY_SCORE_SEXUAL_EXPLICIT: 0.5,
        },
        {
            MODEL_INPUT_COLUMN_NAME: "sample model input",
            PROMPT_COLUMN_NAME: "Summarize: sample model input",
            CATEGORY_COLUMN_NAME: "dummy_category_1",
            MODEL_OUTPUT_COLUMN_NAME: "sample_model_output",
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


class TestSummarizationToxicityToxicity:
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
    @patch.dict(
        "amazon_fmeval.eval_algorithms.summarization_toxicity.TOXICITY_HELPER_MODEL_MAPPING",
        {TOXIGEN_MODEL: get_toxigen_mock()},
    )
    def test_toxicity_evaluate_sample_toxigen(self, test_case, config):
        """
        GIVEN valid inputs
        WHEN SummarizationToxicity.evaluate_sample with toxigen model_type is called
        THEN correct List of EvalScores is returned
        """
        eval_algorithm = SummarizationToxicity(config)
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
        "amazon_fmeval.eval_algorithms.summarization_toxicity.TOXICITY_HELPER_MODEL_MAPPING",
        {DETOXIFY_MODEL: get_detoxify_mock()},
    )
    def test_toxicity_evaluate_sample_detoxify(self, test_case):
        """
        GIVEN valid inputs
        WHEN SummarizationToxicity.evaluate_sample with detoxify model_type is called
        THEN correct List of EvalScores is returned
        """
        config = ToxicityConfig()
        eval_algorithm = SummarizationToxicity(config)
        assert eval_algorithm.evaluate_sample(test_case.model_output) == test_case.expected_response

    @patch.dict(
        "amazon_fmeval.eval_algorithms.summarization_toxicity.TOXICITY_HELPER_MODEL_MAPPING",
        {TOXIGEN_MODEL: get_toxigen_mock()},
    )
    def test_toxicity_evaluate_sample_invalid_input(self, config):
        """
        GIVEN invalid inputs
        WHEN SummarizationToxicity.evaluate_sample is called
        THEN expected error is raised
        """
        eval_algorithm = SummarizationToxicity(config)
        expected_error_message = "Missing required input: target_output, for Toxicity evaluate_sample"
        with pytest.raises(EvalAlgorithmClientError, match=expected_error_message):
            eval_algorithm.evaluate_sample(None)

    class TestCaseToxicityEvaluate(NamedTuple):
        input_dataset: Dataset
        prompt_template: Optional[str]
        dataset_config: Optional[DataConfig]
        input_dataset_with_generated_model_output: Optional[Dataset]
        expected_response: List[EvalOutput]
        dataset_with_scores: Dataset

    @pytest.mark.parametrize(
        "test_case",
        [
            # Built-in datasets evaluate for dataset without category
            TestCaseToxicityEvaluate(
                input_dataset=DATASET_TOXIGEN.drop_columns(
                    cols=[PROMPT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME, CATEGORY_COLUMN_NAME, TOXIGEN_SCORE_NAME]
                ),
                dataset_config=None,
                prompt_template=None,
                input_dataset_with_generated_model_output=DATASET_TOXIGEN.drop_columns(
                    cols=[CATEGORY_COLUMN_NAME, TOXIGEN_SCORE_NAME]
                ),
                dataset_with_scores=DATASET_TOXIGEN.drop_columns(cols=[CATEGORY_COLUMN_NAME]),
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_toxicity",
                        dataset_name="cnn_daily_mail",
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template="Summarise: $feature",
                        category_scores=None,
                        output_path="/tmp/eval_results/",
                    ),
                    EvalOutput(
                        eval_name="summarization_toxicity",
                        dataset_name="xsum",
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template="Summarise: $feature",
                        category_scores=None,
                        output_path="/tmp/eval_results/",
                    ),
                ],
            ),
            # Built-in datasets evaluate for dataset with category
            TestCaseToxicityEvaluate(
                input_dataset=DATASET_TOXIGEN.drop_columns(
                    cols=[PROMPT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME, TOXIGEN_SCORE_NAME]
                ),
                dataset_config=None,
                prompt_template=None,
                input_dataset_with_generated_model_output=DATASET_TOXIGEN.drop_columns(cols=[TOXIGEN_SCORE_NAME]),
                dataset_with_scores=DATASET_TOXIGEN,
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_toxicity",
                        dataset_name="cnn_daily_mail",
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template="Summarise: $feature",
                        category_scores=[
                            CategoryScore(name="dummy_category_1", scores=[EvalScore(name="toxicity", value=1.0)]),
                            CategoryScore(name="dummy_category_2", scores=[EvalScore(name="toxicity", value=1.0)]),
                        ],
                        output_path="/tmp/eval_results/",
                    ),
                    EvalOutput(
                        eval_name="summarization_toxicity",
                        dataset_name="xsum",
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template="Summarise: $feature",
                        category_scores=[
                            CategoryScore(name="dummy_category_1", scores=[EvalScore(name="toxicity", value=1.0)]),
                            CategoryScore(name="dummy_category_2", scores=[EvalScore(name="toxicity", value=1.0)]),
                        ],
                        output_path="/tmp/eval_results/",
                    ),
                ],
            ),
            # Custom dataset evaluate
            TestCaseToxicityEvaluate(
                input_dataset=DATASET_TOXIGEN.drop_columns(
                    cols=[PROMPT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME, CATEGORY_COLUMN_NAME, TOXIGEN_SCORE_NAME]
                ),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                prompt_template="$feature",
                input_dataset_with_generated_model_output=DATASET_TOXIGEN.drop_columns(
                    cols=[CATEGORY_COLUMN_NAME, TOXIGEN_SCORE_NAME]
                ),
                dataset_with_scores=DATASET_TOXIGEN.drop_columns(cols=[CATEGORY_COLUMN_NAME]),
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_toxicity",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template="$feature",
                        category_scores=None,
                        output_path="/tmp/eval_results/",
                    )
                ],
            ),
        ],
    )
    @patch("amazon_fmeval.model_runners.model_runner.ModelRunner")
    @patch("amazon_fmeval.eval_algorithms.toxicity.get_dataset")
    @patch("amazon_fmeval.eval_algorithms.toxicity.save_dataset")
    @patch("amazon_fmeval.eval_algorithms.toxicity.generate_model_predict_response_for_dataset")
    @patch.object(SummarizationToxicity, "_Toxicity__add_scores")
    @patch.dict(
        "amazon_fmeval.eval_algorithms.qa_toxicity.TOXICITY_HELPER_MODEL_MAPPING", {TOXIGEN_MODEL: get_toxigen_mock()}
    )
    def test_toxicity_evaluate(
        self,
        add_score_to_dataset,
        generate_model_predict_response_for_dataset,
        save_dataset,
        get_dataset,
        model,
        test_case,
        config,
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset without model_outputs, an input ModelRunner
            and request to save records with scores
        WHEN SummarizationToxicity evaluate() method is called
        THEN correct EvalOutput is returned
        """
        add_score_to_dataset.return_value = test_case.dataset_with_scores
        get_dataset.return_value = test_case.input_dataset
        generate_model_predict_response_for_dataset.return_value = test_case.input_dataset_with_generated_model_output
        eval_algorithm = SummarizationToxicity(config)
        actual_response = eval_algorithm.evaluate(
            model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template, save=True
        )
        assert save_dataset.called
        assert actual_response == test_case.expected_response

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseToxicityEvaluate(
                input_dataset=DATASET_TOXIGEN.drop_columns(
                    cols=[PROMPT_COLUMN_NAME, CATEGORY_COLUMN_NAME, TOXIGEN_SCORE_NAME]
                ),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                prompt_template="$feature",
                input_dataset_with_generated_model_output=None,
                dataset_with_scores=DATASET_TOXIGEN.drop_columns(cols=[CATEGORY_COLUMN_NAME]),
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_toxicity",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template="$feature",
                        category_scores=None,
                        output_path="/tmp/eval_results/",
                    )
                ],
            ),
        ],
    )
    @patch("amazon_fmeval.eval_algorithms.toxicity.get_dataset")
    @patch("amazon_fmeval.eval_algorithms.toxicity.save_dataset")
    @patch("amazon_fmeval.eval_algorithms.toxicity.generate_model_predict_response_for_dataset")
    @patch.object(SummarizationToxicity, "_Toxicity__add_scores")
    @patch.dict(
        "amazon_fmeval.eval_algorithms.summarization_toxicity.TOXICITY_HELPER_MODEL_MAPPING",
        {TOXIGEN_MODEL: get_toxigen_mock()},
    )
    def test_toxicity_evaluate_no_model(
        self,
        add_score_to_dataset,
        generate_model_predict_response_for_dataset,
        save_dataset,
        get_dataset,
        test_case,
        config,
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset without model_outputs, an input ModelRunner
            and request to save records with scores
        WHEN SummarizationToxicity evaluate() method is called
        THEN correct EvalOutput is returned
        """
        add_score_to_dataset.return_value = test_case.dataset_with_scores
        get_dataset.return_value = test_case.input_dataset
        eval_algorithm = SummarizationToxicity(config)
        actual_response = eval_algorithm.evaluate(
            model=None, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template, save=False
        )
        assert not save_dataset.called
        assert not generate_model_predict_response_for_dataset.called
        assert actual_response == test_case.expected_response

    class TestCaseToxicityEvaluateInvalid(NamedTuple):
        input_dataset: Dataset
        dataset_config: Optional[DataConfig]
        prompt_template: Optional[str]
        model_provided: bool
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseToxicityEvaluateInvalid(
                input_dataset=DATASET_TOXIGEN.drop_columns(
                    cols=[PROMPT_COLUMN_NAME, CATEGORY_COLUMN_NAME, TOXIGEN_SCORE_NAME, MODEL_OUTPUT_COLUMN_NAME]
                ),
                dataset_config=None,
                prompt_template=None,
                model_provided=False,
                expected_error_message="No ModelRunner provided. ModelRunner is required for inference on model_inputs",
            ),
            TestCaseToxicityEvaluateInvalid(
                input_dataset=DATASET_TOXIGEN.drop_columns(
                    cols=[PROMPT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME, TOXIGEN_SCORE_NAME]
                ),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                model_provided=False,
                prompt_template=None,
                expected_error_message="No ModelRunner provided. ModelRunner is required for inference on model_inputs",
            ),
            TestCaseToxicityEvaluateInvalid(
                input_dataset=DATASET_TOXIGEN.drop_columns(
                    cols=[
                        PROMPT_COLUMN_NAME,
                        CATEGORY_COLUMN_NAME,
                        TOXIGEN_SCORE_NAME,
                        MODEL_OUTPUT_COLUMN_NAME,
                        MODEL_INPUT_COLUMN_NAME,
                    ]
                ),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                prompt_template=None,
                model_provided=True,
                expected_error_message="Missing required column: model_input, for evaluate() method",
            ),
            TestCaseToxicityEvaluateInvalid(
                input_dataset=DATASET_TOXIGEN.drop_columns(
                    cols=[PROMPT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME, TOXIGEN_SCORE_NAME]
                ),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                model_provided=True,
                prompt_template=None,
                expected_error_message="Missing required input: prompt_template for evaluating custom dataset :",
            ),
        ],
    )
    @patch("amazon_fmeval.model_runners.model_runner.ModelRunner")
    @patch("amazon_fmeval.eval_algorithms.toxicity.get_dataset")
    @patch.dict(
        "amazon_fmeval.eval_algorithms.summarization_toxicity.TOXICITY_HELPER_MODEL_MAPPING",
        {TOXIGEN_MODEL: get_toxigen_mock()},
    )
    def test_toxicity_evaluate_invalid_input(self, get_dataset, model, test_case, config):
        """
        GIVEN invalid inputs
        WHEN SummarizationToxicity evaluate() method is called
        THEN correct exception with proper message is raised
        """
        eval_algorithm = SummarizationToxicity(config)
        get_dataset.return_value = test_case.input_dataset
        if not test_case.model_provided:
            model = None
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            eval_algorithm.evaluate(
                model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template
            )
