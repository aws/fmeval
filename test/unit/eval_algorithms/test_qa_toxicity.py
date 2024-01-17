import re
from typing import NamedTuple, List, Optional
from unittest.mock import patch, MagicMock

import pytest
import ray
from _pytest.fixtures import fixture
from ray.data import Dataset

from fmeval.constants import (
    ColumnNames,
    MIME_TYPE_JSON,
)
from fmeval.data_loaders.data_config import DataConfig
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
from fmeval.exceptions import EvalAlgorithmClientError

DATASET_TOXIGEN = ray.data.from_items(
    [
        {
            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "sample model input",
            ColumnNames.PROMPT_COLUMN_NAME.value: "sample model input",
            ColumnNames.CATEGORY_COLUMN_NAME.value: "dummy_category_1",
            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "sample_model_output",
            TOXIGEN_SCORE_NAME: 1.0,
        },
        {
            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "sample model input",
            ColumnNames.PROMPT_COLUMN_NAME.value: "sample model input",
            ColumnNames.CATEGORY_COLUMN_NAME.value: "dummy_category_2",
            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "sample_model_output",
            TOXIGEN_SCORE_NAME: 1.0,
        },
    ]
)

DATASET_DETOXIFY = ray.data.from_items(
    [
        {
            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "sample model input",
            ColumnNames.CATEGORY_COLUMN_NAME.value: "dummy_category_1",
            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "sample_model_output",
            DETOXIFY_SCORE_TOXICITY: 0.5,
            DETOXIFY_SCORE_SEVERE_TOXICITY: 0.5,
            DETOXIFY_SCORE_OBSCENE: 0.5,
            DETOXIFY_SCORE_IDENTITY_ATTACK: 0.5,
            DETOXIFY_SCORE_INSULT: 0.5,
            DETOXIFY_SCORE_THREAT: 0.5,
            DETOXIFY_SCORE_SEXUAL_EXPLICIT: 0.5,
        },
        {
            ColumnNames.MODEL_INPUT_COLUMN_NAME.value: "sample model input",
            ColumnNames.CATEGORY_COLUMN_NAME.value: "dummy_category_1",
            ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value: "sample_model_output",
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

    @patch.dict("fmeval.eval_algorithms.qa_toxicity.TOXICITY_HELPER_MODEL_MAPPING", {TOXIGEN_MODEL: get_toxigen_mock()})
    def test_toxicity_evaluate_sample_invalid_input(self, config):
        """
        GIVEN invalid inputs
        WHEN QAToxicity.evaluate_sample is called
        THEN expected error is raised
        """
        eval_algorithm = QAToxicity(config)
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
                    cols=[ColumnNames.PROMPT_COLUMN_NAME.value, ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value, ColumnNames.CATEGORY_COLUMN_NAME.value, TOXIGEN_SCORE_NAME]
                ),
                dataset_config=None,
                prompt_template=None,
                input_dataset_with_generated_model_output=DATASET_TOXIGEN.drop_columns(
                    cols=[ColumnNames.CATEGORY_COLUMN_NAME.value, TOXIGEN_SCORE_NAME]
                ),
                dataset_with_scores=DATASET_TOXIGEN.drop_columns(cols=[ColumnNames.CATEGORY_COLUMN_NAME.value]),
                expected_response=[
                    EvalOutput(
                        eval_name="qa_toxicity",
                        dataset_name=BOOLQ,
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[BOOLQ],
                        category_scores=None,
                        output_path="/tmp/eval_results/qa_toxicity_boolq.jsonl",
                    ),
                    EvalOutput(
                        eval_name="qa_toxicity",
                        dataset_name=TRIVIA_QA,
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[TRIVIA_QA],
                        category_scores=None,
                        output_path="/tmp/eval_results/qa_toxicity_trivia_qa.jsonl",
                    ),
                    EvalOutput(
                        eval_name="qa_toxicity",
                        dataset_name=NATURAL_QUESTIONS,
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[NATURAL_QUESTIONS],
                        category_scores=None,
                        output_path="/tmp/eval_results/qa_toxicity_natural_questions.jsonl",
                    ),
                ],
            ),
            # Built-in datasets evaluate for dataset with category
            TestCaseToxicityEvaluate(
                input_dataset=DATASET_TOXIGEN.drop_columns(
                    cols=[ColumnNames.PROMPT_COLUMN_NAME.value, ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value, TOXIGEN_SCORE_NAME]
                ),
                dataset_config=None,
                prompt_template=None,
                input_dataset_with_generated_model_output=DATASET_TOXIGEN.drop_columns(cols=[TOXIGEN_SCORE_NAME]),
                dataset_with_scores=DATASET_TOXIGEN,
                expected_response=[
                    EvalOutput(
                        eval_name="qa_toxicity",
                        dataset_name=BOOLQ,
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[BOOLQ],
                        category_scores=[
                            CategoryScore(name="dummy_category_1", scores=[EvalScore(name="toxicity", value=1.0)]),
                            CategoryScore(name="dummy_category_2", scores=[EvalScore(name="toxicity", value=1.0)]),
                        ],
                        output_path="/tmp/eval_results/qa_toxicity_boolq.jsonl",
                    ),
                    EvalOutput(
                        eval_name="qa_toxicity",
                        dataset_name=TRIVIA_QA,
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[TRIVIA_QA],
                        category_scores=[
                            CategoryScore(name="dummy_category_1", scores=[EvalScore(name="toxicity", value=1.0)]),
                            CategoryScore(name="dummy_category_2", scores=[EvalScore(name="toxicity", value=1.0)]),
                        ],
                        output_path="/tmp/eval_results/qa_toxicity_trivia_qa.jsonl",
                    ),
                    EvalOutput(
                        eval_name="qa_toxicity",
                        dataset_name=NATURAL_QUESTIONS,
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[NATURAL_QUESTIONS],
                        category_scores=[
                            CategoryScore(name="dummy_category_1", scores=[EvalScore(name="toxicity", value=1.0)]),
                            CategoryScore(name="dummy_category_2", scores=[EvalScore(name="toxicity", value=1.0)]),
                        ],
                        output_path="/tmp/eval_results/qa_toxicity_natural_questions.jsonl",
                    ),
                ],
            ),
            # Custom dataset evaluate with input prompt template
            TestCaseToxicityEvaluate(
                input_dataset=DATASET_TOXIGEN.drop_columns(
                    cols=[ColumnNames.PROMPT_COLUMN_NAME.value, ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value, ColumnNames.CATEGORY_COLUMN_NAME.value, TOXIGEN_SCORE_NAME]
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
                    cols=[ColumnNames.CATEGORY_COLUMN_NAME.value, TOXIGEN_SCORE_NAME]
                ),
                dataset_with_scores=DATASET_TOXIGEN.drop_columns(cols=[ColumnNames.CATEGORY_COLUMN_NAME.value]),
                expected_response=[
                    EvalOutput(
                        eval_name="qa_toxicity",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template="$feature",
                        category_scores=None,
                        output_path="/tmp/eval_results/qa_toxicity_my_custom_dataset.jsonl/",
                    )
                ],
            ),
            # Custom dataset evaluate without input prompt template
            TestCaseToxicityEvaluate(
                input_dataset=DATASET_TOXIGEN.drop_columns(
                    cols=[ColumnNames.PROMPT_COLUMN_NAME.value, ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value, ColumnNames.CATEGORY_COLUMN_NAME.value, TOXIGEN_SCORE_NAME]
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
                input_dataset_with_generated_model_output=DATASET_TOXIGEN.drop_columns(
                    cols=[ColumnNames.CATEGORY_COLUMN_NAME.value, TOXIGEN_SCORE_NAME]
                ),
                dataset_with_scores=DATASET_TOXIGEN.drop_columns(cols=[ColumnNames.CATEGORY_COLUMN_NAME.value]),
                expected_response=[
                    EvalOutput(
                        eval_name="qa_toxicity",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=None,
                        output_path="/tmp/eval_results/qa_toxicity_my_custom_dataset.jsonl",
                    )
                ],
            ),
        ],
    )
    @patch("fmeval.model_runners.model_runner.ModelRunner")
    @patch("fmeval.eval_algorithms.toxicity.get_dataset")
    @patch("fmeval.eval_algorithms.toxicity.save_dataset")
    @patch("fmeval.eval_algorithms.toxicity.generate_model_predict_response_for_dataset")
    @patch.object(QAToxicity, "_Toxicity__add_scores")
    @patch.dict("fmeval.eval_algorithms.qa_toxicity.TOXICITY_HELPER_MODEL_MAPPING", {TOXIGEN_MODEL: get_toxigen_mock()})
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
        WHEN QAToxicity evaluate() method is called
        THEN correct EvalOutput is returned
        """
        add_score_to_dataset.return_value = test_case.dataset_with_scores
        get_dataset.return_value = test_case.input_dataset
        generate_model_predict_response_for_dataset.return_value = test_case.input_dataset_with_generated_model_output
        eval_algorithm = QAToxicity(config)
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
                    cols=[ColumnNames.PROMPT_COLUMN_NAME.value, ColumnNames.CATEGORY_COLUMN_NAME.value, TOXIGEN_SCORE_NAME]
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
                dataset_with_scores=DATASET_TOXIGEN.drop_columns(cols=[ColumnNames.CATEGORY_COLUMN_NAME.value]),
                expected_response=[
                    EvalOutput(
                        eval_name="qa_toxicity",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[EvalScore(name="toxicity", value=1.0)],
                        prompt_template=None,
                        category_scores=None,
                        output_path="/tmp/eval_results/qa_toxicity_my_custom_dataset.jsonl",
                    )
                ],
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.toxicity.get_dataset")
    @patch("fmeval.eval_algorithms.toxicity.save_dataset")
    @patch("fmeval.eval_algorithms.toxicity.generate_model_predict_response_for_dataset")
    @patch.object(QAToxicity, "_Toxicity__add_scores")
    @patch.dict("fmeval.eval_algorithms.qa_toxicity.TOXICITY_HELPER_MODEL_MAPPING", {TOXIGEN_MODEL: get_toxigen_mock()})
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
        WHEN QAToxicity evaluate() method is called
        THEN correct EvalOutput is returned
        """
        add_score_to_dataset.return_value = test_case.dataset_with_scores
        get_dataset.return_value = test_case.input_dataset
        eval_algorithm = QAToxicity(config)
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
                    cols=[ColumnNames.PROMPT_COLUMN_NAME.value, ColumnNames.CATEGORY_COLUMN_NAME.value, TOXIGEN_SCORE_NAME, ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value]
                ),
                dataset_config=None,
                prompt_template=None,
                model_provided=False,
                expected_error_message="No ModelRunner provided. ModelRunner is required for inference on model_inputs",
            ),
            TestCaseToxicityEvaluateInvalid(
                input_dataset=DATASET_TOXIGEN.drop_columns(
                    cols=[ColumnNames.PROMPT_COLUMN_NAME.value, ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value, TOXIGEN_SCORE_NAME]
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
                        ColumnNames.PROMPT_COLUMN_NAME.value,
                        ColumnNames.CATEGORY_COLUMN_NAME.value,
                        TOXIGEN_SCORE_NAME,
                        ColumnNames.MODEL_OUTPUT_COLUMN_NAME.value,
                        ColumnNames.MODEL_INPUT_COLUMN_NAME.value,
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
        ],
    )
    @patch("fmeval.model_runners.model_runner.ModelRunner")
    @patch("fmeval.eval_algorithms.toxicity.get_dataset")
    @patch.dict("fmeval.eval_algorithms.qa_toxicity.TOXICITY_HELPER_MODEL_MAPPING", {TOXIGEN_MODEL: get_toxigen_mock()})
    def test_toxicity_evaluate_invalid_input(self, get_dataset, model, test_case, config):
        """
        GIVEN invalid inputs
        WHEN QAToxicity evaluate() method is called
        THEN correct exception with proper message is raised
        """
        eval_algorithm = QAToxicity(config)
        get_dataset.return_value = test_case.input_dataset
        if not test_case.model_provided:
            model = None
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            eval_algorithm.evaluate(
                model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template
            )
