import re
from typing import NamedTuple, List, Optional
from unittest.mock import patch, MagicMock

import pytest
import ray
from _pytest.fixtures import fixture
from ray.data import Dataset

from amazon_fmeval.constants import (
    MODEL_INPUT_COLUMN_NAME,
    MODEL_OUTPUT_COLUMN_NAME,
    CATEGORY_COLUMN_NAME,
    MIME_TYPE_JSON,
)
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.eval_algorithms import EvalScore, EvalOutput, CategoryScore
from amazon_fmeval.eval_algorithms.general_semantic_robustness import (
    PROMPT_COLUMN_NAME,
    WER_SCORE,
    GeneralSemanticRobustnessConfig,
    GeneralSemanticRobustness,
    RANDOM_UPPER_CASE,
    WHITESPACE_ADD_REMOVE,
    add_perturbed_prompts,
    BUTTER_FINGER,
    compute_wer,
)
from amazon_fmeval.eval_algorithms.helper_models.semantic_preserving_perturbations import ButterFingerConfig
from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.composers.composers import PromptComposer
from amazon_fmeval.model_runners.model_runner import ModelRunner

PERTURBED_INPUT_MODEL_OUTPUT_0 = "perturbed_input_model_output_0"
PERTURBED_INPUT_MODEL_OUTPUT_1 = "perturbed_input_model_output_1"

DATASET_WITH_SCORES_AND_PERTURBED_INPUT_OUTPUTS = ray.data.from_items(
    [
        {
            MODEL_INPUT_COLUMN_NAME: "What is the capital of England?",
            PROMPT_COLUMN_NAME: "What is the capital of England?",
            MODEL_OUTPUT_COLUMN_NAME: "Some model output.",
            CATEGORY_COLUMN_NAME: "dummy_category_1",
            WER_SCORE: (1 / 3 + 0) / 2,
            PERTURBED_INPUT_MODEL_OUTPUT_1: "Some model output.",
            PERTURBED_INPUT_MODEL_OUTPUT_0: "Some model output.",
        },
        {
            MODEL_INPUT_COLUMN_NAME: "What is the capital of England?",
            PROMPT_COLUMN_NAME: "What is the capital of England?",
            MODEL_OUTPUT_COLUMN_NAME: "Some model output.",
            CATEGORY_COLUMN_NAME: "dummy_category_2",
            WER_SCORE: (1 / 3 + 0) / 2,
            PERTURBED_INPUT_MODEL_OUTPUT_1: "Another model output.",
            PERTURBED_INPUT_MODEL_OUTPUT_0: "Some model output.",
        },
    ]
)

DATASET = DATASET_WITH_SCORES_AND_PERTURBED_INPUT_OUTPUTS.drop_columns(
    cols=[PERTURBED_INPUT_MODEL_OUTPUT_1, PERTURBED_INPUT_MODEL_OUTPUT_0, WER_SCORE, MODEL_OUTPUT_COLUMN_NAME]
)

DATASET_NO_CATEGORY = DATASET.drop_columns(cols=CATEGORY_COLUMN_NAME)

EVAL_RESULTS_PATH = "/tmp/eval_results/"


class TestGeneralSemanticRobustness:
    @fixture(scope="module")
    def config(self) -> GeneralSemanticRobustnessConfig:
        return GeneralSemanticRobustnessConfig(num_perturbations=2)

    class TestCaseGeneralSemanticRobustnessEvaluateSample(NamedTuple):
        model_input: str
        original_model_output: str
        perturbed_model_output_1: str
        perturbed_model_output_2: str
        expected_response: List[EvalScore]

    class TestCaseGeneralSemanticRobustnessEvaluateSampleInvalid(NamedTuple):
        model_input: str
        model: ModelRunner
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseGeneralSemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="Some model output.",
                perturbed_model_output_1="Some model output.",
                perturbed_model_output_2="Some model output.",
                expected_response=[
                    EvalScore(name=WER_SCORE, value=0.0),
                ],
            ),
            TestCaseGeneralSemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="Some model output.",
                perturbed_model_output_1="Another model output.",
                perturbed_model_output_2="Some model output.",
                expected_response=[
                    EvalScore(name=WER_SCORE, value=(1 / 3 + 0) / 2),
                ],
            ),
        ],
    )
    def test_semantic_robustness_evaluate_sample(self, test_case, config):
        """
        GIVEN valid inputs
        WHEN GeneralSemanticRobustness.evaluate_sample is called
        THEN correct List of EvalScores is returned
        """
        model = MagicMock()
        model.predict.side_effect = [
            (test_case.original_model_output,),
            (test_case.original_model_output,),
            (test_case.perturbed_model_output_1,),
            (test_case.perturbed_model_output_2,),
        ]

        eval_algorithm = GeneralSemanticRobustness(config)
        assert eval_algorithm.evaluate_sample(test_case.model_input, model) == test_case.expected_response

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseGeneralSemanticRobustnessEvaluateSampleInvalid(
                model_input="I like cake.",
                model=None,
                expected_error_message="Missing required input: model i.e. ModelRunner, for GeneralSemanticRobustness "
                "evaluate_sample",
            ),
            TestCaseGeneralSemanticRobustnessEvaluateSampleInvalid(
                model_input=None,
                model=MagicMock(),
                expected_error_message="Missing required input: model_input, for GeneralSemanticRobustness "
                "evaluate_sample",
            ),
        ],
    )
    def test_semantic_robustness_evaluate_sample_invalid_input(self, test_case, config):
        """
        GIVEN invalid inputs
        WHEN GeneralSemanticRobustness.evaluate_sample is called
        THEN correct exception with proper message is raised
        """
        eval_algorithm = GeneralSemanticRobustness(config)
        with pytest.raises(EvalAlgorithmClientError, match=test_case.expected_error_message):
            eval_algorithm.evaluate_sample(test_case.model_input, test_case.model)

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseGeneralSemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="Some model output.",
                perturbed_model_output_1="Some model output.",
                perturbed_model_output_2="Some model output.",
                expected_response=None,
            )
        ],
    )
    def test_semantic_robustness_evaluate_sample_invalid_model(self, test_case, config):
        """
        GIVEN a non-deterministic model
        WHEN GeneralSemanticRobustness.evaluate_sample is called
        THEN correct exception with proper message is raised
        """
        model = MagicMock()
        model.predict.side_effect = [
            (test_case.original_model_output,),
            (test_case.original_model_output + "1",),
            (test_case.perturbed_model_output_1,),
            (test_case.perturbed_model_output_2,),
        ]

        eval_algorithm = GeneralSemanticRobustness(config)
        eval_algorithm = GeneralSemanticRobustness(config)
        with pytest.raises(
            EvalAlgorithmClientError, match="For evaluating semantic robustness, the model should be " "deterministic."
        ):
            eval_algorithm.evaluate_sample(test_case.model_input, model)

    @pytest.mark.parametrize(
        "perturbation_type, expected_error_message",
        [
            (
                "my_perturb",
                "Invalid perturbation type 'my_perturb requested, please choose from acceptable values: "
                "dict_keys(['butter_finger', 'random_upper_case', 'whitespace_add_remove'])",
            )
        ],
    )
    def test_semantic_robustness_invalid_config(self, perturbation_type, expected_error_message):
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(expected_error_message)):
            GeneralSemanticRobustnessConfig(perturbation_type=perturbation_type)

    class TestCaseSemanticRobustnessEvaluate(NamedTuple):
        input_dataset: Dataset
        prompt_template: Optional[str]
        dataset_config: Optional[DataConfig]
        dataset_with_generated_model_output: Optional[Dataset]
        expected_response: List[EvalOutput]

    @pytest.mark.parametrize(
        "test_case",
        [
            # Built-in datasets evaluate for dataset without category
            TestCaseSemanticRobustnessEvaluate(
                input_dataset=DATASET_NO_CATEGORY,
                dataset_config=None,
                prompt_template=None,
                dataset_with_generated_model_output=DATASET_WITH_SCORES_AND_PERTURBED_INPUT_OUTPUTS.drop_columns(
                    cols=[CATEGORY_COLUMN_NAME, WER_SCORE]
                ),
                expected_response=[
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="bold",
                        dataset_scores=[EvalScore(name="word_error_rate", value=(1 / 3 + 0) / 4)],
                        prompt_template="$feature",
                        category_scores=None,
                        output_path="/tmp/eval_results/",
                    ),
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="trex",
                        dataset_scores=[EvalScore(name="word_error_rate", value=(1 / 3 + 0) / 4)],
                        prompt_template="$feature",
                        category_scores=None,
                        output_path="/tmp/eval_results/",
                    ),
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="wikitext2",
                        dataset_scores=[EvalScore(name="word_error_rate", value=(1 / 3 + 0) / 4)],
                        prompt_template="$feature",
                        category_scores=None,
                        output_path="/tmp/eval_results/",
                    ),
                ],
            ),
            # Built-in datasets evaluate for dataset with category
            TestCaseSemanticRobustnessEvaluate(
                input_dataset=DATASET,
                dataset_config=None,
                prompt_template=None,
                dataset_with_generated_model_output=DATASET_WITH_SCORES_AND_PERTURBED_INPUT_OUTPUTS.drop_columns(
                    cols=[WER_SCORE]
                ),
                expected_response=[
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="bold",
                        dataset_scores=[EvalScore(name="word_error_rate", value=(1 / 3 + 0) / 4)],
                        prompt_template="$feature",
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1", scores=[EvalScore(name="word_error_rate", value=0.0)]
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[EvalScore(name="word_error_rate", value=(1 / 3 + 0) / 2)],
                            ),
                        ],
                        output_path="/tmp/eval_results/",
                    ),
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="trex",
                        dataset_scores=[EvalScore(name="word_error_rate", value=(1 / 3 + 0) / 4)],
                        prompt_template="$feature",
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1", scores=[EvalScore(name="word_error_rate", value=0.0)]
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[EvalScore(name="word_error_rate", value=(1 / 3 + 0) / 2)],
                            ),
                        ],
                        output_path="/tmp/eval_results/",
                    ),
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="wikitext2",
                        dataset_scores=[EvalScore(name="word_error_rate", value=(1 / 3 + 0) / 4)],
                        prompt_template="$feature",
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1", scores=[EvalScore(name="word_error_rate", value=0.0)]
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[EvalScore(name="word_error_rate", value=(1 / 3 + 0) / 2)],
                            ),
                        ],
                        output_path="/tmp/eval_results/",
                    ),
                ],
            ),
            # Custom dataset evaluate
            TestCaseSemanticRobustnessEvaluate(
                input_dataset=DATASET_NO_CATEGORY,
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
                dataset_with_generated_model_output=DATASET_WITH_SCORES_AND_PERTURBED_INPUT_OUTPUTS.drop_columns(
                    cols=[CATEGORY_COLUMN_NAME, WER_SCORE]
                ),
                expected_response=[
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[EvalScore(name="word_error_rate", value=(1 / 3 + 0) / 4)],
                        prompt_template="$feature",
                        category_scores=None,
                        output_path="/tmp/eval_results/",
                    ),
                ],
            ),
        ],
    )
    @patch("amazon_fmeval.model_runners.model_runner.ModelRunner")
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.get_dataset")
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.save_dataset")
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.generate_model_predict_response_for_dataset")
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.is_predictor_deterministic")
    def test_semantic_robustness_evaluate(
        self,
        is_predictor_deterministic,
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
        WHEN GeneralSemanticRobustness evaluate() method is called
        THEN correct EvalOutput is returned
        """
        is_predictor_deterministic.return_value = True
        get_dataset.return_value = test_case.input_dataset
        generate_model_predict_response_for_dataset.return_value = test_case.dataset_with_generated_model_output
        eval_algorithm = GeneralSemanticRobustness(config)
        actual_response = eval_algorithm.evaluate(
            model=model, dataset_config=test_case.dataset_config, save=True, prompt_template=test_case.prompt_template
        )
        assert save_dataset.called
        assert actual_response == test_case.expected_response

    @pytest.mark.parametrize(
        "test_case, eval_algo_config",
        [
            (
                TestCaseSemanticRobustnessEvaluate(
                    input_dataset=DATASET_NO_CATEGORY,
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
                    dataset_with_generated_model_output=DATASET_WITH_SCORES_AND_PERTURBED_INPUT_OUTPUTS.drop_columns(
                        cols=[CATEGORY_COLUMN_NAME, WER_SCORE]
                    ),
                    expected_response=[
                        EvalOutput(
                            eval_name="general_semantic_robustness",
                            dataset_name="my_custom_dataset",
                            dataset_scores=[EvalScore(name="word_error_rate", value=(1 / 3 + 0) / 4)],
                            prompt_template="$feature",
                            category_scores=None,
                            output_path="/tmp/eval_results/",
                        ),
                    ],
                ),
                GeneralSemanticRobustnessConfig(perturbation_type=RANDOM_UPPER_CASE, num_perturbations=2),
            ),
            (
                TestCaseSemanticRobustnessEvaluate(
                    input_dataset=DATASET_NO_CATEGORY,
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
                    dataset_with_generated_model_output=DATASET_WITH_SCORES_AND_PERTURBED_INPUT_OUTPUTS.drop_columns(
                        cols=[CATEGORY_COLUMN_NAME, WER_SCORE]
                    ),
                    expected_response=[
                        EvalOutput(
                            eval_name="general_semantic_robustness",
                            dataset_name="my_custom_dataset",
                            dataset_scores=[EvalScore(name="word_error_rate", value=(1 / 3 + 0) / 4)],
                            prompt_template="$feature",
                            category_scores=None,
                            output_path="/tmp/eval_results/",
                        ),
                    ],
                ),
                GeneralSemanticRobustnessConfig(perturbation_type=WHITESPACE_ADD_REMOVE, num_perturbations=2),
            ),
        ],
    )
    @patch("amazon_fmeval.model_runners.model_runner.ModelRunner")
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.get_dataset")
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.save_dataset")
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.generate_model_predict_response_for_dataset")
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.is_predictor_deterministic")
    def test_semantic_robustness_evaluate_different_perturbation_type(
        self,
        is_predictor_deterministic,
        generate_model_predict_response_for_dataset,
        save_dataset,
        get_dataset,
        model,
        test_case,
        eval_algo_config,
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset without model_outputs, an input ModelRunner
            and request to save records with scores
        WHEN GeneralSemanticRobustness evaluate() method is called
        THEN correct EvalOutput is returned
        """
        is_predictor_deterministic.return_value = True
        get_dataset.return_value = test_case.input_dataset
        generate_model_predict_response_for_dataset.return_value = test_case.dataset_with_generated_model_output
        eval_algorithm = GeneralSemanticRobustness(eval_algo_config)
        actual_response = eval_algorithm.evaluate(
            model=model, dataset_config=test_case.dataset_config, save=False, prompt_template=test_case.prompt_template
        )
        assert not save_dataset.called
        assert actual_response == test_case.expected_response

    class TestCaseSemanticRobustnessEvaluateInvalid(NamedTuple):
        input_dataset: Dataset
        dataset_config: Optional[DataConfig]
        prompt_template: Optional[str]
        model_provided: bool
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSemanticRobustnessEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY,
                dataset_config=None,
                prompt_template=None,
                model_provided=False,
                expected_error_message="Missing required input: model i.e. ModelRunner, for GeneralSemanticRobustness "
                "evaluate method",
            ),
            TestCaseSemanticRobustnessEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(cols=[MODEL_INPUT_COLUMN_NAME]),
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
            TestCaseSemanticRobustnessEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY,
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
            TestCaseSemanticRobustnessEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY,
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
                prompt_template="$feature",
                expected_error_message="For evaluating semantic robustness, the model should be deterministic.",
            ),
        ],
    )
    @patch("amazon_fmeval.model_runners.model_runner.ModelRunner")
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.get_dataset")
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.generate_model_predict_response_for_dataset")
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.is_predictor_deterministic")
    def test_semantic_robustness_evaluate_invalid_input(
        self,
        is_predictor_deterministic,
        generate_model_predict_response_for_dataset,
        get_dataset,
        model,
        test_case,
        config,
    ):
        """
        GIVEN invalid inputs
        WHEN GeneralSemanticRobustness evaluate is called
        THEN correct exception with proper message is raised
        """
        generate_model_predict_response_for_dataset.return_value = (
            DATASET_WITH_SCORES_AND_PERTURBED_INPUT_OUTPUTS.drop_columns(
                cols=[PERTURBED_INPUT_MODEL_OUTPUT_1, PERTURBED_INPUT_MODEL_OUTPUT_0, WER_SCORE, CATEGORY_COLUMN_NAME]
            )
        )
        is_predictor_deterministic.return_value = False
        eval_algorithm = GeneralSemanticRobustness(config)
        get_dataset.return_value = test_case.input_dataset
        if not test_case.model_provided:
            model = None
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            eval_algorithm.evaluate(
                model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template
            )


def test_add_perturbed_prompts():
    """
    GIVEN valid inputs
    WHEN add_perturbed_prompts method is called
    THEN row Dict is returned with expected columns
    """
    input_row = {MODEL_INPUT_COLUMN_NAME: "This is an input"}

    expected_response = {
        MODEL_INPUT_COLUMN_NAME: "This is an input",
        "perturbed_input_prompt_0": "Summarise: This is dn imput",
        "perturbed_input_prompt_1": "Summarise: This ia an inpit",
    }

    assert expected_response == add_perturbed_prompts(
        row=input_row,
        model_input_column=MODEL_INPUT_COLUMN_NAME,
        perturbation_type=BUTTER_FINGER,
        seed=5,
        num_perturbations=2,
        perturbation_config=ButterFingerConfig(),
        prompt_composer=PromptComposer("Summarise: $feature"),
    )


def test_compute_wer():
    """
    GIVEN valid inputs
    WHEN compute_wer method is called
    THEN row Dict is returned with expected score column
    """
    input_row = {
        MODEL_OUTPUT_COLUMN_NAME: "Some model output.",
        "perturbed_input_model_output_0": "Some model output.",
        "perturbed_input_model_output_1": "Another model output.",
    }

    expected_response = {
        MODEL_OUTPUT_COLUMN_NAME: "Some model output.",
        "perturbed_input_model_output_0": "Some model output.",
        "perturbed_input_model_output_1": "Another model output.",
        WER_SCORE: (1 / 3 + 0) / 2,
    }

    assert expected_response == compute_wer(
        row=input_row,
        original_output_column=MODEL_OUTPUT_COLUMN_NAME,
        num_perturbations=2,
        perturbed_input_output_column="perturbed_input_model_output",
    )
