import re
from typing import NamedTuple, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
import ray
from _pytest.fixtures import fixture
from ray.data import Dataset

from amazon_fmeval.constants import (
    MODEL_INPUT_COLUMN_NAME,
    CATEGORY_COLUMN_NAME,
    MIME_TYPE_JSON,
    DEFAULT_EVAL_RESULTS_PATH,
    MODEL_OUTPUT_COLUMN_NAME,
)
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.eval_algorithms import EvalScore, EvalOutput, CategoryScore, DEFAULT_PROMPT_TEMPLATE
from amazon_fmeval.eval_algorithms.general_semantic_robustness import (
    WER_SCORE,
    GeneralSemanticRobustnessConfig,
    GeneralSemanticRobustness,
    RANDOM_UPPER_CASE,
    WHITESPACE_ADD_REMOVE,
    BUTTER_FINGER,
)
from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.model_runner import ModelRunner

DATASET_WITH_MODEL_OUTPUT = ray.data.from_items(
    [
        {
            MODEL_INPUT_COLUMN_NAME: "What is the capital of England?",
            CATEGORY_COLUMN_NAME: "dummy_category_1",
            MODEL_OUTPUT_COLUMN_NAME: "Some model output.",
        },
        {
            MODEL_INPUT_COLUMN_NAME: "What is the capital of England?",
            CATEGORY_COLUMN_NAME: "dummy_category_2",
            MODEL_OUTPUT_COLUMN_NAME: "Some model output.",
        },
    ]
)

DATASET = DATASET_WITH_MODEL_OUTPUT.drop_columns(cols=MODEL_OUTPUT_COLUMN_NAME)

DATASET_NO_CATEGORY = DATASET.drop_columns(cols=CATEGORY_COLUMN_NAME)

DATASET_WITH_MODEL_OUTPUT_NO_CATEGORY = DATASET_WITH_MODEL_OUTPUT.drop_columns(cols=CATEGORY_COLUMN_NAME)


class ConstantModel(ModelRunner):
    def __init__(self):
        super().__init__('{"data": $prompt}')

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        return "Some model output.", None


class TestGeneralSemanticRobustness:
    @fixture(scope="module")
    def config(self) -> GeneralSemanticRobustnessConfig:
        return GeneralSemanticRobustnessConfig(num_perturbations=2)

    class TestCaseGeneralSemanticRobustnessEvaluateSample(NamedTuple):
        model_input: str
        # model_output: Optional[str]
        original_model_output: str
        perturbed_model_output_1: str
        perturbed_model_output_2: str
        expected_response: List[EvalScore]
        config: GeneralSemanticRobustnessConfig

    class TestCaseGeneralSemanticRobustnessEvaluateSampleInvalid(NamedTuple):
        model_input: str
        model: ModelRunner
        expected_error_message: str
        config: GeneralSemanticRobustnessConfig

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
                config=GeneralSemanticRobustnessConfig(num_perturbations=2),
            ),
            TestCaseGeneralSemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="Some model output.",
                perturbed_model_output_1="Another model output.",
                perturbed_model_output_2="Some model output.",
                expected_response=[
                    EvalScore(name=WER_SCORE, value=(1 / 3 + 0) / 2),
                ],
                config=GeneralSemanticRobustnessConfig(num_perturbations=2, perturbation_type=BUTTER_FINGER),
            ),
            TestCaseGeneralSemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="Some model output.",
                perturbed_model_output_1="Another model output.",
                perturbed_model_output_2="Some model output.",
                expected_response=[
                    EvalScore(name=WER_SCORE, value=(1 / 3 + 0) / 2),
                ],
                config=GeneralSemanticRobustnessConfig(num_perturbations=2, perturbation_type=RANDOM_UPPER_CASE),
            ),
            TestCaseGeneralSemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="Some model output.",
                perturbed_model_output_1="Another model output.",
                perturbed_model_output_2="Another model output.",
                expected_response=[
                    EvalScore(name=WER_SCORE, value=(1 / 3 + 1 / 3) / 2),
                ],
                config=GeneralSemanticRobustnessConfig(num_perturbations=2, perturbation_type=WHITESPACE_ADD_REMOVE),
            ),
        ],
    )
    def test_semantic_robustness_evaluate_sample(self, test_case):
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

        eval_algorithm = GeneralSemanticRobustness(test_case.config)
        assert eval_algorithm.evaluate_sample(test_case.model_input, model) == test_case.expected_response
        assert model.predict.call_count == 4

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
                config=GeneralSemanticRobustnessConfig(num_perturbations=2),
            ),
        ],
    )
    def test_semantic_robustness_evaluate_sample_with_model_output(self, test_case):
        """
        GIVEN valid inputs with model_output
        WHEN GeneralSemanticRobustness.evaluate_sample is called
        THEN correct List of EvalScores is returned
        """
        model = MagicMock()
        model.predict.side_effect = [
            (test_case.original_model_output,),
            (test_case.perturbed_model_output_1,),
            (test_case.perturbed_model_output_2,),
        ]

        eval_algorithm = GeneralSemanticRobustness(test_case.config)
        assert (
            eval_algorithm.evaluate_sample(test_case.model_input, model, test_case.original_model_output)
            == test_case.expected_response
        )
        assert model.predict.call_count == 3

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
                config=GeneralSemanticRobustnessConfig(num_perturbations=2),
            ),
        ],
    )
    def test_semantic_robustness_evaluate_sample_with_deterministic_model(self, test_case):
        """
        GIVEN valid inputs with model_output, and a deterministic model
        WHEN GeneralSemanticRobustness.evaluate_sample is called
        THEN correct List of EvalScores is returned
        """
        model = MagicMock()
        model.predict.side_effect = [
            (test_case.perturbed_model_output_1,),
            (test_case.perturbed_model_output_2,),
        ]
        eval_algorithm = GeneralSemanticRobustness(test_case.config)
        eval_algorithm._is_model_deterministic = True
        assert (
            eval_algorithm.evaluate_sample(
                model_input=test_case.model_input,
                model=model,
                model_output=test_case.original_model_output,
            )
            == test_case.expected_response
        )
        assert model.predict.call_count == 2

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseGeneralSemanticRobustnessEvaluateSampleInvalid(
                model_input="I like cake.",
                model=None,
                expected_error_message="Missing required input: model i.e. ModelRunner, for GeneralSemanticRobustness "
                "evaluate_sample",
                config=GeneralSemanticRobustnessConfig(num_perturbations=2),
            ),
            TestCaseGeneralSemanticRobustnessEvaluateSampleInvalid(
                model_input=None,
                model=MagicMock(),
                expected_error_message="Missing required input: model_input, for GeneralSemanticRobustness "
                "evaluate_sample",
                config=GeneralSemanticRobustnessConfig(num_perturbations=2),
            ),
        ],
    )
    def test_semantic_robustness_evaluate_sample_invalid_input(self, test_case):
        """
        GIVEN invalid inputs
        WHEN GeneralSemanticRobustness.evaluate_sample is called
        THEN correct exception with proper message is raised
        """
        eval_algorithm = GeneralSemanticRobustness(test_case.config)
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
                config=GeneralSemanticRobustnessConfig(num_perturbations=2),
            )
        ],
    )
    def test_semantic_robustness_evaluate_sample_invalid_model(self, test_case):
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

        eval_algorithm = GeneralSemanticRobustness(test_case.config)
        with pytest.raises(
            EvalAlgorithmClientError, match="For evaluating semantic robustness, the model should be deterministic."
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
        input_dataset_with_generated_model_output: Dataset
        prompt_template: Optional[str]
        dataset_config: Optional[DataConfig]
        expected_response: List[EvalOutput]
        save_data: bool

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSemanticRobustnessEvaluate(
                input_dataset=DATASET,
                input_dataset_with_generated_model_output=None,
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
                save_data=False,
                expected_response=None,
            ),
        ],
    )
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.get_dataset")
    def test_semantic_robustness_evaluate_invalid_model(self, get_dataset, test_case, config):
        """
        GIVEN a non-deterministic model
        WHEN GeneralSemanticRobustness.evaluate is called
        THEN correct exception with proper message is raised
        """
        model = MagicMock()
        original_model_output = "some model output"
        model.predict.side_effect = [
            (original_model_output,),
            (original_model_output + "1",),
        ]
        get_dataset.return_value = test_case.input_dataset
        eval_algorithm = GeneralSemanticRobustness(config)
        with pytest.raises(
            EvalAlgorithmClientError, match="For evaluating semantic robustness, the model should be deterministic."
        ):
            eval_algorithm.evaluate(model, test_case.dataset_config, prompt_template=test_case.prompt_template)

    @pytest.mark.parametrize(
        "test_case",
        [
            # Built-in datasets evaluate for dataset without category
            TestCaseSemanticRobustnessEvaluate(
                input_dataset=DATASET_NO_CATEGORY,
                input_dataset_with_generated_model_output=DATASET_WITH_MODEL_OUTPUT_NO_CATEGORY,
                dataset_config=None,
                prompt_template=None,
                save_data=True,
                expected_response=[
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="bold",
                        dataset_scores=[EvalScore(name="word_error_rate", value=0.0)],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=None,
                        output_path=DEFAULT_EVAL_RESULTS_PATH,
                    ),
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="trex",
                        dataset_scores=[EvalScore(name="word_error_rate", value=0.0)],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=None,
                        output_path=DEFAULT_EVAL_RESULTS_PATH,
                    ),
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="wikitext2",
                        dataset_scores=[EvalScore(name="word_error_rate", value=0.0)],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=None,
                        output_path=DEFAULT_EVAL_RESULTS_PATH,
                    ),
                ],
            ),
            # Built-in datasets evaluate for dataset with category
            TestCaseSemanticRobustnessEvaluate(
                input_dataset=DATASET,
                input_dataset_with_generated_model_output=DATASET_WITH_MODEL_OUTPUT,
                dataset_config=None,
                prompt_template=None,
                save_data=True,
                expected_response=[
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="bold",
                        dataset_scores=[EvalScore(name="word_error_rate", value=0.0)],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1", scores=[EvalScore(name="word_error_rate", value=0.0)]
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[EvalScore(name="word_error_rate", value=0.0)],
                            ),
                        ],
                        output_path=DEFAULT_EVAL_RESULTS_PATH,
                    ),
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="trex",
                        dataset_scores=[EvalScore(name="word_error_rate", value=0.0)],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1", scores=[EvalScore(name="word_error_rate", value=0.0)]
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[EvalScore(name="word_error_rate", value=0.0)],
                            ),
                        ],
                        output_path=DEFAULT_EVAL_RESULTS_PATH,
                    ),
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="wikitext2",
                        dataset_scores=[EvalScore(name="word_error_rate", value=0.0)],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1", scores=[EvalScore(name="word_error_rate", value=0.0)]
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[EvalScore(name="word_error_rate", value=0.0)],
                            ),
                        ],
                        output_path=DEFAULT_EVAL_RESULTS_PATH,
                    ),
                ],
            ),
            # Custom dataset evaluate with input prompt template
            TestCaseSemanticRobustnessEvaluate(
                input_dataset=DATASET_NO_CATEGORY,
                input_dataset_with_generated_model_output=DATASET_WITH_MODEL_OUTPUT_NO_CATEGORY,
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                prompt_template="Answer: $feature",
                save_data=False,
                expected_response=[
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[EvalScore(name="word_error_rate", value=0.0)],
                        prompt_template="Answer: $feature",
                        category_scores=None,
                        output_path="/tmp/eval_results/",
                    ),
                ],
            ),
            # Custom dataset evaluate without input prompt template
            TestCaseSemanticRobustnessEvaluate(
                input_dataset=DATASET_NO_CATEGORY,
                input_dataset_with_generated_model_output=DATASET_WITH_MODEL_OUTPUT_NO_CATEGORY,
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
                save_data=False,
                expected_response=[
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[EvalScore(name="word_error_rate", value=0.0)],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=None,
                        output_path=DEFAULT_EVAL_RESULTS_PATH,
                    ),
                ],
            ),
        ],
    )
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.get_dataset")
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.save_dataset")
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.generate_model_predict_response_for_dataset")
    def test_semantic_robustness_evaluate(
        self,
        generate_model_predict_response_for_dataset,
        save_dataset,
        get_dataset,
        test_case,
        config,
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset without model_outputs, an input ModelRunner
            and request to save records with scores
        WHEN GeneralSemanticRobustness evaluate() method is called
        THEN correct EvalOutput is returned
        """
        get_dataset.return_value = test_case.input_dataset
        generate_model_predict_response_for_dataset.return_value = test_case.input_dataset_with_generated_model_output
        eval_algorithm = GeneralSemanticRobustness(config)
        actual_response = eval_algorithm.evaluate(
            model=ConstantModel(),
            dataset_config=test_case.dataset_config,
            prompt_template=test_case.prompt_template,
            save=test_case.save_data,
        )
        assert save_dataset.called == test_case.save_data
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
        ],
    )
    @patch("amazon_fmeval.model_runners.model_runner.ModelRunner")
    @patch("amazon_fmeval.eval_algorithms.general_semantic_robustness.get_dataset")
    def test_semantic_robustness_evaluate_invalid_input(
        self,
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
        eval_algorithm = GeneralSemanticRobustness(config)
        get_dataset.return_value = test_case.input_dataset
        if not test_case.model_provided:
            model = None
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            eval_algorithm.evaluate(
                model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template
            )
