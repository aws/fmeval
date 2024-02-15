import random
import re
import string
from typing import NamedTuple, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
import ray
from _pytest.fixtures import fixture
from ray.data import Dataset

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms import EvalScore, EvalOutput, CategoryScore, DEFAULT_PROMPT_TEMPLATE
from fmeval.eval_algorithms.general_semantic_robustness import (
    WER_SCORE,
    BERT_SCORE_DISSIMILARITY,
    GeneralSemanticRobustnessConfig,
    GeneralSemanticRobustness,
    RANDOM_UPPER_CASE,
    WHITESPACE_ADD_REMOVE,
    BUTTER_FINGER,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.constants import BertscoreModels

BERTSCORE_DUMMY_VALUE = (
    0.5  # we don't always evaluate the real BERTScore inside unit tests to reduce runtime, so we hardcode a dummy value
)
BERTSCORE_DISSIMILARITY_DUMMY_VALUE = (
    1 - BERTSCORE_DUMMY_VALUE
)  # By definition, BERT_SCORE_DISSIMILARITY = 1 - BERT_SCORE


DATASET_WITH_SCORES = ray.data.from_items(
    [
        {
            DatasetColumns.MODEL_INPUT.value.name: "What is the capital of England?",
            DatasetColumns.CATEGORY.value.name: "dummy_category_1",
            DatasetColumns.MODEL_OUTPUT.value.name: "Some model output.",
            WER_SCORE: 0.0,
            BERT_SCORE_DISSIMILARITY: BERTSCORE_DUMMY_VALUE,
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "What is the capital of England?",
            DatasetColumns.CATEGORY.value.name: "dummy_category_2",
            DatasetColumns.MODEL_OUTPUT.value.name: "Some model output.",
            WER_SCORE: 0.0,
            BERT_SCORE_DISSIMILARITY: BERTSCORE_DUMMY_VALUE,
        },
    ]
)

DATASET_WITH_ONLY_BERT_SCORE = DATASET_WITH_SCORES.drop_columns(cols=WER_SCORE)

DATASET_WITH_MODEL_OUTPUT = DATASET_WITH_ONLY_BERT_SCORE.drop_columns(cols=BERT_SCORE_DISSIMILARITY)

DATASET = DATASET_WITH_MODEL_OUTPUT.drop_columns(cols=DatasetColumns.MODEL_OUTPUT.value.name)

DATASET_NO_CATEGORY = DATASET.drop_columns(cols=DatasetColumns.CATEGORY.value.name)

DATASET_WITH_MODEL_OUTPUT_NO_CATEGORY = DATASET_WITH_MODEL_OUTPUT.drop_columns(cols=DatasetColumns.CATEGORY.value.name)


class ConstantModel(ModelRunner):
    def __init__(self):
        super().__init__('{"data": $prompt}', output="output")

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        return "Some model output.", None


class NonDeterministicModel(ModelRunner):
    """A model that always returns some random strings in the output regardless of the input."""

    def __init__(self, num_letters: int = 10):
        """
        :param num_letters: The number of letters in the random string.
        """
        super().__init__('{"data": $prompt}', output="output")
        self.num_letters = num_letters

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        return "".join(random.choices(string.ascii_letters, k=self.num_letters)), None


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
                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
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
                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
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
                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
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
                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                    EvalScore(name=WER_SCORE, value=(1 / 3 + 1 / 3) / 2),
                ],
                config=GeneralSemanticRobustnessConfig(num_perturbations=2, perturbation_type=WHITESPACE_ADD_REMOVE),
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.general_semantic_robustness.BertscoreHelperModel")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.get_bert_score")
    def test_semantic_robustness_evaluate_sample(self, mock_get_bert_score, bertscore_helper_model, test_case):
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
        bertscore_helper_model_instance = MagicMock()
        bertscore_helper_model.return_value = bertscore_helper_model_instance
        mock_get_bert_score.return_value = BERTSCORE_DUMMY_VALUE

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
                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                    EvalScore(name=WER_SCORE, value=0.0),
                ],
                config=GeneralSemanticRobustnessConfig(num_perturbations=2),
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.general_semantic_robustness.BertscoreHelperModel")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.get_bert_score")
    def test_semantic_robustness_evaluate_sample_with_model_output(
        self, mock_get_bert_score, bertscore_helper_model, test_case
    ):
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
        bertscore_helper_model_instance = MagicMock()
        bertscore_helper_model.return_value = bertscore_helper_model_instance
        mock_get_bert_score.return_value = BERTSCORE_DUMMY_VALUE

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
                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                    EvalScore(name=WER_SCORE, value=0.0),
                ],
                config=GeneralSemanticRobustnessConfig(num_perturbations=2),
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.general_semantic_robustness.BertscoreHelperModel")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.get_bert_score")
    def test_semantic_robustness_evaluate_sample_with_deterministic_model(
        self, mock_get_bert_score, bertscore_helper_model, test_case
    ):
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
        bertscore_helper_model_instance = MagicMock()
        bertscore_helper_model.return_value = bertscore_helper_model_instance
        mock_get_bert_score.return_value = BERTSCORE_DUMMY_VALUE

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
    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreHelperModel")
    def test_semantic_robustness_evaluate_sample_invalid_input(self, bertscore_helper_model, test_case):
        """
        GIVEN invalid inputs
        WHEN GeneralSemanticRobustness.evaluate_sample is called
        THEN correct exception with proper message is raised
        """
        # We mock the BertscoreHelperModel class so that an actual ray actor doesn't get created
        bertscore_helper_model_instance = MagicMock()
        bertscore_helper_model.return_value = bertscore_helper_model_instance

        eval_algorithm = GeneralSemanticRobustness(test_case.config)
        with pytest.raises(EvalAlgorithmClientError, match=test_case.expected_error_message):
            eval_algorithm.evaluate_sample(test_case.model_input, test_case.model)

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseGeneralSemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="Some model output.",
                perturbed_model_output_1="Another longer model output.",
                perturbed_model_output_2="Yet another model output which is even longer.",
                expected_response=[
                    EvalScore(name=WER_SCORE, value=0),
                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=0),
                ],
                config=GeneralSemanticRobustnessConfig(num_perturbations=2, num_baseline_samples=2),
            )
        ],
    )
    @patch("fmeval.eval_algorithms.general_semantic_robustness.BertscoreHelperModel")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.get_bert_score")
    def test_semantic_robustness_evaluate_sample_non_deterministic_model(
        self, mock_get_bert_score, bertscore_helper_model, test_case
    ):
        """
        GIVEN a non-deterministic model
        WHEN GeneralSemanticRobustness.evaluate_sample is called
        THEN the robustness score valueis smaller than it would be for a deterministic model.
        """
        bertscore_helper_model_instance = MagicMock()
        bertscore_helper_model.return_value = bertscore_helper_model_instance
        mock_get_bert_score.return_value = BERTSCORE_DUMMY_VALUE

        deterministic_model = MagicMock()
        deterministic_model.predict.side_effect = [
            (test_case.original_model_output,),  # Original model output
            (test_case.original_model_output,),  # The determinism check
            (test_case.perturbed_model_output_1,),  # Output on the first perturbation
            (test_case.perturbed_model_output_2,),  # Output on the second perturbation
        ]

        nondeterministic_model = MagicMock()
        nondeterministic_model.predict.side_effect = [
            (test_case.original_model_output,),  # Original model output
            (test_case.original_model_output + "1",),  # The determinism check
            (test_case.perturbed_model_output_1,),  # Output on the first perturbation
            (test_case.perturbed_model_output_2,),  # Output on the second perturbation
            (test_case.original_model_output + "1",),  # Computing baseline: first model call
            (test_case.original_model_output + "1",),  # Computing baseline: second model call
        ]

        eval_algorithm = GeneralSemanticRobustness(test_case.config)
        output_deterministic = eval_algorithm.evaluate_sample(test_case.model_input, deterministic_model)
        output_nondeterministic = eval_algorithm.evaluate_sample(test_case.model_input, nondeterministic_model)
        assert output_nondeterministic[0].value < output_deterministic[0].value  # BERTScore Dissimilarity
        assert output_nondeterministic[1].value < output_deterministic[1].value  # WER

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
    def test_semantic_robustness_invalid_perturbation_type(self, perturbation_type, expected_error_message):
        """
        GIVEN invalid perturbation types
        WHEN GeneralSemanticRobustnessConfig is initiated
        THEN correct exception with proper message is raised
        """
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(expected_error_message)):
            GeneralSemanticRobustnessConfig(perturbation_type=perturbation_type)

    def test_semantic_robustness_invalid_bertscore_model(self):
        """
        GIVEN invalid bertscore model
        WHEN GeneralSemanticRobustnessConfig is initiated
        THEN correct exception with proper message is raised
        """
        model_name = "my_model"
        expected_error_message = (
            f"Invalid model_type_for_bertscore: {model_name} requested in GeneralSemanticRobustnessConfig, "
            f"please choose from acceptable values: {BertscoreModels.model_list()}."
        )
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(expected_error_message)):
            GeneralSemanticRobustnessConfig(model_type_for_bertscore=model_name)

    def test_semantic_robustness_invalid_num_baseline_samples(self):
        """
        GIVEN invalid number of baseline samples
        WHEN GeneralSemanticRobustnessConfig is initiated
        THEN correct exception with proper message is raised
        """
        num_baseline_samples = 1
        expected_error_message = (
            f"Invalid num_baseline_samples: {num_baseline_samples} in GeneralSemanticRobusntessConfig. "
            f"The value should be at least 2."
        )
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(expected_error_message)):
            GeneralSemanticRobustnessConfig(num_baseline_samples=num_baseline_samples)

    class TestCaseSemanticRobustnessEvaluate(NamedTuple):
        input_dataset: Dataset
        input_dataset_with_generated_model_output: Dataset
        prompt_template: Optional[str]
        dataset_config: Optional[DataConfig]
        expected_response: List[EvalOutput]
        save_data: bool
        dataset_with_scores: Dataset

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
                dataset_with_scores=DATASET_WITH_SCORES.drop_columns(cols=DatasetColumns.CATEGORY.value.name),
                expected_response=[
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="bold",
                        dataset_scores=[
                            EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                            EvalScore(name=WER_SCORE, value=0.0),
                        ],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=None,
                        output_path="/tmp/eval_results/general_semantic_robustness_bold.jsonl",
                    ),
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="trex",
                        dataset_scores=[
                            EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                            EvalScore(name=WER_SCORE, value=0.0),
                        ],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=None,
                        output_path="/tmp/eval_results/general_semantic_robustness_trex.jsonl",
                    ),
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="wikitext2",
                        dataset_scores=[
                            EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                            EvalScore(name=WER_SCORE, value=0.0),
                        ],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=None,
                        output_path="/tmp/eval_results/general_semantic_robustness_wikitext2.jsonl",
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
                dataset_with_scores=DATASET_WITH_SCORES,
                expected_response=[
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="bold",
                        dataset_scores=[
                            EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                            EvalScore(name=WER_SCORE, value=0.0),
                        ],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1",
                                scores=[
                                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                                    EvalScore(name=WER_SCORE, value=0.0),
                                ],
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[
                                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                                    EvalScore(name=WER_SCORE, value=0.0),
                                ],
                            ),
                        ],
                        output_path="/tmp/eval_results/general_semantic_robustness_bold.jsonl",
                    ),
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="trex",
                        dataset_scores=[
                            EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                            EvalScore(name=WER_SCORE, value=0.0),
                        ],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1",
                                scores=[
                                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                                    EvalScore(name=WER_SCORE, value=0.0),
                                ],
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[
                                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                                    EvalScore(name=WER_SCORE, value=0.0),
                                ],
                            ),
                        ],
                        output_path="/tmp/eval_results/general_semantic_robustness_trex.jsonl",
                    ),
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="wikitext2",
                        dataset_scores=[
                            EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                            EvalScore(name=WER_SCORE, value=0.0),
                        ],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1",
                                scores=[
                                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                                    EvalScore(name=WER_SCORE, value=0.0),
                                ],
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[
                                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                                    EvalScore(name=WER_SCORE, value=0.0),
                                ],
                            ),
                        ],
                        output_path="/tmp/eval_results/general_semantic_robustness_wikitext2.jsonl",
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
                dataset_with_scores=DATASET_WITH_SCORES.drop_columns(cols=DatasetColumns.CATEGORY.value.name),
                expected_response=[
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[
                            EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                            EvalScore(name=WER_SCORE, value=0.0),
                        ],
                        prompt_template="Answer: $feature",
                        category_scores=None,
                        output_path="/tmp/eval_results/general_semantic_robustness_my_custom_dataset.jsonl",
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
                dataset_with_scores=DATASET_WITH_SCORES.drop_columns(cols=DatasetColumns.CATEGORY.value.name),
                expected_response=[
                    EvalOutput(
                        eval_name="general_semantic_robustness",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[
                            EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                            EvalScore(name=WER_SCORE, value=0.0),
                        ],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=None,
                        output_path="/tmp/eval_results/general_semantic_robustness_my_custom_dataset.jsonl",
                    ),
                ],
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.general_semantic_robustness.get_dataset")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.save_dataset")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.generate_model_predict_response_for_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreHelperModel")
    @patch.object(GeneralSemanticRobustness, "_GeneralSemanticRobustness__add_scores_to_dataset")
    def test_semantic_robustness_evaluate(
        self,
        add_scores_to_dataset,
        bertscore_helper_model,
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
        add_scores_to_dataset.return_value = test_case.dataset_with_scores

        # We mock the BertscoreHelperModel class so that an actual ray actor doesn't get created
        bertscore_helper_model_instance = MagicMock()
        bertscore_helper_model.return_value = bertscore_helper_model_instance

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
                input_dataset=DATASET_NO_CATEGORY.drop_columns(cols=[DatasetColumns.MODEL_INPUT.value.name]),
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
    @patch("fmeval.eval_algorithms.general_semantic_robustness.get_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreHelperModel")
    def test_semantic_robustness_evaluate_invalid_input(
        self,
        bertscore_helper_model,
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
        # We mock the BertscoreHelperModel class so that an actual ray actor doesn't get created
        bertscore_helper_model_instance = MagicMock()
        bertscore_helper_model.return_value = bertscore_helper_model_instance

        eval_algorithm = GeneralSemanticRobustness(config)
        get_dataset.return_value = test_case.input_dataset
        if not test_case.model_provided:
            model = None
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            eval_algorithm.evaluate(
                model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template
            )
