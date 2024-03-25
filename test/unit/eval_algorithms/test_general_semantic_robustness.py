import itertools
import random
import re
import string
from typing import NamedTuple, List, Optional, Tuple
from unittest.mock import MagicMock, patch, Mock

import pytest
import ray
from _pytest.fixtures import fixture

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON,
)
from fmeval.eval_algorithms import CategoryScore, EvalOutput, EvalScore
from fmeval.eval_algorithms.general_semantic_robustness import (
    WER_SCORE,
    BERT_SCORE_DISSIMILARITY,
    GeneralSemanticRobustnessConfig,
    GeneralSemanticRobustness,
    RANDOM_UPPERCASE,
    ADD_REMOVE_WHITESPACE,
    BUTTER_FINGER,
    UpdateRobustnessScores,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.helper_models import BertscoreModel
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModelTypes
from fmeval.transforms.common import GeneratePrompt, GetModelOutputs
from fmeval.transforms.semantic_perturbations import (
    SemanticPerturbation,
    ButterFinger,
    RandomUppercase,
    AddRemoveWhitespace,
)
from fmeval.transforms.semantic_robustness_metrics import BertScoreDissimilarity, WER
from fmeval.transforms.summarization_accuracy_metrics import BertScore

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
            DatasetColumns.PROMPT.value.name: "unused since we mock the model output",
            DatasetColumns.MODEL_OUTPUT.value.name: "Some model output.",
            WER_SCORE: 0.0,
            BERT_SCORE_DISSIMILARITY: BERTSCORE_DUMMY_VALUE,
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "What is the capital of England?",
            DatasetColumns.CATEGORY.value.name: "dummy_category_2",
            DatasetColumns.PROMPT.value.name: "unused since we mock the model output",
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
        return GeneralSemanticRobustnessConfig(num_perturbations=2, num_baseline_samples=3)

    @pytest.mark.parametrize(
        "perturbation_type, expected_error_message",
        [
            (
                "my_perturb",
                "Invalid perturbation type 'my_perturb requested, please choose from acceptable values: "
                "dict_keys(['butter_finger', 'random_uppercase', 'add_remove_whitespace'])",
            )
        ],
    )
    def test_gsr_config_invalid_perturbation_type(self, perturbation_type, expected_error_message):
        """
        GIVEN invalid perturbation types
        WHEN GeneralSemanticRobustnessConfig is initialized
        THEN correct exception with proper message is raised
        """
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(expected_error_message)):
            GeneralSemanticRobustnessConfig(perturbation_type=perturbation_type)

    def test_gsr_config_invalid_bertscore_model(self):
        """
        GIVEN invalid bertscore model
        WHEN GeneralSemanticRobustnessConfig is initiated
        THEN correct exception with proper message is raised
        """
        model_name = "my_model"
        expected_error_message = (
            f"Invalid model_type_for_bertscore: {model_name} requested in GeneralSemanticRobustnessConfig, "
            f"please choose from acceptable values: {BertscoreHelperModelTypes.model_list()}."
        )
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(expected_error_message)):
            GeneralSemanticRobustnessConfig(model_type_for_bertscore=model_name)

    def test_gsr_config_invalid_num_baseline_samples(self):
        """
        GIVEN invalid number of baseline samples
        WHEN GeneralSemanticRobustnessConfig is initiated
        THEN correct exception with proper message is raised
        """
        num_baseline_samples = 1
        expected_error_message = (
            f"Invalid num_baseline_samples: {num_baseline_samples} in GeneralSemanticRobustnessConfig. "
            f"The value should be at least 2."
        )
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(expected_error_message)):
            GeneralSemanticRobustnessConfig(num_baseline_samples=num_baseline_samples)

    @pytest.mark.parametrize("use_ray", [True, False])
    @pytest.mark.parametrize("perturbation_type", [BUTTER_FINGER, RANDOM_UPPERCASE, ADD_REMOVE_WHITESPACE])
    @patch("fmeval.eval_algorithms.general_semantic_robustness.create_shared_resource")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.BertscoreModel")
    def test_init(self, bertscore_model, create_shared_resource, perturbation_type, use_ray):
        """
        GIVEN valid arguments.
        WHEN a GeneralSemanticRobustness is initialized.
        THEN its instance attributes match what is expected and `create_shared_resource` is called
            (or not called) depending on the `use_ray` flag.
        """
        bertscore_model.return_value = Mock(spec=BertscoreModel)

        config = GeneralSemanticRobustnessConfig(perturbation_type=perturbation_type)
        eval_algo = GeneralSemanticRobustness(config, use_ray=use_ray)

        assert eval_algo.num_perturbations == config.num_perturbations
        assert eval_algo.num_baseline_samples == config.num_baseline_samples

        perturbation = eval_algo.perturbation_transform
        if perturbation_type == BUTTER_FINGER:
            assert isinstance(perturbation, ButterFinger)
            assert perturbation.num_perturbations == config.num_perturbations
            assert perturbation.perturbation_prob == config.butter_finger_perturbation_prob
        elif perturbation_type == RANDOM_UPPERCASE:
            assert isinstance(perturbation, RandomUppercase)
            assert perturbation.num_perturbations == config.num_perturbations
            assert perturbation.uppercase_fraction == config.random_uppercase_corrupt_proportion
        else:
            assert isinstance(perturbation, AddRemoveWhitespace)
            assert perturbation.num_perturbations == config.num_perturbations
            assert perturbation.add_prob == config.whitespace_add_prob
            assert perturbation.remove_prob == config.whitespace_remove_prob

        if use_ray:
            create_shared_resource.assert_called_once()
        else:
            create_shared_resource.assert_not_called()
            bertscore_model.assert_called_with(config.model_type_for_bertscore)

    @pytest.mark.parametrize("is_deterministic", [True, False])
    @patch("fmeval.eval_algorithms.general_semantic_robustness.BertscoreModel")
    def test_build_pipeline(self, bertscore_model, is_deterministic, config):
        """
        GIVEN a deterministic model.
        WHEN `build_pipeline` is called.
        THEN a TransformPipeline with the correct Transforms is returned.
        """
        # Mock BertscoreModel so that the actual model doesn't get loaded into memory during test.
        bertscore_model.return_value = Mock(spec=BertscoreModel)

        eval_algo = GeneralSemanticRobustness(config, use_ray=False)
        pipeline = eval_algo.build_pipeline(
            model=Mock(),
            prompt_template="$model_input",
            is_deterministic=is_deterministic,
        )

        transforms = pipeline.transforms
        if is_deterministic:
            assert len(transforms) == 6
        else:
            assert len(transforms) == 11

        perturbation = transforms[0]
        gen_prompts = transforms[1]
        model_responses = transforms[2]
        bert_scores = transforms[3]
        bsd = transforms[4]
        wer = transforms[5]

        assert isinstance(perturbation, ButterFinger)  # default perturbation type used by config is BUTTER_FINGER
        assert isinstance(gen_prompts, GeneratePrompt)
        assert len(gen_prompts.output_keys) == config.num_perturbations
        assert isinstance(model_responses, GetModelOutputs)
        assert len(model_responses.output_keys) == config.num_perturbations
        assert isinstance(bert_scores, BertScore)
        assert len(bert_scores.output_keys) == config.num_perturbations
        assert isinstance(bsd, BertScoreDissimilarity)
        assert isinstance(wer, WER)

        if not is_deterministic:
            baseline_responses = transforms[6]
            baseline_bert = transforms[7]
            baseline_bsd = transforms[8]
            baseline_wer = transforms[9]
            update_scores = transforms[10]

            assert isinstance(baseline_responses, GetModelOutputs)
            assert len(baseline_responses.output_keys) == config.num_baseline_samples - 1
            assert isinstance(baseline_bert, BertScore)
            assert len(baseline_bert.output_keys) == len(
                list(itertools.combinations([i for i in range(config.num_baseline_samples)], 2))
            )
            assert isinstance(baseline_bsd, BertScoreDissimilarity)
            assert isinstance(baseline_wer, WER)
            assert isinstance(update_scores, UpdateRobustnessScores)

    class TestCaseEvaluateSample(NamedTuple):
        model_input: str
        original_model_output: str
        perturbed_model_output_1: str
        perturbed_model_output_2: str
        expected_response: List[EvalScore]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="Some model output.",
                perturbed_model_output_1="Some model output.",
                perturbed_model_output_2="Some model output.",
                expected_response=[
                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                    EvalScore(name=WER_SCORE, value=0.0),
                ],
            ),
            TestCaseEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="Some model output.",
                perturbed_model_output_1="Another model output.",
                perturbed_model_output_2="Some model output.",
                expected_response=[
                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                    EvalScore(name=WER_SCORE, value=(1 / 3 + 0) / 2),
                ],
            ),
            TestCaseEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="Some model output.",
                perturbed_model_output_1="Another model output.",
                perturbed_model_output_2="Some model output.",
                expected_response=[
                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                    EvalScore(name=WER_SCORE, value=(1 / 3 + 0) / 2),
                ],
            ),
            TestCaseEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="Some model output.",
                perturbed_model_output_1="Another model output.",
                perturbed_model_output_2="Another model output.",
                expected_response=[
                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
                    EvalScore(name=WER_SCORE, value=(1 / 3 + 1 / 3) / 2),
                ],
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.general_semantic_robustness.BertscoreModel")
    def test_evaluate_sample_deterministic_model(self, bertscore_model, test_case, config):
        """
        GIVEN a deterministic model
        WHEN GeneralSemanticRobustness.evaluate_sample is called
        THEN correct List of EvalScores is returned
        """
        model = Mock()
        model.predict.side_effect = [
            (test_case.original_model_output, None),  # Original model output
            (test_case.original_model_output, None),  # The determinism check
            (test_case.perturbed_model_output_1, None),  # Output on the first perturbation
            (test_case.perturbed_model_output_2, None),  # Output on the second perturbation
        ]
        bertscore_model_instance = Mock(spec=BertscoreModel)
        bertscore_model_instance.invoke_model = Mock(return_value=BERTSCORE_DUMMY_VALUE)
        bertscore_model.return_value = bertscore_model_instance

        eval_algo = GeneralSemanticRobustness(config, use_ray=False)
        assert eval_algo.evaluate_sample(test_case.model_input, model) == test_case.expected_response
        assert model.predict.call_count == 4

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="Some model output.",
                perturbed_model_output_1="Another longer model output.",
                perturbed_model_output_2="Yet another model output which is even longer.",
                expected_response=[
                    EvalScore(name=WER_SCORE, value=0),
                    EvalScore(name=BERT_SCORE_DISSIMILARITY, value=0),
                ],
            )
        ],
    )
    @patch("fmeval.eval_algorithms.general_semantic_robustness.BertscoreModel")
    def test_semantic_robustness_evaluate_sample_non_deterministic_model(self, bertscore_model, test_case, config):
        """
        GIVEN a non-deterministic model.
        WHEN GeneralSemanticRobustness.evaluate_sample is called.
        THEN the robustness score value is smaller than it would be for a deterministic model.
        """
        bertscore_model_instance = Mock(spec=BertscoreModel)
        bertscore_model_instance.invoke_model = Mock(return_value=BERTSCORE_DUMMY_VALUE)
        bertscore_model.return_value = bertscore_model_instance

        deterministic_model = MagicMock()
        deterministic_model.predict.side_effect = [
            (test_case.original_model_output, None),  # Original model output
            (test_case.original_model_output, None),  # The determinism check
            (test_case.perturbed_model_output_1, None),  # Output on the first perturbation
            (test_case.perturbed_model_output_2, None),  # Output on the second perturbation
        ]

        nondeterministic_model = MagicMock()
        nondeterministic_model.predict.side_effect = [
            (test_case.original_model_output, None),  # Original model output
            (test_case.original_model_output + "1", None),  # The determinism check
            (test_case.perturbed_model_output_1, None),  # Output on the first perturbation
            (test_case.perturbed_model_output_2, None),  # Output on the second perturbation
            (test_case.original_model_output + "1", None),  # Computing baseline: first model call
            (test_case.original_model_output + "1", None),  # Computing baseline: second model call
            (test_case.original_model_output + "1", None),  # Computing baseline: third model call
        ]

        eval_algorithm = GeneralSemanticRobustness(config, use_ray=False)
        output_deterministic = eval_algorithm.evaluate_sample(test_case.model_input, deterministic_model)
        output_nondeterministic = eval_algorithm.evaluate_sample(test_case.model_input, nondeterministic_model)
        assert output_nondeterministic[0].value < output_deterministic[0].value  # BERTScore Dissimilarity
        assert output_nondeterministic[1].value < output_deterministic[1].value  # WER

    class TestCaseEvaluate(NamedTuple):
        user_provided_prompt_template: Optional[str]
        dataset_prompt_template: str
        is_deterministic: bool

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseEvaluate(
                user_provided_prompt_template="Answer: $model_input",
                dataset_prompt_template="Answer: $model_input",
                is_deterministic=True,
            ),
            TestCaseEvaluate(
                user_provided_prompt_template=None,
                dataset_prompt_template="$model_input",
                is_deterministic=False,
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.general_semantic_robustness.save_dataset")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.GeneralSemanticRobustness.build_pipeline")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.TransformPipeline")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.verify_model_determinism")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.GetModelOutputs")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.GeneratePrompt")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.get_dataset")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.get_dataset_configs")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.BertscoreModel")
    def test_evaluate(
        self,
        bertscore_model,
        get_dataset_configs,
        get_dataset,
        generate_prompt,
        get_model_outputs,
        verify_model_determinism,
        transform_pipeline,
        build_pipeline,
        save_dataset,
        test_case,
        config,
    ):
        """
        GIVEN a valid dataset.
        WHEN the GeneralSemanticRobustness evaluate method is called with save=True.
        THEN two transform pipelines are created and executed (one for getting original
            model responses and the other for handling the GSR logic),
            the EvalOutputs that are returned contain the correct scores,
            the EvalOutputs indicate that the correct prompt templates were used,
            and save_dataset is called.
        """
        eval_algo = GeneralSemanticRobustness(
            config,
            use_ray=False,
        )

        model_runner = Mock()
        bertscore_model.return_value = Mock()

        generate_original_prompt = Mock(spec=GeneratePrompt)
        generate_original_prompt.output_keys = ["prompt"]
        generate_prompt.return_value = generate_original_prompt

        get_original_model_outputs = Mock(spec=GetModelOutputs)
        get_original_model_outputs.output_keys = ["model_output"]
        get_model_outputs.return_value = get_original_model_outputs

        dataset_config = Mock()
        dataset_config.dataset_name = "my_custom_dataset"
        get_dataset_configs.return_value = [dataset_config]

        mock_dataset = Mock()
        mock_dataset.columns = Mock(return_value=[DatasetColumns.MODEL_INPUT.value.name])
        get_dataset.return_value = mock_dataset

        verify_model_determinism.return_value = test_case.is_deterministic

        # First pipeline that gets executed generates prompt and model output columns.
        # Second pipeline (created via self.build_pipeline) that gets executed handles everything else.
        transform_pipeline.execute = Mock(return_value=DATASET_WITH_MODEL_OUTPUT)
        build_pipeline.return_value = Mock()
        build_pipeline.return_value.execute = Mock(return_value=DATASET_WITH_SCORES)

        # Expected outputs from calling `evaluate`.
        expected_dataset_scores = [
            EvalScore(name=BERT_SCORE_DISSIMILARITY, value=BERTSCORE_DISSIMILARITY_DUMMY_VALUE),
            EvalScore(name=WER_SCORE, value=0.0),
        ]
        expected_category_scores = [
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
        ]
        expected_outputs = [
            EvalOutput(
                eval_name="general_semantic_robustness",
                prompt_template=test_case.dataset_prompt_template,
                dataset_name="my_custom_dataset",
                dataset_scores=expected_dataset_scores,
                category_scores=expected_category_scores,
                output_path="/tmp/eval_results/general_semantic_robustness_my_custom_dataset.jsonl",
            )
        ]

        # Call `evaluate` and validate outputs.
        eval_outputs = eval_algo.evaluate(
            model=model_runner,
            dataset_config=dataset_config,
            prompt_template=test_case.user_provided_prompt_template,
            save=True,
        )

        generate_prompt.assert_called_with(
            input_keys=[DatasetColumns.MODEL_INPUT.value.name],
            output_keys=[DatasetColumns.PROMPT.value.name],
            prompt_template=test_case.dataset_prompt_template,
        )
        get_model_outputs.assert_called_with(
            input_to_output_keys={DatasetColumns.PROMPT.value.name: [DatasetColumns.MODEL_OUTPUT.value.name]},
            model_runner=model_runner,
        )
        transform_pipeline.assert_called_with([generate_original_prompt, get_original_model_outputs])
        build_pipeline.assert_called_with(
            model_runner, test_case.dataset_prompt_template, is_deterministic=test_case.is_deterministic
        )
        save_dataset.assert_called_once()
        assert eval_outputs == expected_outputs
