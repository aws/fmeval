import re
from typing import NamedTuple, List, Optional, Tuple
from unittest.mock import patch, MagicMock

import pytest
import ray
from _pytest.fixtures import fixture
from ray.data import Dataset

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms import (
    EvalScore,
    EvalOutput,
    CategoryScore,
    BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES,
    XSUM,
    DEFAULT_PROMPT_TEMPLATE,
    GIGAWORD,
    GOV_REPORT,
)
from fmeval.eval_algorithms.general_semantic_robustness import (
    RANDOM_UPPER_CASE,
    WHITESPACE_ADD_REMOVE,
)
from fmeval.eval_algorithms.summarization_accuracy import METEOR_SCORE, ROUGE_SCORE, BERT_SCORE
from fmeval.eval_algorithms.summarization_accuracy_semantic_robustness import (
    SummarizationAccuracySemanticRobustnessConfig,
    SummarizationAccuracySemanticRobustness,
    DELTA_METEOR_SCORE,
    DELTA_ROUGE_SCORE,
    DELTA_BERT_SCORE,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner

DATASET_WITH_SCORES = ray.data.from_items(
    [
        {
            DatasetColumns.MODEL_INPUT.value.name: "Cake is so delicious, I really like cake. I want to open a bakery when I grow up.",
            DatasetColumns.TARGET_OUTPUT.value.name: "I like cake.",
            DatasetColumns.CATEGORY.value.name: "dummy_category_1",
            DatasetColumns.MODEL_OUTPUT.value.name: "Some model output.",
            ROUGE_SCORE: 0.0,
            METEOR_SCORE: 0.0,
            BERT_SCORE: 0.0,
            DELTA_ROUGE_SCORE: 0.0,
            DELTA_METEOR_SCORE: 0.0,
            DELTA_BERT_SCORE: 0.0,
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "The art metropolis of Berlin inspires locals and visitors with its famous "
            "museum landscape and numerous UNESCO World Heritage sites."
            " It is also an international exhibition venue. "
            "You will find a selection of current and upcoming exhibitions here.",
            DatasetColumns.TARGET_OUTPUT.value.name: "Berlin: an art metropolis.",
            DatasetColumns.CATEGORY.value.name: "dummy_category_2",
            DatasetColumns.MODEL_OUTPUT.value.name: "Some model output.",
            ROUGE_SCORE: 0.0,
            METEOR_SCORE: 0.0,
            BERT_SCORE: 0.0,
            DELTA_ROUGE_SCORE: 0.0,
            DELTA_METEOR_SCORE: 0.0,
            DELTA_BERT_SCORE: 0.0,
        },
    ]
)

DATASET_WITH_MODEL_OUTPUT = DATASET_WITH_SCORES.drop_columns(
    cols=[DELTA_ROUGE_SCORE, DELTA_METEOR_SCORE, DELTA_BERT_SCORE]
)

DATASET = DATASET_WITH_MODEL_OUTPUT.drop_columns(cols=[DatasetColumns.MODEL_OUTPUT.value.name])

DATASET_NO_CATEGORY_WITH_MODEL_OUTPUT = DATASET_WITH_MODEL_OUTPUT.drop_columns(
    cols=[DatasetColumns.CATEGORY.value.name]
)

DATASET_NO_CATEGORY = DATASET.drop_columns(cols=[DatasetColumns.CATEGORY.value.name])


class MockModelRunner(ModelRunner):
    def __init__(self):
        super().__init__('{"data": $prompt}', output="output")

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        return "Some model output.", None


class TestSummarizationAccuracySemanticRobustness:
    @fixture(scope="module")
    def config(self) -> SummarizationAccuracySemanticRobustnessConfig:
        return SummarizationAccuracySemanticRobustnessConfig(num_perturbations=2)

    class TestCaseSummarizationAccuracySemanticRobustnessEvaluateSample(NamedTuple):
        model_input: str
        target_output: str
        original_model_output: str
        perturbed_model_output_1: str
        perturbed_model_output_2: str
        sa_eval_score_original: List[EvalScore]
        sa_eval_score_perturbed_1: List[EvalScore]
        sa_eval_score_perturbed_2: List[EvalScore]
        expected_response: List[EvalScore]
        config: SummarizationAccuracySemanticRobustnessConfig

    class TestCaseSummarizationAccuracySemanticRobustnessEvaluateSampleInvalid(NamedTuple):
        model_input: str
        target_output: str
        model: ModelRunner
        expected_error_message: str
        config: SummarizationAccuracySemanticRobustnessConfig

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSummarizationAccuracySemanticRobustnessEvaluateSample(
                model_input="Cake is so delicious, I really like cake. I want to open a bakery when I grow up.",
                target_output="I like cake.",
                original_model_output="Some model output.",
                perturbed_model_output_1="Some other model output.",
                perturbed_model_output_2="Some different model output.",
                sa_eval_score_original=[
                    EvalScore(name=METEOR_SCORE, value=1.0),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                    EvalScore(name=BERT_SCORE, value=0.5),
                ],
                sa_eval_score_perturbed_1=[
                    EvalScore(name=METEOR_SCORE, value=0.75),
                    EvalScore(name=ROUGE_SCORE, value=0.5),
                    EvalScore(name=BERT_SCORE, value=1.0),
                ],
                sa_eval_score_perturbed_2=[
                    EvalScore(name=METEOR_SCORE, value=0.5),
                    EvalScore(name=ROUGE_SCORE, value=0.2),
                    EvalScore(name=BERT_SCORE, value=0.5),
                ],
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=1.0),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                    EvalScore(name=BERT_SCORE, value=0.5),
                    EvalScore(name=DELTA_METEOR_SCORE, value=0.375),
                    EvalScore(name=DELTA_ROUGE_SCORE, value=0.65),
                    EvalScore(name=DELTA_BERT_SCORE, value=0.25),
                ],
                config=SummarizationAccuracySemanticRobustnessConfig(num_perturbations=2),
            ),
            TestCaseSummarizationAccuracySemanticRobustnessEvaluateSample(
                model_input="Cake is so delicious, I really like cake. I want to open a bakery when I grow up.",
                target_output="I like cake.",
                original_model_output="Some model output.",
                perturbed_model_output_1="Some other model output.",
                perturbed_model_output_2="Some different model output.",
                sa_eval_score_original=[
                    EvalScore(name=METEOR_SCORE, value=1.0),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                    EvalScore(name=BERT_SCORE, value=0.5),
                ],
                sa_eval_score_perturbed_1=[
                    EvalScore(name=METEOR_SCORE, value=0.75),
                    EvalScore(name=ROUGE_SCORE, value=0.5),
                    EvalScore(name=BERT_SCORE, value=1.0),
                ],
                sa_eval_score_perturbed_2=[
                    EvalScore(name=METEOR_SCORE, value=0.5),
                    EvalScore(name=ROUGE_SCORE, value=0.2),
                    EvalScore(name=BERT_SCORE, value=0.5),
                ],
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=1.0),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                    EvalScore(name=BERT_SCORE, value=0.5),
                    EvalScore(name=DELTA_METEOR_SCORE, value=0.375),
                    EvalScore(name=DELTA_ROUGE_SCORE, value=0.65),
                    EvalScore(name=DELTA_BERT_SCORE, value=0.25),
                ],
                config=SummarizationAccuracySemanticRobustnessConfig(
                    num_perturbations=2, perturbation_type=RANDOM_UPPER_CASE
                ),
            ),
            TestCaseSummarizationAccuracySemanticRobustnessEvaluateSample(
                model_input="Cake is so delicious, I really like cake. I want to open a bakery when I grow up.",
                target_output="I like cake.",
                original_model_output="Some model output.",
                perturbed_model_output_1="Some other model output.",
                perturbed_model_output_2="Some different model output.",
                sa_eval_score_original=[
                    EvalScore(name=METEOR_SCORE, value=1.0),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                    EvalScore(name=BERT_SCORE, value=0.5),
                ],
                sa_eval_score_perturbed_1=[
                    EvalScore(name=METEOR_SCORE, value=0.75),
                    EvalScore(name=ROUGE_SCORE, value=0.5),
                    EvalScore(name=BERT_SCORE, value=1.0),
                ],
                sa_eval_score_perturbed_2=[
                    EvalScore(name=METEOR_SCORE, value=0.5),
                    EvalScore(name=ROUGE_SCORE, value=0.2),
                    EvalScore(name=BERT_SCORE, value=0.5),
                ],
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=1.0),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                    EvalScore(name=BERT_SCORE, value=0.5),
                    EvalScore(name=DELTA_METEOR_SCORE, value=0.375),
                    EvalScore(name=DELTA_ROUGE_SCORE, value=0.65),
                    EvalScore(name=DELTA_BERT_SCORE, value=0.25),
                ],
                config=SummarizationAccuracySemanticRobustnessConfig(
                    num_perturbations=2, perturbation_type=WHITESPACE_ADD_REMOVE
                ),
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.ray.get")
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.SummarizationAccuracyActor")
    def test_semantic_robustness_evaluate_sample(self, summarization_accuracy_actor_class, ray_get, test_case):
        """
        GIVEN valid inputs
        WHEN SummarizationAccuracySemanticRobustness.evaluate_sample is called
        THEN correct List of EvalScores is returned
        """
        model = MagicMock()
        model.predict.side_effect = [
            (test_case.original_model_output,),
            (test_case.perturbed_model_output_1,),
            (test_case.perturbed_model_output_2,),
        ]

        evaluate_sample_invocation_results = [
            test_case.sa_eval_score_original,
            test_case.sa_eval_score_perturbed_1,
            test_case.sa_eval_score_perturbed_2,
        ]

        ray_get.side_effect = evaluate_sample_invocation_results
        summarization_accuracy_actor = MagicMock()
        summarization_accuracy_actor_class.return_value = summarization_accuracy_actor

        eval_algorithm = SummarizationAccuracySemanticRobustness(test_case.config)
        assert (
            eval_algorithm.evaluate_sample(test_case.model_input, test_case.target_output, model)
            == test_case.expected_response
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSummarizationAccuracySemanticRobustnessEvaluateSample(
                model_input="Cake is so delicious, I really like cake. I want to open a bakery when I grow up.",
                target_output="I like cake.",
                original_model_output="Some model output.",
                perturbed_model_output_1="Some other model output.",
                perturbed_model_output_2="Some different model output.",
                sa_eval_score_original=[
                    EvalScore(name=METEOR_SCORE, value=1.0),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                    EvalScore(name=BERT_SCORE, value=0.5),
                ],
                sa_eval_score_perturbed_1=[
                    EvalScore(name=METEOR_SCORE, value=0.75),
                    EvalScore(name=ROUGE_SCORE, value=0.5),
                    EvalScore(name=BERT_SCORE, value=1.0),
                ],
                sa_eval_score_perturbed_2=[
                    EvalScore(name=METEOR_SCORE, value=0.5),
                    EvalScore(name=ROUGE_SCORE, value=0.2),
                    EvalScore(name=BERT_SCORE, value=0.5),
                ],
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=1.0),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                    EvalScore(name=BERT_SCORE, value=0.5),
                    EvalScore(name=DELTA_METEOR_SCORE, value=0.375),
                    EvalScore(name=DELTA_ROUGE_SCORE, value=0.65),
                    EvalScore(name=DELTA_BERT_SCORE, value=0.25),
                ],
                config=SummarizationAccuracySemanticRobustnessConfig(num_perturbations=2),
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.ray.get")
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.SummarizationAccuracyActor")
    def test_semantic_robustness_evaluate_sample_with_model_output(
        self, summarization_accuracy_actor_class, ray_get, test_case
    ):
        """
        GIVEN valid inputs with model_output
        WHEN SummarizationAccuracySemanticRobustness.evaluate_sample is called
        THEN correct List of EvalScores is returned
        """
        model = MagicMock()
        model.predict.side_effect = [
            (test_case.perturbed_model_output_1,),
            (test_case.perturbed_model_output_2,),
        ]

        evaluate_sample_invocation_results = [
            test_case.sa_eval_score_original,
            test_case.sa_eval_score_perturbed_1,
            test_case.sa_eval_score_perturbed_2,
        ]
        ray_get.side_effect = evaluate_sample_invocation_results
        summarization_accuracy_actor = MagicMock()
        summarization_accuracy_actor_class.return_value = summarization_accuracy_actor

        eval_algorithm = SummarizationAccuracySemanticRobustness(test_case.config)
        assert (
            eval_algorithm.evaluate_sample(
                model_input=test_case.model_input,
                model=model,
                target_output=test_case.target_output,
                model_output=test_case.original_model_output,
            )
            == test_case.expected_response
        )
        assert model.predict.call_count == 2

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSummarizationAccuracySemanticRobustnessEvaluateSampleInvalid(
                model_input="I like cake.",
                target_output="I like cake.",
                model=None,
                expected_error_message="Missing required input: model i.e. ModelRunner, for SummarizationAccuracySemanticRobustness "
                "evaluate_sample",
                config=SummarizationAccuracySemanticRobustnessConfig(num_perturbations=2),
            ),
            TestCaseSummarizationAccuracySemanticRobustnessEvaluateSampleInvalid(
                model_input=None,
                target_output="I like cake.",
                model=MagicMock(),
                expected_error_message="Missing required input: model_input, for SummarizationAccuracySemanticRobustness "
                "evaluate_sample",
                config=SummarizationAccuracySemanticRobustnessConfig(num_perturbations=2),
            ),
            TestCaseSummarizationAccuracySemanticRobustnessEvaluateSampleInvalid(
                model_input="I like cake.",
                target_output=None,
                model=MagicMock(),
                expected_error_message="Missing required input: target_output, for SummarizationAccuracySemanticRobustness "
                "evaluate_sample",
                config=SummarizationAccuracySemanticRobustnessConfig(num_perturbations=2),
            ),
        ],
    )
    def test_semantic_robustness_evaluate_sample_invalid_input(self, test_case):
        """
        GIVEN invalid inputs
        WHEN SummarizationAccuracySemanticRobustness.evaluate_sample is called
        THEN correct exception with proper message is raised
        """

        eval_algorithm = SummarizationAccuracySemanticRobustness(test_case.config, summ_acc_actor=MagicMock())
        with pytest.raises(EvalAlgorithmClientError, match=test_case.expected_error_message):
            eval_algorithm.evaluate_sample(test_case.model_input, test_case.target_output, test_case.model)

    class TestCaseSemanticRobustnessInvalidConfig(NamedTuple):
        rouge_type: str
        bertscore_model_type: str
        perturbation_type: str
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSemanticRobustnessInvalidConfig(
                rouge_type="rouge3",
                bertscore_model_type=None,
                perturbation_type="butter_finger",
                expected_error_message="Invalid rouge_type: rouge3 requested in SummarizationAccuracyConfig, "
                "please choose from acceptable values: ['rouge1', 'rouge2', 'rougeL']",
            ),
            TestCaseSemanticRobustnessInvalidConfig(
                rouge_type="rouge1",
                bertscore_model_type="distilbert-base-uncased",
                perturbation_type="butter_finger",
                expected_error_message="Invalid model_type_for_bertscore: distilbert-base-uncased requested in "
                "SummarizationAccuracyConfig, please choose from acceptable values: ["
                "'microsoft/deberta-xlarge-mnli', 'roberta-large-mnli']",
            ),
            TestCaseSemanticRobustnessInvalidConfig(
                rouge_type="rouge1",
                bertscore_model_type="distilbert-base-uncased",
                perturbation_type="my_perturb",
                expected_error_message="Invalid perturbation type 'my_perturb requested, please choose from "
                "acceptable values: dict_keys(['butter_finger', 'random_upper_case', 'whitespace_add_remove'])",
            ),
        ],
    )
    def test_semantic_robustness_invalid_config(self, test_case):
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            SummarizationAccuracySemanticRobustnessConfig(
                perturbation_type=test_case.perturbation_type,
                rouge_type=test_case.rouge_type,
                model_type_for_bertscore=test_case.bertscore_model_type,
            )

    class TestCaseSummarizationAccuracySemanticRobustnessEvaluate(NamedTuple):
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
            TestCaseSummarizationAccuracySemanticRobustnessEvaluate(
                input_dataset=DATASET_NO_CATEGORY,
                input_dataset_with_generated_model_output=DATASET_NO_CATEGORY_WITH_MODEL_OUTPUT,
                dataset_config=None,
                prompt_template=None,
                save_data=True,
                dataset_with_scores=DATASET_WITH_SCORES.drop_columns(cols=DatasetColumns.CATEGORY.value.name),
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_accuracy_semantic_robustness",
                        dataset_name=XSUM,
                        dataset_scores=[
                            EvalScore(name="rouge", value=0.0),
                            EvalScore(name="bertscore", value=0.0),
                            EvalScore(name="meteor", value=0.0),
                            EvalScore(name="delta_rouge", value=0.0),
                            EvalScore(name="delta_bertscore", value=0.0),
                            EvalScore(name="delta_meteor", value=0.0),
                        ],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[XSUM],
                        category_scores=None,
                        output_path="/tmp/eval_results/summarization_accuracy_semantic_robustness_xsum.jsonl",
                    ),
                    EvalOutput(
                        eval_name="summarization_accuracy_semantic_robustness",
                        dataset_name=GIGAWORD,
                        dataset_scores=[
                            EvalScore(name="rouge", value=0.0),
                            EvalScore(name="bertscore", value=0.0),
                            EvalScore(name="meteor", value=0.0),
                            EvalScore(name="delta_rouge", value=0.0),
                            EvalScore(name="delta_bertscore", value=0.0),
                            EvalScore(name="delta_meteor", value=0.0),
                        ],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[GIGAWORD],
                        category_scores=None,
                        output_path="/tmp/eval_results/summarization_accuracy_semantic_robustness_gigaword.jsonl",
                    ),
                    EvalOutput(
                        eval_name="summarization_accuracy_semantic_robustness",
                        dataset_name=GOV_REPORT,
                        dataset_scores=[
                            EvalScore(name="rouge", value=0.0),
                            EvalScore(name="bertscore", value=0.0),
                            EvalScore(name="meteor", value=0.0),
                            EvalScore(name="delta_rouge", value=0.0),
                            EvalScore(name="delta_bertscore", value=0.0),
                            EvalScore(name="delta_meteor", value=0.0),
                        ],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[GOV_REPORT],
                        category_scores=None,
                        output_path="/tmp/eval_results/summarization_accuracy_semantic_robustness_gov_report.jsonl",
                    ),
                ],
            ),
            # Built-in datasets evaluate for dataset with category
            TestCaseSummarizationAccuracySemanticRobustnessEvaluate(
                input_dataset=DATASET,
                input_dataset_with_generated_model_output=DATASET_WITH_MODEL_OUTPUT,
                dataset_config=None,
                prompt_template=None,
                save_data=True,
                dataset_with_scores=DATASET_WITH_SCORES,
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_accuracy_semantic_robustness",
                        dataset_name=XSUM,
                        dataset_scores=[
                            EvalScore(name="rouge", value=0.0),
                            EvalScore(name="bertscore", value=0.0),
                            EvalScore(name="meteor", value=0.0),
                            EvalScore(name="delta_rouge", value=0.0),
                            EvalScore(name="delta_bertscore", value=0.0),
                            EvalScore(name="delta_meteor", value=0.0),
                        ],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[XSUM],
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1",
                                scores=[
                                    EvalScore(name="rouge", value=0.0),
                                    EvalScore(name="bertscore", value=0.0),
                                    EvalScore(name="meteor", value=0.0),
                                    EvalScore(name="delta_rouge", value=0.0),
                                    EvalScore(name="delta_bertscore", value=0.0),
                                    EvalScore(name="delta_meteor", value=0.0),
                                ],
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[
                                    EvalScore(name="rouge", value=0.0),
                                    EvalScore(name="bertscore", value=0.0),
                                    EvalScore(name="meteor", value=0.0),
                                    EvalScore(name="delta_rouge", value=0.0),
                                    EvalScore(name="delta_bertscore", value=0.0),
                                    EvalScore(name="delta_meteor", value=0.0),
                                ],
                            ),
                        ],
                        output_path="/tmp/eval_results/summarization_accuracy_semantic_robustness_xsum.jsonl",
                    ),
                    EvalOutput(
                        eval_name="summarization_accuracy_semantic_robustness",
                        dataset_name=GIGAWORD,
                        dataset_scores=[
                            EvalScore(name="rouge", value=0.0),
                            EvalScore(name="bertscore", value=0.0),
                            EvalScore(name="meteor", value=0.0),
                            EvalScore(name="delta_rouge", value=0.0),
                            EvalScore(name="delta_bertscore", value=0.0),
                            EvalScore(name="delta_meteor", value=0.0),
                        ],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[GIGAWORD],
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1",
                                scores=[
                                    EvalScore(name="rouge", value=0.0),
                                    EvalScore(name="bertscore", value=0.0),
                                    EvalScore(name="meteor", value=0.0),
                                    EvalScore(name="delta_rouge", value=0.0),
                                    EvalScore(name="delta_bertscore", value=0.0),
                                    EvalScore(name="delta_meteor", value=0.0),
                                ],
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[
                                    EvalScore(name="rouge", value=0.0),
                                    EvalScore(name="bertscore", value=0.0),
                                    EvalScore(name="meteor", value=0.0),
                                    EvalScore(name="delta_rouge", value=0.0),
                                    EvalScore(name="delta_bertscore", value=0.0),
                                    EvalScore(name="delta_meteor", value=0.0),
                                ],
                            ),
                        ],
                        output_path="/tmp/eval_results/summarization_accuracy_semantic_robustness_gigaword.jsonl",
                    ),
                    EvalOutput(
                        eval_name="summarization_accuracy_semantic_robustness",
                        dataset_name=GOV_REPORT,
                        dataset_scores=[
                            EvalScore(name="rouge", value=0.0),
                            EvalScore(name="bertscore", value=0.0),
                            EvalScore(name="meteor", value=0.0),
                            EvalScore(name="delta_rouge", value=0.0),
                            EvalScore(name="delta_bertscore", value=0.0),
                            EvalScore(name="delta_meteor", value=0.0),
                        ],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[GOV_REPORT],
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1",
                                scores=[
                                    EvalScore(name="rouge", value=0.0),
                                    EvalScore(name="bertscore", value=0.0),
                                    EvalScore(name="meteor", value=0.0),
                                    EvalScore(name="delta_rouge", value=0.0),
                                    EvalScore(name="delta_bertscore", value=0.0),
                                    EvalScore(name="delta_meteor", value=0.0),
                                ],
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[
                                    EvalScore(name="rouge", value=0.0),
                                    EvalScore(name="bertscore", value=0.0),
                                    EvalScore(name="meteor", value=0.0),
                                    EvalScore(name="delta_rouge", value=0.0),
                                    EvalScore(name="delta_bertscore", value=0.0),
                                    EvalScore(name="delta_meteor", value=0.0),
                                ],
                            ),
                        ],
                        output_path="/tmp/eval_results/summarization_accuracy_semantic_robustness_gov_report.jsonl",
                    ),
                ],
            ),
            # Custom dataset evaluate with input prompt template
            TestCaseSummarizationAccuracySemanticRobustnessEvaluate(
                input_dataset=DATASET_NO_CATEGORY,
                input_dataset_with_generated_model_output=DATASET_NO_CATEGORY_WITH_MODEL_OUTPUT,
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
                dataset_with_scores=DATASET_WITH_SCORES.drop_columns(cols=DatasetColumns.CATEGORY.value.name),
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_accuracy_semantic_robustness",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[
                            EvalScore(name="rouge", value=0.0),
                            EvalScore(name="bertscore", value=0.0),
                            EvalScore(name="meteor", value=0.0),
                            EvalScore(name="delta_rouge", value=0.0),
                            EvalScore(name="delta_bertscore", value=0.0),
                            EvalScore(name="delta_meteor", value=0.0),
                        ],
                        prompt_template="$feature",
                        category_scores=None,
                        output_path="/tmp/eval_results/summarization_accuracy_semantic_robustness_my_custom_dataset.jsonl",
                    ),
                ],
            ),
            # Custom dataset evaluate without input prompt template
            TestCaseSummarizationAccuracySemanticRobustnessEvaluate(
                input_dataset=DATASET_NO_CATEGORY,
                input_dataset_with_generated_model_output=DATASET_NO_CATEGORY_WITH_MODEL_OUTPUT,
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
                        eval_name="summarization_accuracy_semantic_robustness",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[
                            EvalScore(name="rouge", value=0.0),
                            EvalScore(name="bertscore", value=0.0),
                            EvalScore(name="meteor", value=0.0),
                            EvalScore(name="delta_rouge", value=0.0),
                            EvalScore(name="delta_bertscore", value=0.0),
                            EvalScore(name="delta_meteor", value=0.0),
                        ],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=None,
                        output_path="/tmp/eval_results/summarization_accuracy_semantic_robustness_my_custom_dataset.jsonl",
                    ),
                ],
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.get_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.save_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.SummarizationAccuracyActor")
    @patch.object(SummarizationAccuracySemanticRobustness, "_SummarizationAccuracySemanticRobustness__add_scores")
    def test_semantic_robustness_evaluate(
        self,
        add_scores,
        summarization_accuracy_actor_class,
        save_dataset,
        get_dataset,
        test_case,
        config,
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset without model_outputs, an input ModelRunner
            and request to save records with scores
        WHEN SummarizationAccuracySemanticRobustness evaluate() method is called
        THEN correct EvalOutput is returned
        """
        add_scores.return_value = test_case.dataset_with_scores
        get_dataset.return_value = test_case.input_dataset
        summarization_accuracy_actor_class.return_value = MagicMock()

        eval_algorithm = SummarizationAccuracySemanticRobustness(config)
        actual_response = eval_algorithm.evaluate(
            model=MockModelRunner(),
            dataset_config=test_case.dataset_config,
            save=test_case.save_data,
            prompt_template=test_case.prompt_template,
        )
        assert save_dataset.called == test_case.save_data
        assert actual_response == test_case.expected_response

    class TestCaseSummarizationAccuracySemanticRobustnessEvaluateInvalid(NamedTuple):
        input_dataset: Dataset
        dataset_config: Optional[DataConfig]
        prompt_template: Optional[str]
        model_provided: bool
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSummarizationAccuracySemanticRobustnessEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY,
                dataset_config=None,
                prompt_template=None,
                model_provided=False,
                expected_error_message="Missing required input: model i.e. ModelRunner, for SummarizationAccuracySemanticRobustness "
                "evaluate method",
            ),
            TestCaseSummarizationAccuracySemanticRobustnessEvaluateInvalid(
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
            TestCaseSummarizationAccuracySemanticRobustnessEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(cols=[DatasetColumns.TARGET_OUTPUT.value.name]),
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
                expected_error_message="Missing required column: target_output, for evaluate() method",
            ),
        ],
    )
    @patch("fmeval.model_runners.model_runner.ModelRunner")
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.get_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy_semantic_robustness.SummarizationAccuracyActor")
    def test_semantic_robustness_evaluate_invalid_input(
        self,
        summarization_accuracy_actor_class,
        get_dataset,
        model,
        test_case,
        config,
    ):
        """
        GIVEN invalid inputs
        WHEN SummarizationAccuracySemanticRobustness evaluate is called
        THEN correct exception with proper message is raised
        """
        summarization_accuracy_actor_class.return_value = MagicMock()
        eval_algorithm = SummarizationAccuracySemanticRobustness(config)
        get_dataset.return_value = test_case.input_dataset
        if not test_case.model_provided:
            model = None
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            eval_algorithm.evaluate(
                model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template
            )
