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
    EvalOutput,
    EvalScore,
    CategoryScore,
    TRIVIA_QA,
    BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES,
    BOOLQ,
    NATURAL_QUESTIONS,
    DEFAULT_PROMPT_TEMPLATE,
)
from fmeval.eval_algorithms.qa_accuracy_semantic_robustness import (
    QAAccuracySemanticRobustnessConfig,
    QAAccuracySemanticRobustness,
    RANDOM_UPPER_CASE,
    WHITESPACE_ADD_REMOVE,
    BUTTER_FINGER,
    DELTA_F1_SCORE,
    DELTA_QUASI_EXACT_MATCH_SCORE,
    DELTA_EXACT_MATCH_SCORE,
    DELTA_PRECISION_OVER_WORDS,
    DELTA_RECALL_OVER_WORDS,
)
from fmeval.eval_algorithms.qa_accuracy import (
    F1_SCORE,
    QUASI_EXACT_MATCH_SCORE,
    EXACT_MATCH_SCORE,
    PRECISION_OVER_WORDS,
    RECALL_OVER_WORDS,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner

QA_DATASET_WITH_MODEL_OUTPUT = ray.data.from_items(
    [
        {
            DatasetColumns.MODEL_INPUT.value.name: "What is the capital of Italy?",
            DatasetColumns.TARGET_OUTPUT.value.name: "Rome",
            DatasetColumns.CATEGORY.value.name: "capitals",
            DatasetColumns.MODEL_OUTPUT.value.name: "Some model output.",
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "When did Argentina win the FIFA World Cup?",
            DatasetColumns.TARGET_OUTPUT.value.name: "1978<OR>1986<OR>2022.",
            DatasetColumns.CATEGORY.value.name: "sports",
            DatasetColumns.MODEL_OUTPUT.value.name: "Some model output.",
        },
    ]
)

QA_DATASET = QA_DATASET_WITH_MODEL_OUTPUT.drop_columns(cols=DatasetColumns.MODEL_OUTPUT.value.name)

QA_DATASET_WITHOUT_CATEGORY = QA_DATASET.drop_columns(cols=DatasetColumns.CATEGORY.value.name)

CATEGORY_SCORES = [
    CategoryScore(
        name="capitals",
        scores=[
            EvalScore(name=F1_SCORE, value=0.0),
            EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=PRECISION_OVER_WORDS, value=0.0),
            EvalScore(name=RECALL_OVER_WORDS, value=0.0),
            EvalScore(name=DELTA_F1_SCORE, value=0.0),
            EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=0.0),
            EvalScore(name=DELTA_RECALL_OVER_WORDS, value=0.0),
        ],
    ),
    CategoryScore(
        name="sports",
        scores=[
            EvalScore(name=F1_SCORE, value=0.0),
            EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=PRECISION_OVER_WORDS, value=0.0),
            EvalScore(name=RECALL_OVER_WORDS, value=0.0),
            EvalScore(name=DELTA_F1_SCORE, value=0.0),
            EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=0.0),
            EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=0.0),
            EvalScore(name=DELTA_RECALL_OVER_WORDS, value=0.0),
        ],
    ),
]


class ConstantModel(ModelRunner):
    def __init__(self):
        super().__init__('{"data": $prompt}', output="output")

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        return "Some model output.", None


class TestQAAccuracySemanticRobustness:
    @fixture(scope="module")
    def config(self) -> QAAccuracySemanticRobustnessConfig:
        return QAAccuracySemanticRobustnessConfig(target_output_delimiter="<OR>", num_perturbations=2)

    class TestCaseQAAccuracySemanticRobustnessInvalidConfig(NamedTuple):
        target_output_delimiter: Optional[str]
        perturbation_type: str
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracySemanticRobustnessInvalidConfig(
                target_output_delimiter="<OR>",
                perturbation_type="my_perturb",
                expected_error_message="Invalid perturbation type 'my_perturb requested, please choose from "
                "acceptable values: dict_keys(['butter_finger', 'random_upper_case', 'whitespace_add_remove'])",
            ),
            TestCaseQAAccuracySemanticRobustnessInvalidConfig(
                target_output_delimiter="",
                perturbation_type="butter_finger",
                expected_error_message="Empty target_output_delimiter is provided. Please either provide a non-empty string, or set it to None",
            ),
        ],
    )
    def test_qa_accuracy_semantic_robustness_invalid_config(self, test_case):
        """
        GIVEN invalid configs
        WHEN QAAccuracySemanticRobustnessConfig is initialized
        THEN correct exception with proper message is raised
        """
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            QAAccuracySemanticRobustnessConfig(
                target_output_delimiter=test_case.target_output_delimiter,
                perturbation_type=test_case.perturbation_type,
            )

    class TestCaseQAAccuracySemanticRobustnessEvaluateSample(NamedTuple):
        model_input: str
        original_model_output: str
        perturbed_model_output_1: str
        perturbed_model_output_2: str
        target_output: str
        expected_response: List[EvalScore]
        config: QAAccuracySemanticRobustnessConfig

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracySemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="london!",
                perturbed_model_output_1="Some model output.",
                perturbed_model_output_2="Some model output.",
                target_output="London",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=1.0),
                    EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=PRECISION_OVER_WORDS, value=1.0),
                    EvalScore(name=RECALL_OVER_WORDS, value=1.0),
                    EvalScore(name=DELTA_F1_SCORE, value=1.0),
                    EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=1.0),
                    EvalScore(name=DELTA_RECALL_OVER_WORDS, value=1.0),
                ],
                config=QAAccuracySemanticRobustnessConfig(target_output_delimiter="<OR>", num_perturbations=2),
            ),
            TestCaseQAAccuracySemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="london!",
                perturbed_model_output_1="london",
                perturbed_model_output_2="paris",
                target_output="London",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=1.0),
                    EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=PRECISION_OVER_WORDS, value=1),
                    EvalScore(name=RECALL_OVER_WORDS, value=1),
                    EvalScore(name=DELTA_F1_SCORE, value=0.5),
                    EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=0.5),
                    EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=0.5),
                    EvalScore(name=DELTA_RECALL_OVER_WORDS, value=0.5),
                ],
                config=QAAccuracySemanticRobustnessConfig(
                    target_output_delimiter="<OR>", num_perturbations=2, perturbation_type=BUTTER_FINGER
                ),
            ),
            TestCaseQAAccuracySemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="London is the capital",
                perturbed_model_output_1="Another model output.",
                perturbed_model_output_2="Some model output.",
                target_output="London",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=0.5),
                    EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=PRECISION_OVER_WORDS, value=1 / 3),
                    EvalScore(name=RECALL_OVER_WORDS, value=1),
                    EvalScore(name=DELTA_F1_SCORE, value=0.5),
                    EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=1 / 3),
                    EvalScore(name=DELTA_RECALL_OVER_WORDS, value=1.0),
                ],
                config=QAAccuracySemanticRobustnessConfig(
                    target_output_delimiter="<OR>", num_perturbations=2, perturbation_type=RANDOM_UPPER_CASE
                ),
            ),
            TestCaseQAAccuracySemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="London",
                perturbed_model_output_1="Another model output.",
                perturbed_model_output_2="Another model output.",
                target_output="London",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=1.0),
                    EvalScore(name=EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=PRECISION_OVER_WORDS, value=1.0),
                    EvalScore(name=RECALL_OVER_WORDS, value=1.0),
                    EvalScore(name=DELTA_F1_SCORE, value=1.0),
                    EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=1.0),
                    EvalScore(name=DELTA_RECALL_OVER_WORDS, value=1.0),
                ],
                config=QAAccuracySemanticRobustnessConfig(
                    target_output_delimiter="<OR>", num_perturbations=2, perturbation_type=WHITESPACE_ADD_REMOVE
                ),
            ),
        ],
    )
    def test_qa_accuracy_semantic_robustness_evaluate_sample(self, test_case):
        """
        GIVEN valid inputs
        WHEN QAAccuracySemanticRobustness.evaluate_sample is called
        THEN correct List of EvalScores is returned
        """
        model = MagicMock()
        model.predict.side_effect = [
            (test_case.original_model_output,),
            (test_case.perturbed_model_output_1,),
            (test_case.perturbed_model_output_2,),
        ]

        eval_algorithm = QAAccuracySemanticRobustness(test_case.config)
        assert (
            eval_algorithm.evaluate_sample(
                model_input=test_case.model_input,
                model=model,
                target_output=test_case.target_output,
            )
            == test_case.expected_response
        )
        assert model.predict.call_count == 3

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracySemanticRobustnessEvaluateSample(
                model_input="What is the capital of England?",
                original_model_output="london!",
                perturbed_model_output_1="Some model output.",
                perturbed_model_output_2="Some model output.",
                target_output="London",
                expected_response=[
                    EvalScore(name=F1_SCORE, value=1.0),
                    EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=PRECISION_OVER_WORDS, value=1.0),
                    EvalScore(name=RECALL_OVER_WORDS, value=1.0),
                    EvalScore(name=DELTA_F1_SCORE, value=1.0),
                    EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                    EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=1.0),
                    EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=1.0),
                    EvalScore(name=DELTA_RECALL_OVER_WORDS, value=1.0),
                ],
                config=QAAccuracySemanticRobustnessConfig(target_output_delimiter="<OR>", num_perturbations=2),
            ),
        ],
    )
    def test_qa_accuracy_semantic_robustness_evaluate_sample_with_model_output(self, test_case):
        """
        GIVEN valid inputs with model_output
        WHEN QAAccuracySemanticRobustness.evaluate_sample is called
        THEN correct List of EvalScores is returned
        """
        model = MagicMock()
        model.predict.side_effect = [
            (test_case.perturbed_model_output_1,),
            (test_case.perturbed_model_output_2,),
        ]

        eval_algorithm = QAAccuracySemanticRobustness(test_case.config)
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

    class TestCaseQAAccuracySemanticRobustnessEvaluateSampleInvalid(NamedTuple):
        model_input: str
        target_output: str
        model: ModelRunner
        expected_error_message: List[EvalScore]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracySemanticRobustnessEvaluateSampleInvalid(
                model_input="What is the capital of England?",
                target_output="London",
                model=None,
                expected_error_message="Missing required input: model i.e. ModelRunner, for QAAccuracySemanticRobustness "
                "evaluate_sample",
            ),
            TestCaseQAAccuracySemanticRobustnessEvaluateSampleInvalid(
                model_input=None,
                target_output="London",
                model=MagicMock(),
                expected_error_message="Missing required input: model_input, for QAAccuracySemanticRobustness "
                "evaluate_sample",
            ),
            TestCaseQAAccuracySemanticRobustnessEvaluateSampleInvalid(
                model_input="What is the capital of England?",
                target_output=None,
                model=MagicMock(),
                expected_error_message="Missing required input: target_output, for QAAccuracySemanticRobustness "
                "evaluate_sample",
            ),
        ],
    )
    def test_qa_accuracy_semantic_robustness_evaluate_sample_invalid_input(self, test_case, config):
        """
        GIVEN invalid inputs
        WHEN QAAccuracySemanticRobustness.evaluate_sample is called
        THEN correct exception with proper message is raised
        """
        eval_algorithm = QAAccuracySemanticRobustness(config)
        with pytest.raises(EvalAlgorithmClientError, match=test_case.expected_error_message):
            eval_algorithm.evaluate_sample(test_case.model_input, test_case.model, test_case.target_output)

    class TestCaseQAAccuracySemanticRobustnessEvaluate(NamedTuple):
        input_dataset: Dataset
        input_dataset_with_generated_model_output: Dataset
        prompt_template: Optional[str]
        dataset_config: Optional[DataConfig]
        expected_response: List[EvalOutput]
        save_data: bool

    @pytest.mark.parametrize(
        "test_case",
        [
            # Built-in datasets evaluate for dataset without category
            TestCaseQAAccuracySemanticRobustnessEvaluate(
                input_dataset=QA_DATASET_WITHOUT_CATEGORY,
                input_dataset_with_generated_model_output=QA_DATASET_WITH_MODEL_OUTPUT.drop_columns(
                    cols=DatasetColumns.CATEGORY.value.name
                ),
                dataset_config=None,
                prompt_template=None,
                save_data=True,
                expected_response=[
                    EvalOutput(
                        eval_name="qa_accuracy_semantic_robustness",
                        dataset_name=BOOLQ,
                        dataset_scores=[
                            EvalScore(name=F1_SCORE, value=0.0),
                            EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=RECALL_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_F1_SCORE, value=0.0),
                            EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_RECALL_OVER_WORDS, value=0.0),
                        ],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[BOOLQ],
                        category_scores=None,
                        output_path="/tmp/eval_results/qa_accuracy_semantic_robustness_boolq.jsonl",
                    ),
                    EvalOutput(
                        eval_name="qa_accuracy_semantic_robustness",
                        dataset_name=TRIVIA_QA,
                        dataset_scores=[
                            EvalScore(name=F1_SCORE, value=0.0),
                            EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=RECALL_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_F1_SCORE, value=0.0),
                            EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_RECALL_OVER_WORDS, value=0.0),
                        ],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[TRIVIA_QA],
                        category_scores=None,
                        output_path="/tmp/eval_results/qa_accuracy_semantic_robustness_trivia_qa.jsonl",
                    ),
                    EvalOutput(
                        eval_name="qa_accuracy_semantic_robustness",
                        dataset_name=NATURAL_QUESTIONS,
                        dataset_scores=[
                            EvalScore(name=F1_SCORE, value=0.0),
                            EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=RECALL_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_F1_SCORE, value=0.0),
                            EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_RECALL_OVER_WORDS, value=0.0),
                        ],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[NATURAL_QUESTIONS],
                        category_scores=None,
                        output_path="/tmp/eval_results/qa_accuracy_semantic_robustness_natural_questions.jsonl",
                    ),
                ],
            ),
            # Built-in datasets evaluate for dataset with category
            TestCaseQAAccuracySemanticRobustnessEvaluate(
                input_dataset=QA_DATASET_WITH_MODEL_OUTPUT,
                input_dataset_with_generated_model_output=QA_DATASET,
                dataset_config=None,
                prompt_template=None,
                save_data=True,
                expected_response=[
                    EvalOutput(
                        eval_name="qa_accuracy_semantic_robustness",
                        dataset_name=BOOLQ,
                        dataset_scores=[
                            EvalScore(name=F1_SCORE, value=0.0),
                            EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=RECALL_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_F1_SCORE, value=0.0),
                            EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_RECALL_OVER_WORDS, value=0.0),
                        ],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[BOOLQ],
                        category_scores=CATEGORY_SCORES,
                        output_path="/tmp/eval_results/qa_accuracy_semantic_robustness_boolq.jsonl",
                    ),
                    EvalOutput(
                        eval_name="qa_accuracy_semantic_robustness",
                        dataset_name=TRIVIA_QA,
                        dataset_scores=[
                            EvalScore(name=F1_SCORE, value=0.0),
                            EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=RECALL_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_F1_SCORE, value=0.0),
                            EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_RECALL_OVER_WORDS, value=0.0),
                        ],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[TRIVIA_QA],
                        category_scores=CATEGORY_SCORES,
                        output_path="/tmp/eval_results/qa_accuracy_semantic_robustness_trivia_qa.jsonl",
                    ),
                    EvalOutput(
                        eval_name="qa_accuracy_semantic_robustness",
                        dataset_name=NATURAL_QUESTIONS,
                        dataset_scores=[
                            EvalScore(name=F1_SCORE, value=0.0),
                            EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=RECALL_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_F1_SCORE, value=0.0),
                            EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_RECALL_OVER_WORDS, value=0.0),
                        ],
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[NATURAL_QUESTIONS],
                        category_scores=CATEGORY_SCORES,
                        output_path="/tmp/eval_results/qa_accuracy_semantic_robustness_natural_questions.jsonl",
                    ),
                ],
            ),
            # Custom dataset evaluate, with input prompt template
            TestCaseQAAccuracySemanticRobustnessEvaluate(
                input_dataset=QA_DATASET_WITHOUT_CATEGORY,
                input_dataset_with_generated_model_output=QA_DATASET_WITH_MODEL_OUTPUT.drop_columns(
                    cols=DatasetColumns.CATEGORY.value.name
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
                save_data=False,
                expected_response=[
                    EvalOutput(
                        eval_name="qa_accuracy_semantic_robustness",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[
                            EvalScore(name=F1_SCORE, value=0.0),
                            EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=RECALL_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_F1_SCORE, value=0.0),
                            EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_RECALL_OVER_WORDS, value=0.0),
                        ],
                        prompt_template="$feature",
                        category_scores=None,
                        output_path="/tmp/eval_results/qa_accuracy_my_custom_dataset.jsonl",
                    ),
                ],
            ),
            # Custom dataset evaluate, without input prompt template
            TestCaseQAAccuracySemanticRobustnessEvaluate(
                input_dataset=QA_DATASET_WITHOUT_CATEGORY,
                input_dataset_with_generated_model_output=QA_DATASET_WITH_MODEL_OUTPUT.drop_columns(
                    cols=DatasetColumns.CATEGORY.value.name
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
                save_data=False,
                expected_response=[
                    EvalOutput(
                        eval_name="qa_accuracy_semantic_robustness",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[
                            EvalScore(name=F1_SCORE, value=0.0),
                            EvalScore(name=EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=RECALL_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_F1_SCORE, value=0.0),
                            EvalScore(name=DELTA_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_QUASI_EXACT_MATCH_SCORE, value=0.0),
                            EvalScore(name=DELTA_PRECISION_OVER_WORDS, value=0.0),
                            EvalScore(name=DELTA_RECALL_OVER_WORDS, value=0.0),
                        ],
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        category_scores=None,
                        output_path="/tmp/eval_results/qa_accuracy_my_custom_dataset.jsonl",
                    ),
                ],
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.qa_accuracy_semantic_robustness.get_dataset")
    @patch("fmeval.eval_algorithms.qa_accuracy_semantic_robustness.save_dataset")
    @patch("fmeval.eval_algorithms.general_semantic_robustness.generate_model_predict_response_for_dataset")
    @patch("fmeval.eval_algorithms.qa_accuracy_semantic_robustness.QAAccuracy")
    def test_qa_accuracy_semantic_robustness_evaluate(
        self, qa_accuracy, generate_model_predict_response_for_dataset, save_dataset, get_dataset, test_case, config
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset without model_outputs, an input ModelRunner
            and request to save records with scores
        WHEN QAAccuracySemanticRobustness evaluate() method is called
        THEN correct EvalOutput is returned
        """
        get_dataset.return_value = test_case.input_dataset
        generate_model_predict_response_for_dataset.return_value = test_case.input_dataset_with_generated_model_output
        qa_accuracy.return_value = MagicMock()

        eval_algorithm = QAAccuracySemanticRobustness(config)
        actual_response = eval_algorithm.evaluate(
            model=ConstantModel(),
            dataset_config=test_case.dataset_config,
            save=test_case.save_data,
            prompt_template=test_case.prompt_template,
        )
        assert save_dataset.called == test_case.save_data
        assert actual_response == test_case.expected_response

    class TestCaseQAAccuracySemanticRobustnessEvaluateInvalid(NamedTuple):
        input_dataset: Dataset
        dataset_config: Optional[DataConfig]
        prompt_template: Optional[str]
        model_provided: bool
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseQAAccuracySemanticRobustnessEvaluateInvalid(
                input_dataset=QA_DATASET_WITHOUT_CATEGORY,
                dataset_config=None,
                prompt_template=None,
                model_provided=False,
                expected_error_message="Missing required input: model i.e. ModelRunner, for QAAccuracySemanticRobustness "
                "evaluate",
            ),
            TestCaseQAAccuracySemanticRobustnessEvaluateInvalid(
                input_dataset=QA_DATASET_WITHOUT_CATEGORY.drop_columns(cols=[DatasetColumns.MODEL_INPUT.value.name]),
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
                expected_error_message="Missing required column: model_input, for evaluate",
            ),
            TestCaseQAAccuracySemanticRobustnessEvaluateInvalid(
                input_dataset=QA_DATASET_WITHOUT_CATEGORY.drop_columns(cols=[DatasetColumns.TARGET_OUTPUT.value.name]),
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
                expected_error_message="Missing required column: target_output, for evaluate",
            ),
        ],
    )
    @patch("fmeval.model_runners.model_runner.ModelRunner")
    @patch("fmeval.eval_algorithms.qa_accuracy_semantic_robustness.get_dataset")
    @patch("fmeval.eval_algorithms.qa_accuracy_semantic_robustness.QAAccuracy")
    def test_qa_accuracy_semantic_robustness_evaluate_invalid_input(
        self,
        qa_accuracy,
        get_dataset,
        model,
        test_case,
        config,
    ):
        """
        GIVEN invalid inputs
        WHEN QAAccuracySemanticRobustness evaluate is called
        THEN correct exception with proper message is raised
        """
        qa_accuracy.return_value = MagicMock()
        eval_algorithm = QAAccuracySemanticRobustness(config)
        get_dataset.return_value = test_case.input_dataset
        if not test_case.model_provided:
            model = None
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            eval_algorithm.evaluate(
                model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template
            )
