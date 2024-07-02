from typing import List, NamedTuple, Optional
from unittest.mock import patch, Mock

import pytest
import ray
from datasets import Dataset

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms import EvalAlgorithm, EvalScore, EvalOutput
from fmeval.eval_algorithms.answer_relevance import AnswerRelevance, AnswerRelevanceScore

DATASET_WITH_CONTEXT = ray.data.from_items(
    [
        {
            DatasetColumns.MODEL_INPUT.value.name: "When was the first super bowl?",
            DatasetColumns.MODEL_OUTPUT.value.name: "The first superbowl was held on Jan 15, 1967.",
        },
    ]
)

DATASET_SCORES = [
    EvalScore(name=EvalAlgorithm.ANSWER_RELEVANCE.value, value=0.87),
]

QUESTION_EMBEDDINGS = [0.00834976, 0.01356759, -0.03592299, 0.0121677, 0.0183047, -0.02220628]
GEN_QUESTION_EMBEDDINGS = [-0.00489958, 0.00642186, -0.0452478, 0.01276442, -0.00164369, -0.02824315]


class TestAnswerRelevance:
    class TestCaseAnswerRelevanceEvaluateSample(NamedTuple):
        model_input: str
        model_output: str
        generated_questions: str
        question_vector: List[float]
        gen_question_vector: List[float]
        expected_score: List[EvalScore]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseAnswerRelevanceEvaluateSample(
                model_input="When was the first super bowl?",
                model_output="The first superbowl was held on Jan 15, 1967.",
                generated_questions="Question: What was the date of the first Super Bowl?",
                question_vector=QUESTION_EMBEDDINGS,
                gen_question_vector=GEN_QUESTION_EMBEDDINGS,
                expected_score=[EvalScore(name=EvalAlgorithm.ANSWER_RELEVANCE.value, value=0.87)],
            ),
        ],
    )
    @patch("fmeval.model_runners.sm_jumpstart_model_runner.is_text_embedding_js_model", return_value=True)
    def test_answer_relevance_evaluate_sample(self, is_text_embedding_js_model, test_case):
        """
        GIVEN valid inputs
        WHEN AnswerRelevance.evaluate_sample is called
        THEN correct EvalScore is returned
        """
        # GIVEN
        mock_judge_model_runner = Mock()
        mock_judge_model_runner.predict.side_effect = [
            (test_case.generated_questions, None),
        ]
        mock_embedding_model_runner = Mock()
        mock_embedding_model_runner.predict.side_effect = [test_case.question_vector, test_case.gen_question_vector]
        # WHEN
        eval_algorithm = AnswerRelevance()
        actual_score = eval_algorithm.evaluate_sample(
            model_input=test_case.model_input,
            model_output=test_case.model_output,
            judge_model=mock_judge_model_runner,
            embeddings_model=mock_embedding_model_runner,
        )
        # THEN
        assert test_case.expected_score == actual_score

    class TestCaseAnswerRelevanceEvaluate(NamedTuple):
        input_dataset: Dataset
        dataset_config: Optional[DataConfig]
        question_vector: List[float]
        gen_question_vector: List[float]
        expected_response: List[EvalOutput]
        generated_questions: str

    @pytest.mark.parametrize(
        "test_case",
        [
            #
            TestCaseAnswerRelevanceEvaluate(
                input_dataset=DATASET_WITH_CONTEXT,
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                question_vector=QUESTION_EMBEDDINGS,
                gen_question_vector=GEN_QUESTION_EMBEDDINGS,
                generated_questions="Question: What was the date of the first Super Bowl?",
                expected_response=[
                    EvalOutput(
                        eval_name="answer_relevance",
                        dataset_name="my_custom_dataset",
                        prompt_template=None,
                        dataset_scores=DATASET_SCORES,
                        output_path="/tmp/eval_results/answer_relevance_my_custom_dataset.jsonl",
                    )
                ],
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.answer_relevance.get_dataset")
    def test_answer_relevance_evaluate(self, mock_get_dataset, test_case):
        """
        GIVEN datasets.
        WHEN the Answer Relevance evaluate method is called.
        THEN the correct EvalOutput is returned.

        Note: this is basically an integration test rather than a unit test like `test_evaluate` above.
        It uses a special toy dataset where we are able to compute the exact expected scores by hand.
        The purpose of this test is really to ensure that the correct scores are being generated.
        """
        # GIVEN
        mock_judge_model_runner = Mock()
        mock_judge_model_runner.predict.side_effect = [
            (test_case.generated_questions, None),
        ]
        mock_embedding_model_runner = Mock()
        mock_embedding_model_runner.predict.side_effect = [test_case.question_vector, test_case.gen_question_vector]
        mock_get_dataset.return_value = test_case.input_dataset
        eval_algorithm = AnswerRelevance()
        # WHEN
        actual_response = eval_algorithm.evaluate(
            judge_model=mock_judge_model_runner,
            embeddings_model=mock_embedding_model_runner,
            dataset_config=test_case.dataset_config,
            save=True,
        )
        # THEN
        assert actual_response == test_case.expected_response

    class TestCaseAnswerRelevanceScore(NamedTuple):
        question: str
        gen_questions_str: str
        question_vector: List[float]
        gen_question_vectors: List[List[float]]
        expected_score: float

    @pytest.mark.parametrize(
        "test_case",
        [
            # calculate score based embedding
            TestCaseAnswerRelevanceScore(
                question="When was the first super bowl?",
                gen_questions_str="Question: What was the date of the first Super Bowl?",
                question_vector=QUESTION_EMBEDDINGS,
                gen_question_vectors=[GEN_QUESTION_EMBEDDINGS],
                expected_score=0.87,
            ),
            # calculate score when multiple questions being generated
            TestCaseAnswerRelevanceScore(
                question="When was the first super bowl?",
                gen_questions_str="Question: What was the date of the first Super Bowl?\nQuestion: When was the first super bowl?",
                question_vector=QUESTION_EMBEDDINGS,
                gen_question_vectors=[GEN_QUESTION_EMBEDDINGS, QUESTION_EMBEDDINGS],
                # mean value of 0.87 and 1.0
                expected_score=0.935,
            ),
            # no questions being generated
            TestCaseAnswerRelevanceScore(
                question="When was the first super bowl?",
                gen_questions_str="",
                question_vector=QUESTION_EMBEDDINGS,
                gen_question_vectors=[GEN_QUESTION_EMBEDDINGS],
                expected_score=0,
            ),
        ],
    )
    def test_answer_relevance_score_transform(self, test_case):
        """
        GIVEN a AnswerRelevanceScore instance.
        WHEN its __call__ method is invoked.
        THEN the correct output is returned.
        """
        mock_embedding_model_runner = Mock()
        mock_embedding_model_runner.predict.side_effect = [test_case.question_vector] + test_case.gen_question_vectors
        get_scores = AnswerRelevanceScore(embeddings_model=mock_embedding_model_runner)
        sample = {
            "model_output": test_case.question,
            "gen_questions": test_case.gen_questions_str,
        }
        result = get_scores(sample)
        assert result[EvalAlgorithm.ANSWER_RELEVANCE.value] == pytest.approx(test_case.expected_score, rel=1e-2)
