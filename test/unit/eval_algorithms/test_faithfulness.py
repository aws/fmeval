from typing import List, NamedTuple, Optional, Dict
from unittest.mock import patch, Mock

import pytest
import ray
from ray.data import Dataset

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON,
    MEAN,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms import EvalAlgorithm, EvalOutput, EvalScore
from fmeval.eval_algorithms.faithfulness import (
    Faithfulness,
    FaithfulnessScore,
    GetStatements,
    LONG_FORM_ANSWER_PROMPT,
    NLI_STATEMENTS_MESSAGE,
)

SAMPLE_MODEL_INPUT = "Where and when was Einstein born?"
SAMPLE_MODEL_OUTPUT = "Einstein was born in Germany on 20th March 1879."
SAMPLE_TARGET_CONTEXT = "Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time"
SAMPLE_STATEMENTS_OUTPUT = "Here are the statements created from the given answer:\nStatement: Einstein was born in Germany.\nStatement: Einstein was born on 20th March 1879."
SAMPLE_VERDICTS_OUTPUT = 'here are the verdicts for the statements:\n1. statement: einstein was born in germany.\nexplanation: the context states that einstein was "a german-born theoretical physicist". this supports that he was born in germany.\nverdict: yes\n2. statement: einstein was born on 20th march 1879.  \nexplanation: the context states that einstein was "born 14 march 1879". this contradicts the statement that he was born on 20th march 1879.\nverdict: no\nfinal verdicts in order:\nyes. no.'

NO_STATEMENTS_OUTPUT = "Can't find statements. Here are the statements created from the given answer"
NO_VERDICTS_OUTPUT = "No verdicts as no statements can be found."

DATASET_WITH_CONTEXT = ray.data.from_items(
    [
        {
            DatasetColumns.MODEL_INPUT.value.name: SAMPLE_MODEL_INPUT,
            DatasetColumns.MODEL_OUTPUT.value.name: SAMPLE_MODEL_OUTPUT,
            DatasetColumns.TARGET_CONTEXT.value.name: SAMPLE_TARGET_CONTEXT,
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "random question",
            DatasetColumns.MODEL_OUTPUT.value.name: "random answer",
            DatasetColumns.TARGET_CONTEXT.value.name: "random context",
        },
    ]
)

DATASET_SCORES = [
    EvalScore(name=EvalAlgorithm.FAITHFULNESS.value, value=1 / 2),
]


class TestFaithfulness:
    class TestCaseFaithfulnessEvaluateSample(NamedTuple):
        model_input: str
        model_output: str
        target_context: str
        statements_output: List[str]
        verdicts_output: List[str]
        expected_score: List[EvalScore]

    @pytest.mark.parametrize(
        "test_case",
        [
            # successful case
            TestCaseFaithfulnessEvaluateSample(
                model_input=SAMPLE_MODEL_INPUT,
                model_output=SAMPLE_MODEL_OUTPUT,
                target_context=SAMPLE_TARGET_CONTEXT,
                statements_output=SAMPLE_STATEMENTS_OUTPUT,
                verdicts_output=SAMPLE_VERDICTS_OUTPUT,
                expected_score=[EvalScore(name=EvalAlgorithm.FAITHFULNESS.value, value=1 / 2)],
            ),
            # No statements get from judge model
            TestCaseFaithfulnessEvaluateSample(
                model_input=SAMPLE_MODEL_INPUT,
                model_output=SAMPLE_MODEL_OUTPUT,
                target_context=SAMPLE_TARGET_CONTEXT,
                statements_output=NO_STATEMENTS_OUTPUT,
                verdicts_output=NO_VERDICTS_OUTPUT,
                expected_score=[
                    EvalScore(
                        name=EvalAlgorithm.FAITHFULNESS.value,
                        value=None,
                        error="No statements were generated from the answer.",
                    )
                ],
            ),
        ],
    )
    def test_faithfulness_evaluate_sample(self, test_case):
        """
        GIVEN valid inputs
        WHEN Faithfulness.evaluate_sample is called
        THEN correct EvalScore is returned
        """
        # GIVEN
        mock_model_runner = Mock()
        mock_model_runner.predict.side_effect = [
            (test_case.statements_output, None),
            (test_case.verdicts_output, None),
        ]
        # WHEN
        eval_algorithm = Faithfulness()
        actual_score = eval_algorithm.evaluate_sample(
            model_input=test_case.model_input,
            model_output=test_case.model_output,
            target_context=test_case.target_context,
            judge_model=mock_model_runner,
        )
        # THEN
        assert test_case.expected_score == actual_score

    @patch("fmeval.eval_algorithms.faithfulness.get_eval_results_path")
    @patch("fmeval.eval_algorithms.faithfulness.evaluate_dataset")
    @patch("fmeval.eval_algorithms.faithfulness.Faithfulness._build_pipeline")
    @patch("fmeval.eval_algorithms.faithfulness.get_dataset")
    @patch("fmeval.eval_algorithms.faithfulness.get_dataset_configs")
    def test_evaluate(
        self,
        mock_get_dataset_configs,
        mock_get_dataset,
        mock_build_pipeline,
        mock_evaluate_dataset,
        mock_get_results_path,
    ):
        """
        GIVEN a Faithfulness instance.
        WHEN its evaluate method is called with valid arguments.
        THEN `evaluate_dataset` is called with the correct arguments.
        """
        dataset_config = Mock()
        dataset_config.dataset_name = "my_custom_dataset"
        mock_get_dataset_configs.return_value = [dataset_config]

        mock_dataset = Mock()
        # So that validate_dataset does not error
        mock_dataset.columns = Mock(
            return_value=[
                DatasetColumns.MODEL_INPUT.value.name,
                DatasetColumns.MODEL_OUTPUT.value.name,
                DatasetColumns.TARGET_CONTEXT.value.name,
            ]
        )
        mock_get_dataset.return_value = mock_dataset

        mock_get_results_path.return_value = "/path/to/results"
        model_runner = Mock()

        mock_build_pipeline.return_value = Mock()
        # WHEN
        faithfulness_algo = Faithfulness()
        output = faithfulness_algo.evaluate(
            judge_model=model_runner,
            dataset_config=dataset_config,
            num_records=162,
            save=True,
        )
        # THEN
        mock_build_pipeline.assert_called_with(
            model_runner,
            LONG_FORM_ANSWER_PROMPT,
            NLI_STATEMENTS_MESSAGE,
        )
        mock_evaluate_dataset.assert_called_once_with(
            dataset=mock_dataset,
            pipeline=mock_build_pipeline.return_value,
            dataset_name=dataset_config.dataset_name,
            eval_name=faithfulness_algo.eval_name,
            metric_names=["faithfulness"],
            eval_results_path="/path/to/results",
            agg_method=MEAN,
            save=True,
            save_strategy=None,
        )
        assert output == [mock_evaluate_dataset.return_value]

    class TestCaseFaithfulnessEvaluate(NamedTuple):
        input_dataset: Dataset
        dataset_config: Optional[DataConfig]
        statements_output: List[str]
        verdicts_output: List[str]
        expected_response: List[EvalOutput]

    @pytest.mark.parametrize(
        "test_case",
        [
            #
            TestCaseFaithfulnessEvaluate(
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
                statements_output=SAMPLE_STATEMENTS_OUTPUT,
                verdicts_output=SAMPLE_VERDICTS_OUTPUT,
                expected_response=[
                    EvalOutput(
                        eval_name="faithfulness",
                        dataset_name="my_custom_dataset",
                        prompt_template=None,
                        dataset_scores=DATASET_SCORES,
                        output_path="/tmp/eval_results/faithfulness_my_custom_dataset.jsonl",
                    )
                ],
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.faithfulness.get_dataset")
    def test_faithfulness_evaluate(self, mock_get_dataset, test_case):
        """
        GIVEN datasets.
        WHEN the Faithfulness evaluate method is called.
        THEN the correct EvalOutput is returned.

        Note: this is basically an integration test rather than a unit test like `test_evaluate` above.
        It uses a special toy dataset where we are able to compute the exact expected scores by hand.
        The purpose of this test is really to ensure that the correct scores are being generated.
        """
        # GIVEN
        mock_model_runner = Mock()

        def predict_side_effect(prompt):
            if (
                "Your task is to rewrite the answer into one or more simple and coherent statements"
                and "random question" in prompt
            ):
                return NO_STATEMENTS_OUTPUT, None
            if "provide your final verdict" and "random context" in prompt:
                return NO_VERDICTS_OUTPUT, None
            if "Your task is to rewrite the answer into one or more simple and coherent statements" in prompt:
                return test_case.statements_output, None
            if "provide your final verdict" in prompt:
                return test_case.verdicts_output, None

        mock_model_runner.predict.side_effect = predict_side_effect
        mock_get_dataset.return_value = test_case.input_dataset
        eval_algorithm = Faithfulness()
        # WHEN
        actual_response = eval_algorithm.evaluate(
            judge_model=mock_model_runner,
            dataset_config=test_case.dataset_config,
            save=True,
        )
        # THEN
        assert actual_response == test_case.expected_response

    class TestCaseFaithfulnessScore(NamedTuple):
        raw_verdicts: str
        statements: str
        expected_score: float

    @pytest.mark.parametrize(
        "test_case",
        [
            # calculate score based on count of verdicts from raw verdicts string
            TestCaseFaithfulnessScore(
                raw_verdicts="verdict: yes\nverdict: no",
                statements="Statement: statement1\nStatement: statement2",
                expected_score=0.5,
            ),
            # unknown raw verdicts string
            TestCaseFaithfulnessScore(
                raw_verdicts="judge model didn't output as expected",
                statements="Statement: statement1\nStatement: statement2",
                expected_score=0.0,
            ),
            # no statements, score is None
            TestCaseFaithfulnessScore(
                raw_verdicts="judge model didn't output as expected",
                statements="",
                expected_score=None,
            ),
        ],
    )
    def test_faithfulness_transform(self, test_case):
        """
        GIVEN a FaithfulnessScores instance.
        WHEN its __call__ method is invoked.
        THEN the correct output is returned.
        """
        get_scores = FaithfulnessScore()
        sample = {
            "raw_verdicts": test_case.raw_verdicts,
            "statements": test_case.statements,
        }
        result = get_scores(sample)
        assert result[EvalAlgorithm.FAITHFULNESS.value] == test_case.expected_score
        if test_case.expected_score is None:
            assert result[DatasetColumns.ERROR.value.name] == "No statements were generated from the answer."

    class TestCaseGetStatements(NamedTuple):
        record: Dict[str, str]
        input_key: str
        output_key: str
        raw_statements: str
        expected_record: Dict[str, str]

    @pytest.mark.parametrize(
        "test_case",
        [
            # find 0 statements from raw statements
            TestCaseGetStatements(
                record={"prompt": "prompt1", "other_input": "other"},
                input_key="prompt",
                output_key="statements",
                raw_statements="raw_statement",
                expected_record={
                    "prompt": "prompt1",
                    "other_input": "other",
                    "statements": "",
                    "raw_statements": "raw_statement",
                },
            ),
            # find statements with prefix string
            TestCaseGetStatements(
                record={"prompt": "prompt1"},
                input_key="prompt",
                output_key="statements",
                raw_statements="Statement: statement1\nStatement: statement2",
                expected_record={
                    "prompt": "prompt1",
                    "statements": "1. Statement: statement1\n2. Statement: statement2",
                    "raw_statements": "Statement: statement1\nStatement: statement2",
                },
            ),
        ],
    )
    def test_get_statements_call(self, test_case):
        """
        GIVEN a GenerateStatement instance.
        WHEN its __call__ method is called on a record.
        THEN the correct output is returned.
        """
        mock_model_runner = Mock()
        mock_model_runner.predict.side_effect = [
            (test_case.raw_statements, None),
        ]
        gen_statements = GetStatements(
            input_key=test_case.input_key, output_key=test_case.output_key, judge_model=mock_model_runner
        )
        result = gen_statements(test_case.record)
        assert result == test_case.expected_record
