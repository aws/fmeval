from typing import List, NamedTuple, Optional
from unittest.mock import patch, Mock

import pytest
import ray
from ray.data import Dataset

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms import EvalAlgorithm, EvalOutput, EvalScore
from fmeval.eval_algorithms.faithfulness import Faithfulness

DATASET_WITH_CONTEXT = ray.data.from_items(
    [
        {
            DatasetColumns.MODEL_INPUT.value.name: "Where and when was Einstein born?",
            DatasetColumns.MODEL_OUTPUT.value.name: "Einstein was born in Germany on 20th March 1879.",
            DatasetColumns.TARGET_CONTEXT.value.name: "Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time",
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
            TestCaseFaithfulnessEvaluateSample(
                model_input="Where and when was Einstein born?",
                model_output="Einstein was born in Germany on 20th March 1879.",
                target_context="Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time",
                statements_output=
                    "Here are the statements created from the given answer:\nStatement: Einstein was born in Germany.\nStatement: Einstein was born on 20th March 1879."
                ,
                verdicts_output=
                    'here are the verdicts for the statements:\n1. statement: einstein was born in germany.\nexplanation: the context states that einstein was "a german-born theoretical physicist". this supports that he was born in germany.\nverdict: yes\n2. statement: einstein was born on 20th march 1879.  \nexplanation: the context states that einstein was "born 14 march 1879". this contradicts the statement that he was born on 20th march 1879.\nverdict: no\nfinal verdicts in order:\nyes. no.'
                ,
                expected_score=[EvalScore(name=EvalAlgorithm.FAITHFULNESS.value, value=1 / 2)],
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
                statements_output="Here are the statements created from the given answer:\nStatement: Einstein was born in Germany.\nStatement: Einstein was born on 20th March 1879.",
                verdicts_output='here are the verdicts for the statements:\n1. statement: einstein was born in germany.\nexplanation: the context states that einstein was "a german-born theoretical physicist". this supports that he was born in germany.\nverdict: yes\n2. statement: einstein was born on 20th march 1879.  \nexplanation: the context states that einstein was "born 14 march 1879". this contradicts the statement that he was born on 20th march 1879.\nverdict: no\nfinal verdicts in order:\nyes. no.',
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
        GIVEN dataset(s) with model outputs already present.
        WHEN the Faithfulness evaluate method is called.
        THEN the correct EvalOutput is returned.

        Note: this is basically an integration test rather than a unit test like `test_evaluate` above.
        It uses a special toy dataset where we are able to compute the exact expected scores by hand.
        The purpose of this test is really to ensure that the correct scores are being generated.
        """
        # GIVEN
        mock_model_runner = Mock()
        mock_model_runner.predict.side_effect = [
            (test_case.statements_output, None),
            (test_case.verdicts_output, None),
        ]
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
