from dataclasses import dataclass
from typing import NamedTuple, Optional
from unittest.mock import patch, Mock

import pytest
import ray
import numpy as np

from fmeval.constants import (
    DatasetColumns,
    MIME_TYPE_JSON,
    DEFAULT_EVAL_RESULTS_PATH,
)
from fmeval.eval_algorithms import (
    EvalOutput,
    CategoryScore,
    EvalScore,
    EvalAlgorithm,
    DEFAULT_PROMPT_TEMPLATE,
    CROWS_PAIRS,
)
from fmeval.eval_algorithms.prompt_stereotyping import (
    PromptStereotyping,
    PROMPT_STEREOTYPING,
    LOG_PROBABILITY_DIFFERENCE,
)
from fmeval.exceptions import EvalAlgorithmClientError

GENDER_MORE_STEREOTYPICAL_STATEMENT = "Statement that is more stereotypical about gender"
GENDER_LESS_STEREOTYPICAL_STATEMENT = "Statement that is less stereotypical about gender"
SOCIO_ECONOMIC_MORE_STEREOTYPICAL_STATEMENT = "Statement that is more stereotypical about socio-economic status"
SOCIO_ECONOMIC_LESS_STEREOTYPICAL_STATEMENT = "Statement that is less stereotypical about socio-economic status"
NATIONALITY_MORE_STEREOTYPICAL_STATEMENT = "Statement that is more stereotypical about nationality"
NATIONALITY_LESS_STEREOTYPICAL_STATEMENT = "Statement that is less stereotypical about nationality"
SEXUAL_ORIENTIATION_MORE_STEREOTYPICAL_STATEMENT = "Statement that is more stereotypical about sexual orientation"
SEXUAL_ORIENTIATION_LESS_STEREOTYPICAL_STATEMENT = "Statement that is less stereotypical about sexual orientation"


@dataclass
class TestCasePromptStereotypingEvaluateSampleInvalid:
    sent_more_log_probability: float
    sent_less_log_probability: float
    expected_error_message: str


class TestPromptStereotyping:
    @pytest.mark.parametrize(
        "test_case",
        [
            TestCasePromptStereotypingEvaluateSampleInvalid(
                sent_less_log_probability=np.log(0.8),
                sent_more_log_probability=None,
                expected_error_message="Prompt stereotyping evaluation requires sent_more_log_probability and sent_less_log_probability",
            ),
            TestCasePromptStereotypingEvaluateSampleInvalid(
                sent_less_log_probability=None,
                sent_more_log_probability=np.log(0.7),
                expected_error_message="Prompt stereotyping evaluation requires sent_more_log_probability and sent_less_log_probability",
            ),
            TestCasePromptStereotypingEvaluateSampleInvalid(
                sent_less_log_probability=np.log(0.8),
                sent_more_log_probability="prob",
                expected_error_message="Prompt stereotyping evaluation requires sent_more_log_probability and sent_less_log_probability to be float",
            ),
            TestCasePromptStereotypingEvaluateSampleInvalid(
                sent_less_log_probability="-0.8",
                sent_more_log_probability=np.log(0.7),
                expected_error_message="Prompt stereotyping evaluation requires sent_more_log_probability and sent_less_log_probability to be float",
            ),
            TestCasePromptStereotypingEvaluateSampleInvalid(
                sent_less_log_probability=-0.8,
                sent_more_log_probability=0.7,
                expected_error_message="Log-probabilities cannot be positive values. You might have passed raw probabilities instead.",
            ),
        ],
    )
    def test_prompt_stereotyping_evaluate_sample_invalid_input(self, test_case):
        """
        GIVEN invalid inputs
        WHEN FactualKnowledge.evaluate_sample is called
        THEN correct exception with proper message is raised
        """
        eval_algorithm = PromptStereotyping()
        with pytest.raises(EvalAlgorithmClientError, match=test_case.expected_error_message):
            eval_algorithm.evaluate_sample(test_case.sent_more_log_probability, test_case.sent_less_log_probability)

    class TestCasePromptStereotypingEvaluateWithModel(NamedTuple):
        user_provided_prompt_template: Optional[str]
        dataset_prompt_template: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCasePromptStereotypingEvaluateWithModel(
                user_provided_prompt_template=None,
                dataset_prompt_template="$model_input",
            ),
            TestCasePromptStereotypingEvaluateWithModel(
                user_provided_prompt_template="Do something with $model_input",
                dataset_prompt_template="Do something with $model_input",
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.prompt_stereotyping.save_dataset")
    @patch("fmeval.eval_algorithms.prompt_stereotyping.generate_output_dataset_path")
    @patch("fmeval.eval_algorithms.prompt_stereotyping.get_dataset")
    @patch("fmeval.eval_algorithms.prompt_stereotyping.get_dataset_configs")
    def test_prompt_stereotyping_evaluate_with_model(
        self,
        mock_get_dataset_configs,
        mock_get_dataset,
        mock_generate_output_dataset_path,
        mock_save_dataset,
        test_case,
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset without model_outputs,
            an input ModelRunner, and request to save records with scores.
        WHEN PromptStereotyping.evaluate is called
        THEN correct EvalOutput is returned
        """
        input_dataset = ray.data.from_items(
            [
                {
                    DatasetColumns.SENT_MORE_INPUT.value.name: GENDER_MORE_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.SENT_LESS_INPUT.value.name: GENDER_LESS_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.CATEGORY.value.name: "gender",
                },
                {
                    DatasetColumns.SENT_MORE_INPUT.value.name: SOCIO_ECONOMIC_MORE_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.SENT_LESS_INPUT.value.name: SOCIO_ECONOMIC_LESS_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.CATEGORY.value.name: "socioeconomic",
                },
                {
                    DatasetColumns.SENT_MORE_INPUT.value.name: NATIONALITY_MORE_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.SENT_LESS_INPUT.value.name: NATIONALITY_LESS_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.CATEGORY.value.name: "nationality",
                },
                {
                    DatasetColumns.SENT_MORE_INPUT.value.name: SEXUAL_ORIENTIATION_MORE_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.SENT_LESS_INPUT.value.name: SEXUAL_ORIENTIATION_LESS_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.CATEGORY.value.name: "sexual-orientation",
                },
            ]
        )

        dataset_config = Mock()
        dataset_config.dataset_name = "my_custom_dataset"
        mock_get_dataset_configs.return_value = [dataset_config]

        mock_get_dataset.return_value = input_dataset

        model_runner = Mock()
        model_runner.predict.side_effect = [
            (None, np.log(0.9)),  # sent_more
            (None, np.log(0.5)),  # sent_less
            (None, np.log(0.2)),
            (None, np.log(0.7)),
            (None, np.log(0.8)),
            (None, np.log(0.6)),
            (None, np.log(0.1)),
            (None, np.log(0.4)),
        ]

        mock_generate_output_dataset_path.return_value = "/path/to/output/dataset"

        # Expected scores
        dataset_scores = [EvalScore(name=PROMPT_STEREOTYPING, value=0.5)]
        category_scores = [
            CategoryScore(name="gender", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=1)]),
            CategoryScore(name="socioeconomic", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=0)]),
            CategoryScore(name="nationality", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=1)]),
            CategoryScore(name="sexual-orientation", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=0)]),
        ]
        expected_output = EvalOutput(
            eval_name=PROMPT_STEREOTYPING,
            dataset_name=dataset_config.dataset_name,
            prompt_template=test_case.dataset_prompt_template,
            dataset_scores=dataset_scores,
            category_scores=category_scores,
            output_path="/path/to/output/dataset",
        )

        eval_algo = PromptStereotyping()
        eval_output = eval_algo.evaluate(
            model=model_runner, prompt_template=test_case.user_provided_prompt_template, save=True
        )[0]

        mock_save_dataset.assert_called_once()
        assert eval_output == expected_output

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCasePromptStereotypingEvaluateWithModel(
                user_provided_prompt_template=None,
                dataset_prompt_template=None,
            ),
            TestCasePromptStereotypingEvaluateWithModel(
                user_provided_prompt_template="Do something with $model_input",
                dataset_prompt_template=None,
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.prompt_stereotyping.save_dataset")
    @patch("fmeval.eval_algorithms.prompt_stereotyping.generate_output_dataset_path")
    @patch("fmeval.eval_algorithms.prompt_stereotyping.get_dataset")
    @patch("fmeval.eval_algorithms.prompt_stereotyping.get_dataset_configs")
    def test_evaluate_without_model(
        self,
        mock_get_dataset_configs,
        mock_get_dataset,
        mock_generate_output_dataset_path,
        mock_save_dataset,
        test_case,
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset with model_outputs,
            and no request to save records with scores
        WHEN PromptStereotyping.evaluate is called
        THEN correct EvalOutput is returned
        """
        input_dataset = ray.data.from_items(
            [
                {
                    DatasetColumns.SENT_MORE_INPUT.value.name: GENDER_MORE_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.SENT_LESS_INPUT.value.name: GENDER_LESS_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.CATEGORY.value.name: "gender",
                    DatasetColumns.SENT_MORE_LOG_PROB.value.name: np.log(0.9),
                    DatasetColumns.SENT_LESS_LOG_PROB.value.name: np.log(0.5),
                },
                {
                    DatasetColumns.SENT_MORE_INPUT.value.name: SOCIO_ECONOMIC_MORE_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.SENT_LESS_INPUT.value.name: SOCIO_ECONOMIC_LESS_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.CATEGORY.value.name: "socioeconomic",
                    DatasetColumns.SENT_MORE_LOG_PROB.value.name: np.log(0.2),
                    DatasetColumns.SENT_LESS_LOG_PROB.value.name: np.log(0.7),
                },
                {
                    DatasetColumns.SENT_MORE_INPUT.value.name: NATIONALITY_MORE_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.SENT_LESS_INPUT.value.name: NATIONALITY_LESS_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.CATEGORY.value.name: "nationality",
                    DatasetColumns.SENT_MORE_LOG_PROB.value.name: np.log(0.8),
                    DatasetColumns.SENT_LESS_LOG_PROB.value.name: np.log(0.6),
                },
                {
                    DatasetColumns.SENT_MORE_INPUT.value.name: SEXUAL_ORIENTIATION_MORE_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.SENT_LESS_INPUT.value.name: SEXUAL_ORIENTIATION_LESS_STEREOTYPICAL_STATEMENT,
                    DatasetColumns.CATEGORY.value.name: "sexual-orientation",
                    DatasetColumns.SENT_MORE_LOG_PROB.value.name: np.log(0.1),
                    DatasetColumns.SENT_LESS_LOG_PROB.value.name: np.log(0.4),
                },
            ]
        )

        dataset_config = Mock()
        dataset_config.dataset_name = "my_custom_dataset"
        mock_get_dataset_configs.return_value = [dataset_config]

        mock_get_dataset.return_value = input_dataset
        mock_generate_output_dataset_path.return_value = "/path/to/output/dataset"

        # Expected scores
        dataset_scores = [EvalScore(name=PROMPT_STEREOTYPING, value=0.5)]
        category_scores = [
            CategoryScore(name="gender", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=1)]),
            CategoryScore(name="socioeconomic", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=0)]),
            CategoryScore(name="nationality", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=1)]),
            CategoryScore(name="sexual-orientation", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=0)]),
        ]

        expected_output = EvalOutput(
            eval_name=PROMPT_STEREOTYPING,
            dataset_name=dataset_config.dataset_name,
            prompt_template=test_case.dataset_prompt_template,
            dataset_scores=dataset_scores,
            category_scores=category_scores,
            output_path="/path/to/output/dataset",
        )

        eval_algo = PromptStereotyping()
        eval_output = eval_algo.evaluate(
            model=None,
            prompt_template=test_case.user_provided_prompt_template,
            save=False,
        )[0]

        mock_save_dataset.assert_not_called()
        assert eval_output == expected_output

    def test_evaluate_sample(self):
        assert PromptStereotyping().evaluate_sample(-3.0, -5.0) == [
            EvalScore(name=LOG_PROBABILITY_DIFFERENCE, value=2.0)
        ]
