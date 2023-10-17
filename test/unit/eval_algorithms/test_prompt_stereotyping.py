from dataclasses import dataclass
from typing import NamedTuple, Optional, List
from unittest.mock import patch

import pytest
import ray
from ray.data import Dataset

from amazon_fmeval.constants import (
    MIME_TYPE_JSON,
    CATEGORY_COLUMN_NAME,
    SENT_MORE_INPUT_COLUMN_NAME,
    SENT_LESS_INPUT_COLUMN_NAME,
    SENT_LESS_LOG_PROB_COLUMN_NAME,
    SENT_MORE_LOG_PROB_COLUMN_NAME,
    DEFAULT_EVAL_RESULTS_PATH,
)
from amazon_fmeval.data_loaders.util import DataConfig
from amazon_fmeval.eval_algorithms import EvalOutput, CategoryScore, EvalScore, EvalAlgorithm
from amazon_fmeval.eval_algorithms.prompt_stereotyping import (
    PromptStereotyping,
    PROMPT_STEREOTYPING,
    LOG_PROBABILITY_DIFFERENCE,
)
from amazon_fmeval.exceptions import EvalAlgorithmClientError

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
                sent_less_log_probability=0.8,
                sent_more_log_probability=None,
                expected_error_message="Stereoptyping evaluation requires sent_more_log_probability and sent_less_log_probability",
            ),
            TestCasePromptStereotypingEvaluateSampleInvalid(
                sent_less_log_probability=None,
                sent_more_log_probability=0.7,
                expected_error_message="Stereoptyping evaluation requires sent_more_log_probability and sent_less_log_probability",
            ),
            TestCasePromptStereotypingEvaluateSampleInvalid(
                sent_less_log_probability=0.8,
                sent_more_log_probability="prob",
                expected_error_message="Stereoptyping evaluation requires sent_more_log_probability and sent_less_log_probability to be float",
            ),
            TestCasePromptStereotypingEvaluateSampleInvalid(
                sent_less_log_probability="0.8",
                sent_more_log_probability=0.7,
                expected_error_message="Stereoptyping evaluation requires sent_more_log_probability and sent_less_log_probability to be float",
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

    class TestCasePromptStereotypingEvaluate(NamedTuple):
        input_dataset: Dataset
        input_dataset_with_generated_model_output: Optional[Dataset]
        expected_response: List[EvalOutput]
        prompt_template: Optional[str] = None
        dataset_config: Optional[DataConfig] = None
        output_results_path: Optional[str] = None

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCasePromptStereotypingEvaluate(
                input_dataset=ray.data.from_items(
                    [
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: GENDER_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: GENDER_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "gender",
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: SOCIO_ECONOMIC_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: SOCIO_ECONOMIC_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "socioeconomic",
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: NATIONALITY_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: NATIONALITY_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "nationality",
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: SEXUAL_ORIENTIATION_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: SEXUAL_ORIENTIATION_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "sexual-orientation",
                        },
                    ]
                ),
                input_dataset_with_generated_model_output=ray.data.from_items(
                    [
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: GENDER_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: GENDER_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "gender",
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.9,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.5,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: SOCIO_ECONOMIC_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: SOCIO_ECONOMIC_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "socioeconomic",
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.2,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.7,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: NATIONALITY_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: NATIONALITY_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "nationality",
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.8,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.6,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: SEXUAL_ORIENTIATION_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: SEXUAL_ORIENTIATION_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "sexual-orientation",
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.1,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.4,
                        },
                    ]
                ),
                expected_response=[
                    EvalOutput(
                        eval_name=EvalAlgorithm.PROMPT_STEREOTYPING.value,
                        dataset_name="crow-pairs",
                        prompt_template="$feature",
                        dataset_scores=[EvalScore(name=PROMPT_STEREOTYPING, value=0.5)],
                        category_scores=[
                            CategoryScore(name="gender", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=1)]),
                            CategoryScore(name="socioeconomic", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=0)]),
                            CategoryScore(name="nationality", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=1)]),
                            CategoryScore(
                                name="sexual-orientation", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=0)]
                            ),
                        ],
                        output_path="/tmp/eval_results/",
                    )
                ],
            ),
            TestCasePromptStereotypingEvaluate(
                input_dataset=ray.data.from_items(
                    [
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: GENDER_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: GENDER_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "gender",
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: SOCIO_ECONOMIC_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: SOCIO_ECONOMIC_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "socioeconomic",
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: NATIONALITY_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: NATIONALITY_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "nationality",
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: SEXUAL_ORIENTIATION_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: SEXUAL_ORIENTIATION_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "sexual-orientation",
                        },
                    ]
                ),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    sent_more_input_location="sent_more",
                    sent_less_input_location="sent_less",
                ),
                prompt_template="Is this stereotypical? $feature",
                input_dataset_with_generated_model_output=ray.data.from_items(
                    [
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: GENDER_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: GENDER_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "gender",
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.9,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.5,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: SOCIO_ECONOMIC_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: SOCIO_ECONOMIC_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "socioeconomic",
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.2,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.7,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: NATIONALITY_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: NATIONALITY_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "nationality",
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.8,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.6,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: SEXUAL_ORIENTIATION_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: SEXUAL_ORIENTIATION_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "sexual-orientation",
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.1,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.4,
                        },
                    ]
                ),
                output_results_path="/output/results/path",
                expected_response=[
                    EvalOutput(
                        eval_name=EvalAlgorithm.PROMPT_STEREOTYPING.value,
                        dataset_name="my_custom_dataset",
                        prompt_template="Is this stereotypical? $feature",
                        dataset_scores=[EvalScore(name=PROMPT_STEREOTYPING, value=0.5)],
                        category_scores=[
                            CategoryScore(name="gender", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=1)]),
                            CategoryScore(name="socioeconomic", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=0)]),
                            CategoryScore(name="nationality", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=1)]),
                            CategoryScore(
                                name="sexual-orientation", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=0)]
                            ),
                        ],
                        output_path="/tmp/eval_results/",
                    )
                ],
            ),
            TestCasePromptStereotypingEvaluate(
                input_dataset=ray.data.from_items(
                    [
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: GENDER_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: GENDER_LESS_STEREOTYPICAL_STATEMENT,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: SOCIO_ECONOMIC_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: SOCIO_ECONOMIC_LESS_STEREOTYPICAL_STATEMENT,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: NATIONALITY_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: NATIONALITY_LESS_STEREOTYPICAL_STATEMENT,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: SEXUAL_ORIENTIATION_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: SEXUAL_ORIENTIATION_LESS_STEREOTYPICAL_STATEMENT,
                        },
                    ]
                ),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                ),
                prompt_template="$feature",
                input_dataset_with_generated_model_output=ray.data.from_items(
                    [
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: GENDER_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: GENDER_LESS_STEREOTYPICAL_STATEMENT,
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.9,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.5,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: SOCIO_ECONOMIC_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: SOCIO_ECONOMIC_LESS_STEREOTYPICAL_STATEMENT,
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.2,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.7,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: NATIONALITY_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: NATIONALITY_LESS_STEREOTYPICAL_STATEMENT,
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.8,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.6,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: SEXUAL_ORIENTIATION_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: SEXUAL_ORIENTIATION_LESS_STEREOTYPICAL_STATEMENT,
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.1,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.4,
                        },
                    ]
                ),
                expected_response=[
                    EvalOutput(
                        eval_name=EvalAlgorithm.PROMPT_STEREOTYPING.value,
                        dataset_name="my_custom_dataset",
                        prompt_template="$feature",
                        dataset_scores=[EvalScore(name=PROMPT_STEREOTYPING, value=0.5)],
                        category_scores=None,
                        output_path="/tmp/eval_results/",
                    )
                ],
            ),
        ],
    )
    @patch("amazon_fmeval.model_runners.model_runner.ModelRunner")
    @patch("amazon_fmeval.eval_algorithms.prompt_stereotyping.get_dataset")
    @patch("amazon_fmeval.eval_algorithms.prompt_stereotyping.save_dataset")
    @patch("amazon_fmeval.eval_algorithms.prompt_stereotyping.generate_model_predict_response_for_dataset")
    def test_prompt_stereotyping_evaluate(
        self,
        generate_model_output_for_dataset,
        save_dataset,
        get_dataset,
        model,
        test_case,
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset without model_outputs, an input ModelRunner
            and request to save records with scores
        WHEN PromptStereotyping.evaluate is called
        THEN correct EvalOutput is returned
        """
        get_dataset.return_value = test_case.input_dataset
        generate_model_output_for_dataset.return_value = test_case.input_dataset_with_generated_model_output
        eval_algorithm = PromptStereotyping()
        actual_response = eval_algorithm.evaluate(
            model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template, save=True
        )
        assert actual_response == test_case.expected_response
        assert save_dataset.called
        assert generate_model_output_for_dataset.called

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCasePromptStereotypingEvaluate(
                input_dataset=ray.data.from_items(
                    [
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: GENDER_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: GENDER_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "gender",
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.9,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.5,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: SOCIO_ECONOMIC_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: SOCIO_ECONOMIC_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "socioeconomic",
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.2,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.7,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: NATIONALITY_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: NATIONALITY_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "nationality",
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.8,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.6,
                        },
                        {
                            SENT_MORE_INPUT_COLUMN_NAME: SEXUAL_ORIENTIATION_MORE_STEREOTYPICAL_STATEMENT,
                            SENT_LESS_INPUT_COLUMN_NAME: SEXUAL_ORIENTIATION_LESS_STEREOTYPICAL_STATEMENT,
                            CATEGORY_COLUMN_NAME: "sexual-orientation",
                            SENT_MORE_LOG_PROB_COLUMN_NAME: 0.1,
                            SENT_LESS_LOG_PROB_COLUMN_NAME: 0.4,
                        },
                    ]
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
                input_dataset_with_generated_model_output=None,
                expected_response=[
                    EvalOutput(
                        eval_name=EvalAlgorithm.PROMPT_STEREOTYPING.value,
                        dataset_name="my_custom_dataset",
                        prompt_template=None,
                        dataset_scores=[EvalScore(name=PROMPT_STEREOTYPING, value=0.5)],
                        category_scores=[
                            CategoryScore(name="gender", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=1)]),
                            CategoryScore(name="socioeconomic", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=0)]),
                            CategoryScore(name="nationality", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=1)]),
                            CategoryScore(
                                name="sexual-orientation", scores=[EvalScore(name=PROMPT_STEREOTYPING, value=0)]
                            ),
                        ],
                        output_path="/tmp/eval_results/",
                    )
                ],
            )
        ],
    )
    @patch("amazon_fmeval.eval_algorithms.prompt_stereotyping.get_dataset")
    @patch("amazon_fmeval.eval_algorithms.prompt_stereotyping.save_dataset")
    @patch("amazon_fmeval.eval_algorithms.prompt_stereotyping.generate_model_predict_response_for_dataset")
    def test_prompt_stereotyping_evaluate_without_model(
        self, generate_model_output_for_dataset, save_dataset, get_dataset, test_case
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset with model_outputs,
            and no request to save records with scores
        WHEN PromptStereotyping.evaluate is called
        THEN correct EvalOutput is returned
        """
        get_dataset.return_value = test_case.input_dataset
        generate_model_output_for_dataset.return_value = test_case.input_dataset_with_generated_model_output
        eval_algorithm = PromptStereotyping()
        actual_response = eval_algorithm.evaluate(model=None, dataset_config=test_case.dataset_config)
        assert not generate_model_output_for_dataset.called
        assert not save_dataset.called
        assert actual_response == test_case.expected_response

    def test_evaluate_sample(self):
        assert PromptStereotyping().evaluate_sample(0.5, 0.3) == [EvalScore(name=LOG_PROBABILITY_DIFFERENCE, value=0.2)]
