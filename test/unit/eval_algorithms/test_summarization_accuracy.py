import re
from typing import NamedTuple, List, Optional
from unittest.mock import patch, MagicMock

import nltk
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
    CategoryScore,
    EvalOutput,
    EvalScore,
    BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES,
    XSUM,
    DEFAULT_PROMPT_TEMPLATE,
    GIGAWORD,
    GOV_REPORT,
)
from fmeval.eval_algorithms.summarization_accuracy import (
    SummarizationAccuracyConfig,
    SummarizationAccuracy,
    METEOR_SCORE,
    ROUGE_SCORE,
    ROUGE_2,
    ROUGE_1,
    ROUGE_L,
    BERT_SCORE,
    get_meteor_score,
    get_rouge_score,
    add_score_to_dataset,
)
from fmeval.eval_algorithms.util import get_bert_score
from fmeval.exceptions import EvalAlgorithmClientError

BERTSCORE_DUMMY_VALUE = (
    0.5  # we don't evaluate the real BERTScore inside unit tests because of runtime, so we hardcode a dummy value
)
DATASET_WITH_SCORES = ray.data.from_items(
    [
        {
            DatasetColumns.MODEL_INPUT.value.name: "Cake is so delicious, I really like cake. I want to open a bakery when I grow up.",
            DatasetColumns.TARGET_OUTPUT.value.name: "I like cake.",
            DatasetColumns.PROMPT.value.name: "Summarize: Cake is so delicious, I really like cake. I want to open a bakery when I "
            "grow up.",
            DatasetColumns.MODEL_OUTPUT.value.name: "I like cake.",
            DatasetColumns.CATEGORY.value.name: "dummy_category_1",
            METEOR_SCORE: 0.1,
            ROUGE_SCORE: 0.1,
            BERT_SCORE: 0.1,
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "The art metropolis of Berlin inspires locals and visitors with its famous "
            "museum landscape and numerous UNESCO World Heritage sites."
            " It is also an international exhibition venue. "
            "You will find a selection of current and upcoming exhibitions here.",
            DatasetColumns.TARGET_OUTPUT.value.name: "Berlin: an art metropolis.",
            DatasetColumns.PROMPT.value.name: "Summarise: The art metropolis of Berlin inspires locals and visitors with its "
            "famous museum landscape and numerous UNESCO World Heritage sites."
            " It is also an international exhibition venue. "
            "You will find a selection of current and upcoming exhibitions here.",
            DatasetColumns.MODEL_OUTPUT.value.name: "Berlin: Art, Heritage, Exhibitions Hub.",
            DatasetColumns.CATEGORY.value.name: "dummy_category_2",
            METEOR_SCORE: 0.2,
            ROUGE_SCORE: 0.2,
            BERT_SCORE: 0.2,
        },
        {
            DatasetColumns.MODEL_INPUT.value.name: "The art metropolis of Berlin inspires locals and visitors with its famous "
            "museum landscape and numerous UNESCO World Heritage sites."
            " It is also an international exhibition venue. "
            "You will find a selection of current and upcoming exhibitions here.",
            DatasetColumns.TARGET_OUTPUT.value.name: "Berlin: an art metropolis.",
            DatasetColumns.PROMPT.value.name: "Summarise: The art metropolis of Berlin inspires locals and visitors with its "
            "famous museum landscape and numerous UNESCO World Heritage sites."
            " It is also an international exhibition venue. "
            "You will find a selection of current and upcoming exhibitions here.",
            DatasetColumns.MODEL_OUTPUT.value.name: "Berlin: Art, Heritage, Exhibitions Hub.",
            DatasetColumns.CATEGORY.value.name: "dummy_category_1",
            METEOR_SCORE: 0.3,
            ROUGE_SCORE: 0.3,
            BERT_SCORE: 0.3,
        },
    ]
)

DATASET = DATASET_WITH_SCORES.drop_columns(cols=[BERT_SCORE, METEOR_SCORE, ROUGE_SCORE])

DATASET_NO_CATEGORY = DATASET.drop_columns(cols=DatasetColumns.CATEGORY.value.name)

EVAL_RESULTS_PATH = "/tmp/eval_results/"


def assert_eval_output(expected_eval_output: EvalOutput, actual_eval_output: EvalOutput):
    """
    Assert Util for eval output, specifically for cases where eval scores are to be asserted with tolerance
    """
    assert actual_eval_output.eval_name == expected_eval_output.eval_name
    assert actual_eval_output.prompt_template == expected_eval_output.prompt_template
    assert actual_eval_output.dataset_name == expected_eval_output.dataset_name
    assert actual_eval_output.output_path == expected_eval_output.output_path

    def _assert_eval_scores(actual_eval_scores, expected_eval_scores):
        assert len(actual_eval_scores) == len(expected_eval_scores)
        for actual_eval_score, expected_eval_score in zip(actual_eval_scores, expected_eval_scores):
            assert actual_eval_score.name == expected_eval_score.name
            assert actual_eval_score.value == pytest.approx(expected_eval_score.value, rel=1e-5)

    _assert_eval_scores(actual_eval_output.dataset_scores, expected_eval_output.dataset_scores)

    if actual_eval_output.category_scores:
        assert len(actual_eval_output.category_scores) == len(expected_eval_output.category_scores)
        for actual_scores, expected_scores in zip(
            actual_eval_output.category_scores, expected_eval_output.category_scores
        ):
            assert actual_scores.name == expected_scores.name
            _assert_eval_scores(actual_scores.scores, expected_scores.scores)


class TestSummarizationAccuracy:
    @fixture(scope="module")
    def config(self) -> SummarizationAccuracyConfig:
        return SummarizationAccuracyConfig()

    @fixture(scope="module")
    def load_meteor_helpers(self):
        nltk.download("wordnet")
        nltk.download("punkt")
        nltk.download("omw-1.4")

    class TestCaseSummarizationAccuracyEvaluateSample(NamedTuple):
        model_output: str
        target_output: str
        expected_response: List[EvalScore]
        rouge_type: str

    class TestCaseSummarizationAccuracyEvaluateSampleInvalid(NamedTuple):
        model_output: str
        target_output: Optional[str]
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSummarizationAccuracyEvaluateSample(
                model_output="I like cake.",
                target_output="I like cake.",
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=0.9921875),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                    EvalScore(name=BERT_SCORE, value=BERTSCORE_DUMMY_VALUE),
                ],
                rouge_type=ROUGE_2,
            ),
            TestCaseSummarizationAccuracyEvaluateSample(
                model_output="Berlin: Art, Heritage, Exhibitions Hub.",
                target_output="Berlin: an art metropolis.",
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=0.5009920634920636),
                    EvalScore(name=ROUGE_SCORE, value=0.0),
                    EvalScore(name=BERT_SCORE, value=BERTSCORE_DUMMY_VALUE),
                ],
                rouge_type=ROUGE_2,
            ),
            TestCaseSummarizationAccuracyEvaluateSample(
                model_output="I like cake.",
                target_output="I like cake.",
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=0.9921875),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                    EvalScore(name=BERT_SCORE, value=BERTSCORE_DUMMY_VALUE),
                ],
                rouge_type=ROUGE_1,
            ),
            TestCaseSummarizationAccuracyEvaluateSample(
                model_output="Berlin: Art, Heritage, Exhibitions Hub.",
                target_output="Berlin: an art metropolis.",
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=0.5009920634920636),
                    EvalScore(name=ROUGE_SCORE, value=0.4444444444444445),
                    EvalScore(name=BERT_SCORE, value=BERTSCORE_DUMMY_VALUE),
                ],
                rouge_type=ROUGE_1,
            ),
            TestCaseSummarizationAccuracyEvaluateSample(
                model_output="I like cake.",
                target_output="I like cake.",
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=0.9921875),
                    EvalScore(name=ROUGE_SCORE, value=1.0),
                    EvalScore(name=BERT_SCORE, value=BERTSCORE_DUMMY_VALUE),
                ],
                rouge_type=ROUGE_L,
            ),
            TestCaseSummarizationAccuracyEvaluateSample(
                model_output="Berlin: Art, Heritage, Exhibitions Hub.",
                target_output="Berlin: an art metropolis.",
                expected_response=[
                    EvalScore(name=METEOR_SCORE, value=0.5009920634920636),
                    EvalScore(name=ROUGE_SCORE, value=0.4444444444444445),
                    EvalScore(name=BERT_SCORE, value=BERTSCORE_DUMMY_VALUE),
                ],
                rouge_type=ROUGE_L,
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreHelperModel")
    @patch("fmeval.eval_algorithms.summarization_accuracy.get_bert_score")
    def test_summarization_accuracy_evaluate_sample(self, mock_get_bert_score, bertscore_helper_model, test_case):
        """
        GIVEN valid inputs
        WHEN SummarizationAccuracy.evaluate_sample is called
        THEN correct List of EvalScores is returned
        """
        # We mock the BertscoreHelperModel class so that an actual ray actor doesn't get created
        bertscore_helper_model_instance = MagicMock()
        bertscore_helper_model.return_value = bertscore_helper_model_instance
        mock_get_bert_score.return_value = BERTSCORE_DUMMY_VALUE
        config = SummarizationAccuracyConfig(rouge_type=test_case.rouge_type)
        eval_algorithm = SummarizationAccuracy(config)
        actual_response = eval_algorithm.evaluate_sample(test_case.target_output, test_case.model_output)
        for actual_eval_score, expected_eval_score in zip(actual_response, test_case.expected_response):
            assert actual_eval_score.name == expected_eval_score.name
            assert actual_eval_score.value == pytest.approx(expected_eval_score.value, rel=1e-5)

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSummarizationAccuracyEvaluateSampleInvalid(
                model_output="I like cake.",
                target_output=None,
                expected_error_message="Missing required input: target_output, for Summarization Accuracy "
                "evaluate_sample",
            ),
            TestCaseSummarizationAccuracyEvaluateSampleInvalid(
                model_output=None,
                target_output="I like cake.",
                expected_error_message="Missing required input: model_output, for Summarization Accuracy "
                "evaluate_sample",
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreHelperModel")
    def test_summarization_accuracy_evaluate_sample_invalid_input(self, bertscore_helper_model, test_case, config):
        """
        GIVEN invalid inputs
        WHEN SummarizationAccuracy.evaluate_sample is called
        THEN correct exception with proper message is raised
        """
        # We mock the BertscoreHelperModel class so that an actual ray actor doesn't get created
        bertscore_helper_model_instance = MagicMock()
        bertscore_helper_model.return_value = bertscore_helper_model_instance

        eval_algorithm = SummarizationAccuracy(config)
        with pytest.raises(EvalAlgorithmClientError, match=test_case.expected_error_message):
            eval_algorithm.evaluate_sample(test_case.target_output, test_case.model_output)

    @pytest.mark.parametrize(
        "rouge_type, bertscore_model_type, expected_error_message",
        [
            (
                "rouge3",
                None,
                "Invalid rouge_type: rouge3 requested in SummarizationAccuracyConfig, please choose "
                "from acceptable values: ['rouge1', 'rouge2', 'rougeL']",
            ),
            (
                "rouge1",
                "distilbert-base-uncased",
                "Invalid model_type_for_bertscore: distilbert-base-uncased requested in "
                "SummarizationAccuracyConfig, please choose from acceptable values: ["
                "'microsoft/deberta-xlarge-mnli', 'roberta-large-mnli']",
            ),
        ],
    )
    def test_summarization_accuracy_invalid_config(self, rouge_type, bertscore_model_type, expected_error_message):
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(expected_error_message)):
            SummarizationAccuracyConfig(rouge_type=rouge_type, model_type_for_bertscore=bertscore_model_type)

    class TestCaseSummarizationAccuracyEvaluate(NamedTuple):
        input_dataset: Dataset
        prompt_template: Optional[str]
        dataset_config: Optional[DataConfig]
        input_dataset_with_generated_model_output: Optional[Dataset]
        expected_response: List[EvalOutput]

    @pytest.mark.parametrize(
        "test_case",
        [
            # Built-in datasets evaluate for dataset without category
            TestCaseSummarizationAccuracyEvaluate(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(
                    cols=[DatasetColumns.PROMPT.value.name, DatasetColumns.MODEL_OUTPUT.value.name]
                ),
                dataset_config=None,
                prompt_template=None,
                input_dataset_with_generated_model_output=DATASET_NO_CATEGORY,
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[XSUM],
                        dataset_name=XSUM,
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.2),
                            EvalScore(name="rouge", value=0.2),
                            EvalScore(name="bertscore", value=0.2),
                        ],
                        category_scores=None,
                        output_path="/tmp/eval_results/summarization_accuracy_xsum.jsonl",
                    ),
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[GIGAWORD],
                        dataset_name=GIGAWORD,
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.2),
                            EvalScore(name="rouge", value=0.2),
                            EvalScore(name="bertscore", value=0.2),
                        ],
                        category_scores=None,
                        output_path="/tmp/eval_results/summarization_accuracy_gigaword.jsonl",
                    ),
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[GOV_REPORT],
                        dataset_name=GOV_REPORT,
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.2),
                            EvalScore(name="rouge", value=0.2),
                            EvalScore(name="bertscore", value=0.2),
                        ],
                        category_scores=None,
                        output_path="/tmp/eval_results/summarization_accuracy_gov_report.jsonl",
                    ),
                ],
            ),
            # Built-in datasets evaluate for dataset with category
            TestCaseSummarizationAccuracyEvaluate(
                input_dataset=DATASET.drop_columns(
                    cols=[DatasetColumns.PROMPT.value.name, DatasetColumns.MODEL_OUTPUT.value.name]
                ),
                dataset_config=None,
                prompt_template=None,
                input_dataset_with_generated_model_output=DATASET,
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[XSUM],
                        dataset_name=XSUM,
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.2),
                            EvalScore(name="rouge", value=0.2),
                            EvalScore(name="bertscore", value=0.2),
                        ],
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1",
                                scores=[
                                    EvalScore(name="meteor", value=0.2),
                                    EvalScore(name="rouge", value=0.2),
                                    EvalScore(name="bertscore", value=0.2),
                                ],
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[
                                    EvalScore(name="meteor", value=0.2),
                                    EvalScore(name="rouge", value=0.2),
                                    EvalScore(name="bertscore", value=0.2),
                                ],
                            ),
                        ],
                        output_path="/tmp/eval_results/summarization_accuracy_xsum.jsonl",
                    ),
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[GIGAWORD],
                        dataset_name=GIGAWORD,
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.2),
                            EvalScore(name="rouge", value=0.2),
                            EvalScore(name="bertscore", value=0.2),
                        ],
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1",
                                scores=[
                                    EvalScore(name="meteor", value=0.2),
                                    EvalScore(name="rouge", value=0.2),
                                    EvalScore(name="bertscore", value=0.2),
                                ],
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[
                                    EvalScore(name="meteor", value=0.2),
                                    EvalScore(name="rouge", value=0.2),
                                    EvalScore(name="bertscore", value=0.2),
                                ],
                            ),
                        ],
                        output_path="/tmp/eval_results/summarization_accuracy_gigaword.jsonl",
                    ),
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template=BUILT_IN_DATASET_DEFAULT_PROMPT_TEMPLATES[GOV_REPORT],
                        dataset_name=GOV_REPORT,
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.2),
                            EvalScore(name="rouge", value=0.2),
                            EvalScore(name="bertscore", value=0.2),
                        ],
                        category_scores=[
                            CategoryScore(
                                name="dummy_category_1",
                                scores=[
                                    EvalScore(name="meteor", value=0.2),
                                    EvalScore(name="rouge", value=0.2),
                                    EvalScore(name="bertscore", value=0.2),
                                ],
                            ),
                            CategoryScore(
                                name="dummy_category_2",
                                scores=[
                                    EvalScore(name="meteor", value=0.2),
                                    EvalScore(name="rouge", value=0.2),
                                    EvalScore(name="bertscore", value=0.2),
                                ],
                            ),
                        ],
                        output_path="/tmp/eval_results/summarization_accuracy_gov_report.jsonl",
                    ),
                ],
            ),
            # Custom dataset evaluate with input prompt template
            TestCaseSummarizationAccuracyEvaluate(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(
                    cols=[DatasetColumns.PROMPT.value.name, DatasetColumns.MODEL_OUTPUT.value.name]
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
                prompt_template="Summarise: $feature",
                input_dataset_with_generated_model_output=DATASET_NO_CATEGORY,
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template="Summarise: $feature",
                        dataset_name="my_custom_dataset",
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.2),
                            EvalScore(name="rouge", value=0.2),
                            EvalScore(name="bertscore", value=0.2),
                        ],
                        category_scores=None,
                        output_path="/tmp/eval_results/summarization_accuracy_my_custom_dataset.jsonl",
                    )
                ],
            ),
            # Custom dataset evaluate without input prompt template
            TestCaseSummarizationAccuracyEvaluate(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(
                    cols=[DatasetColumns.PROMPT.value.name, DatasetColumns.MODEL_OUTPUT.value.name]
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
                input_dataset_with_generated_model_output=DATASET_NO_CATEGORY,
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template=DEFAULT_PROMPT_TEMPLATE,
                        dataset_name="my_custom_dataset",
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.2),
                            EvalScore(name="rouge", value=0.2),
                            EvalScore(name="bertscore", value=0.2),
                        ],
                        category_scores=None,
                        output_path="/tmp/eval_results/summarization_accuracy_my_custom_dataset.jsonl",
                    )
                ],
            ),
        ],
    )
    @patch("fmeval.model_runners.model_runner.ModelRunner")
    @patch("fmeval.eval_algorithms.summarization_accuracy.get_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy.save_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy.generate_model_predict_response_for_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreHelperModel")
    @patch("fmeval.eval_algorithms.summarization_accuracy.add_score_to_dataset")
    def test_summarization_accuracy_evaluate(
        self,
        add_score_to_dataset,
        bertscore_helper_model,
        generate_model_predict_response_for_dataset,
        save_dataset,
        get_dataset,
        model,
        test_case,
        config,
    ):
        """
        GIVEN valid inputs i.e. input data config for a dataset without model_outputs, an input ModelRunner
            and request to save records with scores
        WHEN SummarizationAccuracy evaluate() method is called
        THEN correct EvalOutput is returned
        """
        # We mock the BertscoreHelperModel class so that an actual ray actor doesn't get created
        bertscore_helper_model_instance = MagicMock()
        bertscore_helper_model.return_value = bertscore_helper_model_instance

        add_score_to_dataset.return_value = DATASET_WITH_SCORES
        get_dataset.return_value = test_case.input_dataset
        generate_model_predict_response_for_dataset.return_value = test_case.input_dataset_with_generated_model_output
        eval_algorithm = SummarizationAccuracy(config)
        actual_response = eval_algorithm.evaluate(
            model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template, save=True
        )
        assert save_dataset.called
        assert len(actual_response) == len(test_case.expected_response)
        for i in range(len(actual_response)):
            assert_eval_output(actual_response[i], test_case.expected_response[i])

    @pytest.mark.parametrize(
        "test_case",
        [
            # Built-in datasets evaluate for dataset without category
            TestCaseSummarizationAccuracyEvaluate(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(cols=[DatasetColumns.PROMPT.value.name]),
                dataset_config=DataConfig(
                    dataset_name="my_custom_dataset",
                    dataset_uri="tba",
                    dataset_mime_type=MIME_TYPE_JSON,
                    model_input_location="tba",
                    target_output_location="tba",
                    model_output_location=None,
                    category_location="tba",
                ),
                prompt_template="Summarise: $feature",
                input_dataset_with_generated_model_output=DATASET_NO_CATEGORY,
                expected_response=[
                    EvalOutput(
                        eval_name="summarization_accuracy",
                        prompt_template=None,
                        dataset_name="my_custom_dataset",
                        dataset_scores=[
                            EvalScore(name="meteor", value=0.2),
                            EvalScore(name="rouge", value=0.2),
                            EvalScore(name="bertscore", value=0.2),
                        ],
                        category_scores=None,
                        output_path="/tmp/eval_results/summarization_accuracy_my_custom_dataset.jsonl",
                    )
                ],
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.summarization_accuracy.get_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy.save_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy.generate_model_predict_response_for_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreHelperModel")
    @patch("fmeval.eval_algorithms.summarization_accuracy.add_score_to_dataset")
    def test_summarization_accuracy_evaluate_without_model(
        self,
        add_score_to_dataset,
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
        WHEN SummarizationAccuracy evaluate() method is called
        THEN correct EvalOutput is returned
        """
        # We mock the BertscoreHelperModel class so that an actual ray actor doesn't get created
        bertscore_helper_model_instance = MagicMock()
        bertscore_helper_model.return_value = bertscore_helper_model_instance

        add_score_to_dataset.return_value = DATASET_WITH_SCORES
        get_dataset.return_value = test_case.input_dataset
        eval_algorithm = SummarizationAccuracy(config)
        actual_response = eval_algorithm.evaluate(
            model=None, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template, save=False
        )
        assert not save_dataset.called
        assert not generate_model_predict_response_for_dataset.called
        assert len(actual_response) == len(test_case.expected_response)
        for i in range(len(actual_response)):
            assert_eval_output(actual_response[i], test_case.expected_response[i])

    class TestCaseSummarizationAccuracyEvaluateInvalid(NamedTuple):
        input_dataset: Dataset
        dataset_config: Optional[DataConfig]
        prompt_template: Optional[str]
        model_provided: bool
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSummarizationAccuracyEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(
                    cols=[DatasetColumns.PROMPT.value.name, DatasetColumns.MODEL_OUTPUT.value.name]
                ),
                dataset_config=None,
                prompt_template=None,
                model_provided=False,
                expected_error_message="No ModelRunner provided. ModelRunner is required for inference on model_inputs",
            ),
            TestCaseSummarizationAccuracyEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(
                    cols=[DatasetColumns.PROMPT.value.name, DatasetColumns.MODEL_OUTPUT.value.name]
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
                model_provided=False,
                prompt_template=None,
                expected_error_message="No ModelRunner provided. ModelRunner is required for inference on model_inputs",
            ),
            TestCaseSummarizationAccuracyEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(
                    cols=[
                        DatasetColumns.PROMPT.value.name,
                        DatasetColumns.MODEL_OUTPUT.value.name,
                        DatasetColumns.TARGET_OUTPUT.value.name,
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
                model_provided=True,
                expected_error_message="Missing required column: target_output, for evaluate() method",
            ),
            TestCaseSummarizationAccuracyEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(
                    cols=[
                        DatasetColumns.PROMPT.value.name,
                        DatasetColumns.MODEL_OUTPUT.value.name,
                        DatasetColumns.MODEL_INPUT.value.name,
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
                model_provided=True,
                expected_error_message="Missing required column: model_input, for evaluate() method",
            ),
        ],
    )
    @patch("fmeval.model_runners.model_runner.ModelRunner")
    @patch("fmeval.eval_algorithms.summarization_accuracy.get_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreHelperModel")
    def test_summarization_accuracy_evaluate_invalid_input(
        self, bertscore_helper_model, get_dataset, model, test_case, config
    ):
        """
        GIVEN invalid inputs
        WHEN SummarizationAccuracy.evaluate_sample is called
        THEN correct exception with proper message is raised
        """
        # We mock the BertscoreHelperModel class so that an actual ray actor doesn't get created
        bertscore_helper_model_instance = MagicMock()
        bertscore_helper_model.return_value = bertscore_helper_model_instance

        eval_algorithm = SummarizationAccuracy(config)
        get_dataset.return_value = test_case.input_dataset
        if not test_case.model_provided:
            model = None
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            eval_algorithm.evaluate(
                model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template
            )

    class TestCaseSummarizationAccuracyScores(NamedTuple):
        model_output: str
        target_output: str
        rouge_type: Optional[str]
        expected_score: float

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSummarizationAccuracyScores(
                model_output="I like cake.", target_output="I like cake.", expected_score=0.9921875, rouge_type=None
            ),
            TestCaseSummarizationAccuracyScores(
                model_output="Berlin: Art, Heritage, Exhibitions Hub.",
                target_output="Berlin: an art metropolis.",
                expected_score=0.5009920634920636,
                rouge_type=None,
            ),
        ],
    )
    def test_get_meteor_score(self, test_case, load_meteor_helpers):
        assert pytest.approx(test_case.expected_score, rel=1e-5) == get_meteor_score(
            test_case.target_output,
            test_case.model_output,
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSummarizationAccuracyScores(
                model_output="I like cake.", target_output="I like cake.", expected_score=1.0, rouge_type="rouge1"
            ),
            TestCaseSummarizationAccuracyScores(
                model_output="Berlin: Art, Heritage, Exhibitions Hub.",
                target_output="Berlin: an art metropolis.",
                expected_score=0.0,
                rouge_type="rouge1",
            ),
            TestCaseSummarizationAccuracyScores(
                model_output="I like cake.", target_output="I like cake.", expected_score=1.0, rouge_type="rouge2"
            ),
            TestCaseSummarizationAccuracyScores(
                model_output="Berlin: Art, Heritage, Exhibitions Hub.",
                target_output="Berlin: an art metropolis.",
                expected_score=0.0,
                rouge_type="rouge2",
            ),
            TestCaseSummarizationAccuracyScores(
                model_output="I like cake.", target_output="I like cake.", expected_score=1.0, rouge_type="rougeL"
            ),
            TestCaseSummarizationAccuracyScores(
                model_output="Berlin: Art, Heritage, Exhibitions Hub.",
                target_output="Berlin: an art metropolis.",
                expected_score=0.0,
                rouge_type="rougeL",
            ),
        ],
    )
    def test_get_rouge_score(self, test_case, load_meteor_helpers, config):
        assert pytest.approx(test_case.expected_score, rel=1e-5) == get_rouge_score(
            test_case.target_output,
            test_case.model_output,
            config=config,
        )

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseSummarizationAccuracyScores(
                model_output="I like cake.", target_output="I like cake.", expected_score=0.500000, rouge_type=None
            ),
            TestCaseSummarizationAccuracyScores(
                model_output="Berlin: Art, Heritage, Exhibitions Hub.",
                target_output="Berlin: an art metropolis.",
                expected_score=0.500000,
                rouge_type=None,
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.util.ray.get")
    def test_get_bert_score(self, mock_ray_get, test_case, config):
        mock_ray_get.return_value = BERTSCORE_DUMMY_VALUE
        assert test_case.expected_score == get_bert_score(
            test_case.target_output, test_case.model_output, helper_model=MagicMock()
        )

    @pytest.mark.parametrize(
        "input_dataset",
        [DATASET],
    )
    def test_add_score_to_dataset(self, input_dataset, config):
        response_dataset = add_score_to_dataset(
            dataset=input_dataset,
            score_name_to_func={ROUGE_SCORE: get_rouge_score},
            config=config,
            helper_model=MagicMock(),
        )
        assert response_dataset.count() == input_dataset.count()
        response_dataset_df = response_dataset.to_pandas()
        response_dataset_df = response_dataset_df.sort_values(by=["rouge"])
        assert response_dataset_df.iloc[0]["rouge"] == 0.0
        assert response_dataset_df.iloc[1]["rouge"] == 0.0
        assert response_dataset_df.iloc[2]["rouge"] == 1.0
