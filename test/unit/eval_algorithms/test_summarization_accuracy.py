import pytest
import re
import ray

from typing import NamedTuple, Optional
from unittest.mock import Mock, patch
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
    DEFAULT_PROMPT_TEMPLATE,
    GIGAWORD,
    GOV_REPORT,
)
from fmeval.helper_models import BertscoreModel
from fmeval.transforms.common import GetModelResponse, GeneratePrompt
from fmeval.transforms.summarization_accuracy_metrics import (
    ROUGE_1,
    ROUGE_2,
    ROUGE_L,
)
from fmeval.eval_algorithms.summarization_accuracy import (
    SummarizationAccuracyConfig,
    SummarizationAccuracy,
    METEOR_SCORE,
    ROUGE_SCORE,
    BERT_SCORE,
)
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
    def eval_algo(self) -> SummarizationAccuracy:
        return SummarizationAccuracy(SummarizationAccuracyConfig(), use_ray=False)

    @pytest.mark.parametrize("use_ray", [True, False])
    @patch("fmeval.eval_algorithms.summarization_accuracy.create_shared_resource")
    def test_init(self, create_shared_resource, use_ray):
        """
        GIVEN valid arguments.
        WHEN a SummarizationAccuracy is initialized.
        THEN create_shared_resource is called when use_ray is True,
            and not called when use_ray is False.
        """
        SummarizationAccuracy(SummarizationAccuracyConfig(), use_ray=use_ray)
        if use_ray:
            create_shared_resource.assert_called_once()
        else:
            create_shared_resource.assert_not_called()

    class TestCaseSummarizationAccuracyInvalidConfig(NamedTuple):
        rouge_type: str
        bertscore_model_type: str
        err_msg: str

    @pytest.mark.parametrize(
        "rouge_type, bertscore_model_type, err_msg",
        [
            TestCaseSummarizationAccuracyInvalidConfig(
                rouge_type="rouge3",
                bertscore_model_type="n/a",
                err_msg="Invalid rouge_type: rouge3 requested in SummarizationAccuracyConfig. Please choose "
                "from acceptable values: ['rouge1', 'rouge2', 'rougeL'].",
            ),
            TestCaseSummarizationAccuracyInvalidConfig(
                rouge_type="rouge1",
                bertscore_model_type="distilbert-base-uncased",
                err_msg="Invalid model_type_for_bertscore: distilbert-base-uncased requested in "
                "SummarizationAccuracyConfig. Please choose from acceptable values: ["
                "'microsoft/deberta-xlarge-mnli', 'roberta-large-mnli'].",
            ),
        ],
    )
    def test_summarization_accuracy_invalid_config(self, rouge_type, bertscore_model_type, err_msg):
        """
        GIVEN invalid inputs.
        WHEN a SummarizationAccuracyConfig is initialized.
        THEN an exception with the correct error message is raised.
        """
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(err_msg)):
            SummarizationAccuracyConfig(rouge_type=rouge_type, model_type_for_bertscore=bertscore_model_type)

    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreModel")
    def test_evaluate_sample(self, bertscore_model):
        """
        GIVEN valid inputs.
        WHEN SummarizationAccuracy.evaluate_sample is called.
        THEN the correct list of EvalScores is returned.
        """
        # Mock the BertscoreModel class so that the actual model doesn't get loaded into memory.
        bertscore_model_instance = Mock(spec=BertscoreModel)
        bertscore_model_instance.invoke_model = Mock(return_value=BERTSCORE_DUMMY_VALUE)
        bertscore_model.return_value = bertscore_model_instance

        model_output = "Berlin: Art, Heritage, Exhibitions Hub."
        target_output = "Berlin: an art metropolis."
        expected_response = [
            EvalScore(name=METEOR_SCORE, value=0.5009920634920636),
            EvalScore(name=ROUGE_SCORE, value=0.4444444444444445),
            EvalScore(name=BERT_SCORE, value=BERTSCORE_DUMMY_VALUE),
        ]
        config = SummarizationAccuracyConfig(rouge_type=ROUGE_L)
        eval_algorithm = SummarizationAccuracy(config, use_ray=False)
        actual_response = eval_algorithm.evaluate_sample(target_output, model_output)
        for actual_eval_score, expected_eval_score in zip(actual_response, expected_response):
            assert actual_eval_score.name == expected_eval_score.name
            assert actual_eval_score.value == pytest.approx(expected_eval_score.value, rel=1e-5)

    @pytest.mark.parametrize(
        "missing_col", [DatasetColumns.TARGET_OUTPUT.value.name, DatasetColumns.MODEL_INPUT.value.name]
    )
    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreModel")
    @patch("fmeval.eval_algorithms.summarization_accuracy.get_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy.get_dataset_configs")
    def test_evaluate_dataset_validation_failure(
        self,
        get_dataset_configs,
        get_dataset,
        bertscore_model,
        missing_col,
        eval_algo,
    ):
        """
        GIVEN a dataset that is missing a required column.
        WHEN SummarizationAccuracy.evaluate is called.
        THEN validate_dataset raises an exception.
        """
        get_dataset_configs.return_value = [Mock()]
        required_cols = {DatasetColumns.TARGET_OUTPUT.value.name, DatasetColumns.MODEL_INPUT.value.name}
        mock_dataset = Mock()
        mock_dataset.columns = Mock(return_value=list(required_cols - {missing_col}))
        get_dataset.return_value = mock_dataset

        with pytest.raises(
            EvalAlgorithmClientError, match=re.escape(f"Missing required column: {missing_col}, for evaluate() method")
        ):
            bertscore_model.return_value = Mock()
            eval_algo.evaluate(
                model=Mock(),
                dataset_config=Mock(),
                prompt_template="",
            )

    class TestCaseEvaluateDatasetWithoutModelOutputColumn(NamedTuple):
        user_provided_prompt_template: Optional[str]
        eval_output_prompt_template: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseEvaluateDatasetWithoutModelOutputColumn(
                user_provided_prompt_template="Summarize $model_input",
                eval_output_prompt_template="Summarize $model_input",
            ),
            TestCaseEvaluateDatasetWithoutModelOutputColumn(
                user_provided_prompt_template=None,
                eval_output_prompt_template="$model_input",
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.summarization_accuracy.save_dataset")
    @patch("fmeval.transforms.transform_pipeline.TransformPipeline.execute")
    @patch("fmeval.eval_algorithms.summarization_accuracy.GetModelResponse")
    @patch("fmeval.eval_algorithms.summarization_accuracy.GeneratePrompt")
    @patch("fmeval.eval_algorithms.summarization_accuracy.get_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy.get_dataset_configs")
    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreModel")
    def test_evaluate_where_dataset_is_missing_model_output(
        self,
        bertscore_model,
        get_dataset_configs,
        get_dataset,
        generate_prompt,
        get_model_response,
        pipeline_execute,
        save_dataset,
        test_case,
        eval_algo,
    ):
        """
        GIVEN a valid dataset that doesn't contain a column for model outputs.
        WHEN the SummarizationAccuracy evaluate method is called with save=True.
        THEN the expected prompt-generation and model-invocation transforms are initialized,
            EvalOutputs that are returned contain the correct scores,
            the EvalOutputs indicate that the correct prompt templates were used,
            and save_dataset is called.
        """
        # Set up mocks
        model_runner = Mock()
        bertscore_model.return_value = Mock()

        generate_prompt_instance = Mock(spec=GeneratePrompt)
        generate_prompt_instance.output_keys = ["prompt"]
        generate_prompt.return_value = generate_prompt_instance

        get_model_response_instance = Mock(spec=GetModelResponse)
        get_model_response_instance.output_keys = ["model_output"]
        get_model_response.return_value = get_model_response_instance

        dataset_config = Mock()
        dataset_config.dataset_name = "my_custom_dataset"
        get_dataset_configs.return_value = [dataset_config]

        mock_dataset = Mock()
        mock_dataset.columns = Mock(
            return_value=[DatasetColumns.TARGET_OUTPUT.value.name, DatasetColumns.MODEL_INPUT.value.name]
        )
        get_dataset.return_value = mock_dataset

        pipeline_execute.return_value = DATASET_WITH_SCORES

        # Expected outputs from calling `evaluate`.
        expected_dataset_scores = [
            EvalScore(name="meteor", value=0.2),
            EvalScore(name="rouge", value=0.2),
            EvalScore(name="bertscore", value=0.2),
        ]
        expected_category_scores = [
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
        ]
        expected_outputs = [
            EvalOutput(
                eval_name="summarization_accuracy",
                prompt_template=test_case.eval_output_prompt_template,
                dataset_name="my_custom_dataset",
                dataset_scores=expected_dataset_scores,
                category_scores=expected_category_scores,
                output_path="/tmp/eval_results/summarization_accuracy_my_custom_dataset.jsonl",
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
            prompt_template=test_case.eval_output_prompt_template,
        )
        get_model_response.assert_called_with(
            input_to_output_keys={DatasetColumns.PROMPT.value.name: [DatasetColumns.MODEL_OUTPUT.value.name]},
            model_runner=model_runner,
        )
        save_dataset.assert_called_once()
        assert len(eval_outputs) == len(expected_outputs)
        for expected, actual in zip(expected_outputs, eval_outputs):
            assert_eval_output(expected, actual)

    @patch("fmeval.eval_algorithms.summarization_accuracy.save_dataset")
    @patch("fmeval.transforms.transform_pipeline.TransformPipeline.execute")
    @patch("fmeval.eval_algorithms.summarization_accuracy.GetModelResponse")
    @patch("fmeval.eval_algorithms.summarization_accuracy.GeneratePrompt")
    @patch("fmeval.eval_algorithms.summarization_accuracy.get_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy.get_dataset_configs")
    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreModel")
    def test_evaluate_where_dataset_contains_model_output(
        self,
        bertscore_model,
        get_dataset_configs,
        get_dataset,
        generate_prompt,
        get_model_response,
        pipeline_execute,
        save_dataset,
        eval_algo,
    ):
        """
        GIVEN a valid dataset that already contains a column for model outputs.
        WHEN the SummarizationAccuracy evaluate method is called with save=False.
        THEN the prompt-generation and model-invocation transforms are not initialized,
            EvalOutputs that are returned contain the correct scores,
            the EvalOutputs indicate that the correct prompt templates were used,
            and save_dataset is not called.
        """
        # Set up mocks
        bertscore_model.return_value = Mock()

        dataset_config = Mock()
        dataset_config.dataset_name = "my_custom_dataset"
        get_dataset_configs.return_value = [dataset_config]

        mock_dataset = Mock()
        mock_dataset.columns = Mock(
            return_value=[
                DatasetColumns.TARGET_OUTPUT.value.name,
                DatasetColumns.MODEL_INPUT.value.name,
                DatasetColumns.MODEL_OUTPUT.value.name,
            ]
        )
        get_dataset.return_value = mock_dataset

        pipeline_execute.return_value = DATASET_WITH_SCORES.drop_columns(cols=DatasetColumns.CATEGORY.value.name)

        # Expected outputs from calling `evaluate`.
        expected_outputs = [
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
        ]

        # Call `evaluate` and validate outputs.
        eval_outputs = eval_algo.evaluate(
            model=None, dataset_config=dataset_config, prompt_template="Summarize $model_input", save=False
        )

        generate_prompt.assert_not_called()
        get_model_response.assert_not_called()
        save_dataset.assert_not_called()
        assert len(eval_outputs) == len(expected_outputs)
        for expected, actual in zip(expected_outputs, eval_outputs):
            assert_eval_output(expected, actual)

    class TestCaseEvaluateInvalid(NamedTuple):
        input_dataset: Dataset
        dataset_config: Optional[DataConfig]
        prompt_template: Optional[str]
        model_provided: bool
        expected_error_message: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseEvaluateInvalid(
                input_dataset=DATASET_NO_CATEGORY.drop_columns(
                    cols=[DatasetColumns.PROMPT.value.name, DatasetColumns.MODEL_OUTPUT.value.name]
                ),
                dataset_config=None,
                prompt_template=None,
                model_provided=False,
                expected_error_message="No ModelRunner provided. ModelRunner is required for inference on model_inputs",
            ),
            TestCaseEvaluateInvalid(
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
            TestCaseEvaluateInvalid(
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
            TestCaseEvaluateInvalid(
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
    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreModel")
    def test_summarization_accuracy_evaluate_invalid_input(
        self, bertscore_model, get_dataset, model, test_case, eval_algo
    ):
        """
        GIVEN invalid inputs.
        WHEN SummarizationAccuracy evaluate method is called.
        THEN an exception with the proper message is raised.
        """
        bertscore_model.return_value = Mock()
        get_dataset.return_value = test_case.input_dataset
        if not test_case.model_provided:
            model = None
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(test_case.expected_error_message)):
            eval_algo.evaluate(
                model=model, dataset_config=test_case.dataset_config, prompt_template=test_case.prompt_template
            )
