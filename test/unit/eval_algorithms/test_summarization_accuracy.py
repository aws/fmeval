import pytest
import re
import ray

from typing import NamedTuple, Optional
from unittest.mock import Mock, patch, call
from _pytest.fixtures import fixture

from fmeval.constants import DatasetColumns, MEAN
from fmeval.eval_algorithms import EvalScore
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModel
from fmeval.transforms.summarization_accuracy_metrics import ROUGE_L
from fmeval.eval_algorithms.summarization_accuracy import (
    SummarizationAccuracyConfig,
    SummarizationAccuracy,
    METEOR_SCORE,
    ROUGE_SCORE,
    BERT_SCORE,
    METRIC_NAMES,
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


class TestSummarizationAccuracy:
    @fixture(scope="module")
    def eval_algo(self) -> SummarizationAccuracy:
        return SummarizationAccuracy(SummarizationAccuracyConfig())

    @patch("fmeval.eval_algorithms.summarization_accuracy.TransformPipeline")
    @patch("fmeval.eval_algorithms.summarization_accuracy.SummarizationAccuracy._create_transforms")
    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreHelperModel")
    def test_init(self, bertscore_model_cls, mock_create_transforms, mock_transform_pipeline_cls):
        """
        GIVEN default arguments.
        WHEN a SummarizationAccuracy is initialized.
        THEN SummarizationAccuracy._create_transforms is called,
            a TransformPipeline is initialized with the correct Transforms,
            and said pipeline is set to the instance's `pipeline` attribute.
        """
        mock_meteor, mock_rouge, mock_bertscore = Mock(), Mock(), Mock()
        mock_create_transforms.return_value = mock_meteor, mock_rouge, mock_bertscore
        config = SummarizationAccuracyConfig()
        summ_acc = SummarizationAccuracy(config)

        bertscore_model_cls.assert_called_with(config.model_type_for_bertscore)
        assert summ_acc.bertscore_model == bertscore_model_cls.return_value

        mock_transform_pipeline_cls.assert_called_with([mock_meteor, mock_rouge, mock_bertscore])
        assert summ_acc.pipeline == mock_transform_pipeline_cls.return_value

    class TestCaseSummarizationAccuracyInvalidConfig(NamedTuple):
        rouge_type: str
        model_type_for_bertscore: str
        err_msg: str

    @pytest.mark.parametrize(
        "rouge_type, model_type_for_bertscore, err_msg",
        [
            TestCaseSummarizationAccuracyInvalidConfig(
                rouge_type="rouge3",
                model_type_for_bertscore="n/a",
                err_msg="Invalid rouge_type: rouge3 requested in SummarizationAccuracyConfig. Please choose "
                "from acceptable values: ['rouge1', 'rouge2', 'rougeL'].",
            ),
            TestCaseSummarizationAccuracyInvalidConfig(
                rouge_type="rouge1",
                model_type_for_bertscore="distilbert-base-uncased",
                err_msg="Invalid model_type_for_bertscore: distilbert-base-uncased requested in "
                "SummarizationAccuracyConfig. Please choose from acceptable values: ["
                "'microsoft/deberta-xlarge-mnli', 'roberta-large-mnli'].",
            ),
        ],
    )
    def test_summarization_accuracy_invalid_config(self, rouge_type, model_type_for_bertscore, err_msg):
        """
        GIVEN invalid inputs.
        WHEN a SummarizationAccuracyConfig is initialized.
        THEN an exception with the correct error message is raised.
        """
        with pytest.raises(EvalAlgorithmClientError, match=re.escape(err_msg)):
            SummarizationAccuracyConfig(rouge_type=rouge_type, model_type_for_bertscore=model_type_for_bertscore)

    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreHelperModel")
    def test_evaluate_sample(self, bertscore_model_cls):
        """
        GIVEN valid inputs.
        WHEN SummarizationAccuracy.evaluate_sample is called.
        THEN the correct list of EvalScores is returned.
        """
        # Mock the BertscoreHelperModel class so that the actual model doesn't get loaded into memory.
        bertscore_model_instance = Mock(spec=BertscoreHelperModel)
        bertscore_model_instance.get_helper_scores = Mock(return_value=BERTSCORE_DUMMY_VALUE)
        bertscore_model_cls.return_value = bertscore_model_instance

        model_output = "Berlin: Art, Heritage, Exhibitions Hub."
        target_output = "Berlin: an art metropolis."
        expected_response = [
            EvalScore(name=METEOR_SCORE, value=0.5009920634920636),
            EvalScore(name=ROUGE_SCORE, value=0.4444444444444445),
            EvalScore(name=BERT_SCORE, value=BERTSCORE_DUMMY_VALUE),
        ]
        config = SummarizationAccuracyConfig(rouge_type=ROUGE_L)
        eval_algorithm = SummarizationAccuracy(config)
        actual_response = eval_algorithm.evaluate_sample(target_output, model_output)
        for actual_eval_score, expected_eval_score in zip(actual_response, expected_response):
            assert actual_eval_score.name == expected_eval_score.name
            assert actual_eval_score.value == pytest.approx(expected_eval_score.value, rel=1e-5)

    class TestCaseEvaluate(NamedTuple):
        user_provided_prompt_template: Optional[str]
        dataset_prompt_template: str

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCaseEvaluate(
                user_provided_prompt_template="Summarize $model_input, please.",
                dataset_prompt_template="Summarize $model_input, please.",
            ),
            TestCaseEvaluate(
                user_provided_prompt_template=None,
                dataset_prompt_template=None,
            ),
        ],
    )
    @patch("fmeval.eval_algorithms.summarization_accuracy.get_eval_results_path")
    @patch("fmeval.eval_algorithms.summarization_accuracy.cleanup_shared_resource")
    @patch("fmeval.eval_algorithms.summarization_accuracy.evaluate_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy.create_shared_resource")
    @patch("fmeval.eval_algorithms.summarization_accuracy.TransformPipeline")
    @patch("fmeval.eval_algorithms.summarization_accuracy.SummarizationAccuracy._create_transforms")
    @patch("fmeval.eval_algorithms.summarization_accuracy.get_dataset")
    @patch("fmeval.eval_algorithms.summarization_accuracy.get_dataset_configs")
    def test_evaluate(
        self,
        mock_get_dataset_configs,
        mock_get_dataset,
        mock_create_transforms,
        mock_transform_pipeline_cls,
        mock_create_shared_resource,
        mock_evaluate_dataset,
        mock_cleanup_shared_resource,
        mock_get_results_path,
        test_case,
    ):
        """
        GIVEN a SummarizationAccuracy instance.
        WHEN its evaluate method is called with valid arguments.
        THEN a new TransformPipeline that uses a BertscoreHelperModel shared resource
            is created, and `evaluate_dataset` is called with the correct arguments.
        """
        # The transforms that are saved as instance attributes of the SummarizationAccuracy instance
        meteor_score, rouge_score, bert_score = Mock(), Mock(), Mock()
        # The transforms that get used in the pipeline that gets executed by `evaluate`.
        pipeline_meteor, pipeline_rouge, pipeline_bertscore = Mock(), Mock(), Mock()

        mock_create_transforms.side_effect = [
            (meteor_score, rouge_score, bert_score),
            (pipeline_meteor, pipeline_rouge, pipeline_bertscore),
        ]

        instance_pipeline = Mock()  # The self.pipeline of the SummarizationAccuracy instance
        executed_pipeline = Mock()  # The pipeline that gets created and executed in `evaluate`
        mock_transform_pipeline_cls.side_effect = [instance_pipeline, executed_pipeline]

        mock_get_results_path.return_value = "/path/to/results"
        model_runner = Mock()

        dataset_config = Mock()
        dataset_config.dataset_name = "my_custom_dataset"
        mock_get_dataset_configs.return_value = [dataset_config]

        mock_dataset = Mock()
        # So that validate_dataset does not error
        mock_dataset.columns = Mock(
            return_value=[DatasetColumns.MODEL_INPUT.value.name, DatasetColumns.TARGET_OUTPUT.value.name]
        )
        mock_get_dataset.return_value = mock_dataset

        summ_acc = SummarizationAccuracy()
        output = summ_acc.evaluate(
            model=model_runner,
            dataset_config=dataset_config,
            prompt_template=test_case.user_provided_prompt_template,
            num_records=162,
            save=True,
        )

        mock_create_shared_resource.assert_called_once_with(summ_acc.bertscore_model)
        assert mock_create_transforms.call_count == 2  # once during initialization, once during evaluate
        mock_transform_pipeline_cls.assert_has_calls(
            [call([meteor_score, rouge_score, bert_score]), call([pipeline_meteor, pipeline_rouge, pipeline_bertscore])]
        )
        mock_evaluate_dataset.assert_called_once_with(
            dataset=mock_dataset,
            pipeline=executed_pipeline,
            dataset_name=dataset_config.dataset_name,
            eval_name=summ_acc.eval_name,
            metric_names=METRIC_NAMES,
            eval_results_path="/path/to/results",
            model=model_runner,
            prompt_template=test_case.dataset_prompt_template,
            agg_method=MEAN,
            save=True,
        )
        mock_cleanup_shared_resource.assert_called_once_with(mock_create_shared_resource.return_value)
        assert output == [mock_evaluate_dataset.return_value]
        assert summ_acc.pipeline == instance_pipeline
