import pytest
import re
import ray

from typing import NamedTuple
from unittest.mock import Mock, patch
from _pytest.fixtures import fixture

from fmeval.constants import DatasetColumns, BERTSCORE_DEFAULT_MODEL
from fmeval.eval_algorithms import EvalScore
from fmeval.helper_models import BertscoreModel
from fmeval.transforms.summarization_accuracy_metrics import ROUGE_L, MeteorScore, RougeScore, BertScore
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
        return SummarizationAccuracy(SummarizationAccuracyConfig(), use_ray=False)

    @patch("fmeval.eval_algorithms.summarization_accuracy.TransformPipeline")
    @patch("fmeval.eval_algorithms.summarization_accuracy.SummarizationAccuracy.build_pipeline")
    def test_init(self, mock_build_pipeline, mock_transform_pipeline_cls):
        """
        GIVEN default arguments.
        WHEN a SummarizationAccuracy is initialized.
        THEN SummarizationAccuracy.build_pipeline is called, a TransformPipeline
            is initialized with the correct Transforms, and said pipeline is set
            to the instance's `pipeline` attribute.
        """
        mock_meteor, mock_rouge, mock_bertscore = Mock(), Mock(), Mock()
        mock_build_pipeline.return_value = mock_meteor, mock_rouge, mock_bertscore, Mock()
        summ_acc = SummarizationAccuracy()
        mock_transform_pipeline_cls.assert_called_with([mock_meteor, mock_rouge, mock_bertscore])
        assert summ_acc.pipeline == mock_transform_pipeline_cls.return_value

    @pytest.mark.parametrize("use_ray", [True, False])
    @patch("fmeval.eval_algorithms.summarization_accuracy.create_shared_resource")
    def test_build_pipeline(self, mock_shared_resource, use_ray):
        """
        GIVEN valid arguments where the bertscore_model argument is None.
        WHEN SummarizationAccuracy's build_pipeline method is called.
        THEN the correct outputs are returned, and create_shared_resource
            is called if `use_ray` is True (and not called otherwise).
        """
        meteor_score, rouge_score, bert_score, bertscore_model = SummarizationAccuracy.build_pipeline(
            target_output_keys=["target_output"],
            model_output_keys=["model_output"],
            meteor_keys=[METEOR_SCORE],
            rouge_keys=[ROUGE_SCORE],
            bertscore_keys=[BERT_SCORE],
            rouge_type=ROUGE_L,
            use_stemmer_for_rouge=True,
            bertscore_model_type=BERTSCORE_DEFAULT_MODEL,
            use_ray=use_ray,
        )
        assert isinstance(meteor_score, MeteorScore)
        assert isinstance(rouge_score, RougeScore)
        assert isinstance(bert_score, BertScore)
        if use_ray:
            mock_shared_resource.assert_called_once()
            assert bertscore_model == mock_shared_resource.return_value
        else:
            mock_shared_resource.assert_not_called()
            assert isinstance(bertscore_model, BertscoreModel)

    @patch("fmeval.eval_algorithms.summarization_accuracy.BertscoreModel")
    def test_build_pipeline_with_existing_bertscore_model(self, mock_bertscore_model_cls):
        """
        GIVEN a `bertscore_model` argument that is not None.
        WHEN SummarizationAccuracy's build_pipeline method is called.
        THEN the bertscore_model that is returned by build_pipeline is
            the same object that was passed in.
        """
        bertscore_model_instance = Mock()
        _, _, _, bertscore_model = SummarizationAccuracy.build_pipeline(
            target_output_keys=["target_output"],
            model_output_keys=["model_output"],
            meteor_keys=[METEOR_SCORE],
            rouge_keys=[ROUGE_SCORE],
            bertscore_keys=[BERT_SCORE],
            rouge_type=ROUGE_L,
            use_stemmer_for_rouge=True,
            bertscore_model=bertscore_model_instance,
            use_ray=False,
        )
        assert bertscore_model is bertscore_model_instance
        mock_bertscore_model_cls.assert_not_called()

    def test_build_pipeline_missing_bertscore_model_type(self):
        """
        GIVEN bertscore_model and bertscore_model_type arguments with value None.
        WHEN SummarizationAccuracy's build_pipeline method is called.
        THEN an exception is raised.
        """
        with pytest.raises(
            EvalAlgorithmClientError,
            match="bertscore_model_type must not be None when bertscore_model is not provided.",
        ):
            SummarizationAccuracy.build_pipeline(
                target_output_keys=["target_output"],
                model_output_keys=["model_output"],
                meteor_keys=[METEOR_SCORE],
                rouge_keys=[ROUGE_SCORE],
                bertscore_keys=[BERT_SCORE],
                rouge_type=ROUGE_L,
                use_stemmer_for_rouge=True,
                use_ray=False,
            )

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
                err_msg="Invalid bertscore_model_type: distilbert-base-uncased requested in "
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
            SummarizationAccuracyConfig(rouge_type=rouge_type, bertscore_model_type=bertscore_model_type)

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

    @patch("fmeval.eval_algorithms.summarization_accuracy.get_eval_results_path")
    @patch("fmeval.eval_algorithms.summarization_accuracy.util.evaluate")
    @patch("fmeval.eval_algorithms.summarization_accuracy.TransformPipeline")
    @patch("fmeval.eval_algorithms.summarization_accuracy.SummarizationAccuracy.build_pipeline")
    def test_evaluate(
        self, mock_build_pipeline, mock_transform_pipeline_cls, mock_util_evaluate, mock_get_results_path
    ):
        """
        GIVEN a SummarizationAccuracy instance whose `use_ray` attribute is True.
        WHEN its evaluate method is called with valid arguments.
        THEN `util.evaluate` is called with the correct arguments.
        """
        mock_build_pipeline.return_value = Mock(), Mock(), Mock(), Mock()
        mock_get_results_path.return_value = "/path/to/results"
        model_runner = Mock()
        dataset_config = Mock()

        summ_acc = SummarizationAccuracy()
        output = summ_acc.evaluate(
            model=model_runner,
            dataset_config=dataset_config,
            prompt_template="Summarize $model_input, please.",
            num_records=162,
            save=True,
        )

        mock_util_evaluate.assert_called_once_with(
            eval_name=summ_acc.eval_name,
            pipeline=summ_acc.pipeline,
            metric_names=METRIC_NAMES,
            required_columns=[DatasetColumns.TARGET_OUTPUT.value.name, DatasetColumns.MODEL_INPUT.value.name],
            eval_results_path="/path/to/results",
            model=model_runner,
            dataset_config=dataset_config,
            prompt_template="Summarize $model_input, please.",
            num_records=162,
            save=True,
        )
        assert output == mock_util_evaluate.return_value

    @patch("fmeval.eval_algorithms.summarization_accuracy.TransformPipeline")
    @patch("fmeval.eval_algorithms.summarization_accuracy.SummarizationAccuracy.build_pipeline")
    def test_evaluate_failure(self, mock_build_pipeline, mock_transform_pipeline_cls):
        """
        GIVEN a SummarizationAccuracy instance whose `use_ray` attribute is False.
        WHEN its evaluate method is called.
        THEN an exception is raised.
        """

        mock_build_pipeline.return_value = Mock(), Mock(), Mock(), Mock()
        summ_acc = SummarizationAccuracy(use_ray=False)
        err_msg = (
            "The use_ray instance attribute of SummarizationAccuracy must be True in order "
            "for the evaluate method to run successfully."
        )
        with pytest.raises(EvalAlgorithmClientError, match=err_msg):
            summ_acc.evaluate()
