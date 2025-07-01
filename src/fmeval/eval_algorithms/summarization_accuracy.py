import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from ray.actor import ActorHandle

from fmeval.data_loaders.util import get_dataset
from fmeval.eval_algorithms import EvalAlgorithm, EvalOutput, EvalScore
from fmeval.eval_algorithms.common import evaluate_dataset
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface, EvalAlgorithmConfig
from fmeval.eval_algorithms.save_strategy import SaveStrategy
from fmeval.eval_algorithms.util import get_dataset_configs, validate_dataset
from fmeval.util import (
    assert_condition,
    require,
    create_shared_resource,
    get_eval_results_path,
    cleanup_shared_resource,
)
from fmeval.constants import BERTSCORE_DEFAULT_MODEL, DatasetColumns, MEAN
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModelTypes, BertscoreHelperModel
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.summarization_accuracy_metrics import (
    MeteorScore,
    RougeScore,
    BertScore,
    METEOR_SCORE,
    ROUGE_SCORE,
    BERT_SCORE,
    ROUGE_2,
    ROUGE_TYPES,
)


METRIC_NAMES = [METEOR_SCORE, ROUGE_SCORE, BERT_SCORE]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SummarizationAccuracyConfig(EvalAlgorithmConfig):
    """Configures the summarization accuracy evaluation algorithm.

    :param rouge_type: ROUGE metric type.
    :param use_stemmer_for_rouge: Whether to use stemmer when computing ROUGE metric.
    :param model_type_for_bertscore: BERT model type to use for computing BERT score.
    """

    rouge_type: str = ROUGE_2
    use_stemmer_for_rouge: bool = True
    model_type_for_bertscore: str = BERTSCORE_DEFAULT_MODEL

    def __post_init__(self):
        require(
            self.rouge_type in ROUGE_TYPES,
            f"Invalid rouge_type: {self.rouge_type} requested in SummarizationAccuracyConfig. "
            f"Please choose from acceptable values: {ROUGE_TYPES}.",
        )
        require(
            BertscoreHelperModelTypes.model_is_allowed(self.model_type_for_bertscore),
            f"Invalid model_type_for_bertscore: {self.model_type_for_bertscore} requested in "
            f"SummarizationAccuracyConfig. Please choose from acceptable values: "
            f"{BertscoreHelperModelTypes.model_list()}.",
        )


class SummarizationAccuracy(EvalAlgorithmInterface):
    """Summarization Accuracy evaluation algorithm.

    This evaluation measures how accurately a model can summarize text. By default, we carry out this evaluation by benchmarking on two built-in datasets containing pairs of input text and target summary. The model summaries are then compared to the target summaries using three built-in metrics that measure how similar the summaries are in different ways:

    1. ROUGE-N: ROUGE scores are a class of metrics that compute N-gram word overlaps between reference and model summary.  The metrics are case insensitive and the values are in the range of 0 (no match) to 1 (perfect match). It has the following configurable parameters which can be set in the `SummarizationAccuracyConfig`:
        * N: the length of N-grams to be matched. The three supported values are
            *  N=1 matches single words (unigrams)
            *  N=2 (default) matches word pairs (bigrams)
            *  N=L matches the longest common subsequence.  For computing the longest common subsequence, order is accounted for, but consecutiveness is discounted. E.g., for model summary = "It is autumn"  and  reference = "It is once again autumn" we have that LCS(prediction, reference)=3.
        * use_stemmer: If True (default), uses [Porter stemmer](https://www.cs.toronto.edu/~frank/csc2501/Readings/R2_Porter/Porter-1980.pdf) to strip word suffices. For example, "raining" → "rain".
    To obtain ROUGE-N, N-gram precision and recall are computed. Those are then aggregated into the final score:
    ROUGE-N = 2 * (precision_N * recall_N) / (precision_N + recall_N).

    2. [Meteor](https://aclanthology.org/W05-0909.pdf) is similar to ROUGE-1, but includes stemming (with Porter stemmer) and synonym matching via synonym lists (e.g. “fall” → “autumn”).  The words that are matched by the Meteor score are marked in yellow above. Because Meteor can match synonyms, it is more flexible to paraphrasing than ROUGE.
    2. [BERTScore](https://arxiv.org/pdf/1904.09675.pdf) uses a second ML model (from the BERT family) to compute sentence embeddings and compare their cosine similarity. This score may account for additional linguistic flexibility over ROUGE and METEOR since semantically similar sentences should be embedded closer to each other.

    Parameters which can be set in the `SummarizationAccuracyConfig` are:
    * model_name: Name of the model to be used for scoring, choose one of "microsoft/deberta-xlarge-mnli"  (default) and “roberta-large-mnli" .


    """

    eval_name = EvalAlgorithm.SUMMARIZATION_ACCURACY.value

    def __init__(self, eval_algorithm_config: SummarizationAccuracyConfig = SummarizationAccuracyConfig()):
        """SummarizationAccuracy initializer.

        :param eval_algorithm_config: Summarization Accuracy evaluation algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.bertscore_model = BertscoreHelperModel(eval_algorithm_config.model_type_for_bertscore)
        meteor_score, rouge_score, bert_score = SummarizationAccuracy._create_transforms(
            target_output_keys=[DatasetColumns.TARGET_OUTPUT.value.name],
            model_output_keys=[DatasetColumns.MODEL_OUTPUT.value.name],
            meteor_keys=[METEOR_SCORE],
            rouge_keys=[ROUGE_SCORE],
            bertscore_keys=[BERT_SCORE],
            rouge_type=eval_algorithm_config.rouge_type,
            use_stemmer_for_rouge=eval_algorithm_config.use_stemmer_for_rouge,
            bertscore_model=self.bertscore_model,
        )
        self.meteor_score = meteor_score
        self.rouge_score = rouge_score
        self.bert_score = bert_score
        self.pipeline = TransformPipeline([meteor_score, rouge_score, bert_score])

    @staticmethod
    def _create_transforms(
        target_output_keys: List[str],
        model_output_keys: List[str],
        meteor_keys: List[str],
        rouge_keys: List[str],
        bertscore_keys: List[str],
        rouge_type: str,
        use_stemmer_for_rouge: bool,
        bertscore_model: Union[BertscoreHelperModel, ActorHandle],
    ) -> Tuple[MeteorScore, RougeScore, BertScore]:
        """Create a TransformPipeline containing summarization accuracy score transforms.

        :param target_output_keys: See the corresponding parameter in MeteorScore, RougeScore, and BertScore.
        :param model_output_keys: See the corresponding parameter in MeteorScore, RougeScore, and BertScore.
        :param meteor_keys: The `output_keys` parameter for the returned MeteorScore instance.
        :param rouge_keys: The `output_keys` parameter for the returned RougeScore instance.
        :param bertscore_keys: The `output_keys` parameter for the returned BertScore instance.
        :param rouge_type: See the corresponding parameter in RougeScore.
        :param use_stemmer_for_rouge: See `use_stemmer` in RougeScore.
        :param bertscore_model: A BertscoreHelperModel or Ray actor handle corresponding to a BertscoreHelperModel
            (i.e. a shared resource) used in the creation of the returned BertScore instance.
        :returns: A tuple containing the created MeteorScore, RougeScore, and BertScore instances.
        """
        meteor_transform = MeteorScore(
            target_output_keys=target_output_keys,
            model_output_keys=model_output_keys,
            output_keys=meteor_keys,
            allow_duplicate_input_keys=True,
        )
        rouge_transform = RougeScore(
            target_output_keys=target_output_keys,
            model_output_keys=model_output_keys,
            output_keys=rouge_keys,
            allow_duplicate_input_keys=True,
            rouge_type=rouge_type,
            use_stemmer=use_stemmer_for_rouge,
        )
        bert_transform = BertScore(
            target_output_keys=target_output_keys,
            model_output_keys=model_output_keys,
            output_keys=bertscore_keys,
            allow_duplicate_input_keys=True,
            bertscore_model=bertscore_model,
        )
        return meteor_transform, rouge_transform, bert_transform

    def evaluate_sample(self, target_output: str, model_output: str) -> List[EvalScore]:  # type: ignore[override]
        """Compute summarization accuracy metrics for a single sample.

        :param target_output: The expected/desired model output.
        :param model_output: The actual model output.
        :returns: A list of EvalScore objects, one for each of the summarization accuracy metrics.
        """
        sample = {
            DatasetColumns.TARGET_OUTPUT.value.name: target_output,
            DatasetColumns.MODEL_OUTPUT.value.name: model_output,
        }
        output_record = self.pipeline.execute_record(sample)
        assert_condition(
            all(metric_name in output_record for metric_name in METRIC_NAMES),
            "Summarization Accuracy evaluate_sample has computed an output that is missing at least one metric. "
            f"The output record is {output_record}.",
        )
        return [EvalScore(name=metric_name, value=output_record[metric_name]) for metric_name in METRIC_NAMES]

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[Union[DataConfig, List[DataConfig]]] = None,
        prompt_template: Optional[str] = None,
        num_records: int = 100,
        save: bool = False,
        save_strategy: Optional[SaveStrategy] = None,
    ) -> List[EvalOutput]:
        """Compute summarization accuracy metrics on one or more datasets.

        :param model: An instance of ModelRunner representing the model under evaluation.
            If this argument is None, the `dataset_config` argument must not be None,
            and must correspond to a dataset that already contains a column with model outputs.
        :param dataset_config: Configures a single dataset or list of datasets used for the
            evaluation. If not provided, this method will run evaluations using all of its
            supported built-in datasets.
        :param prompt_template: A template used to generate prompts that are fed to the model.
            If not provided, defaults will be used. If provided, `model` must not be None.
        :param num_records: The number of records to be sampled randomly from the input dataset(s)
            used to perform the evaluation(s).
        :param save: If set to true, prompt responses and scores will be saved to a file.
        :param save_strategy: Specifies the strategy to use the save the localized outputs of the evaluations. If not
            specified, it will save it to the path that can be configured by the EVAL_RESULTS_PATH environment variable.
            If that environment variable is also not configured, it will be saved to the default path `/tmp/eval_results/`.

        :return: A list of EvalOutput objects.
        """
        # Create a shared resource to be used during the evaluation.
        bertscore_shared_resource = create_shared_resource(self.bertscore_model)
        # Create a new pipeline that uses the shared resource instead of self.bertscore_model.
        meteor_score, rouge_score, bert_score = SummarizationAccuracy._create_transforms(
            target_output_keys=[DatasetColumns.TARGET_OUTPUT.value.name],
            model_output_keys=[DatasetColumns.MODEL_OUTPUT.value.name],
            meteor_keys=[METEOR_SCORE],
            rouge_keys=[ROUGE_SCORE],
            bertscore_keys=[BERT_SCORE],
            rouge_type=self.rouge_score.rouge_type,
            use_stemmer_for_rouge=self.rouge_score.use_stemmer,
            bertscore_model=bertscore_shared_resource,
        )
        pipeline = TransformPipeline([meteor_score, rouge_score, bert_score])

        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [DatasetColumns.MODEL_INPUT.value.name, DatasetColumns.TARGET_OUTPUT.value.name])
            eval_output = evaluate_dataset(
                dataset=dataset,
                pipeline=pipeline,
                dataset_name=dataset_config.dataset_name,
                eval_name=self.eval_name,
                metric_names=METRIC_NAMES,
                eval_results_path=get_eval_results_path(),
                model=model,
                prompt_template=prompt_template,
                agg_method=MEAN,
                save=save,
                save_strategy=save_strategy,
            )
            eval_outputs.append(eval_output)

        cleanup_shared_resource(bertscore_shared_resource)
        return eval_outputs
