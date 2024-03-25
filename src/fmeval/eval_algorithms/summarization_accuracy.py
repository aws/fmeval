import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from fmeval.perf_util import timed_block
from fmeval.transforms.common import GeneratePrompt, GetModelOutputs
from fmeval.util import require
from fmeval.constants import BERTSCORE_DEFAULT_MODEL, DatasetColumns, MEAN
from fmeval.data_loaders.data_config import DataConfig
from fmeval.data_loaders.util import get_dataset
from fmeval.eval_algorithms import EvalAlgorithm, EvalScore, EvalOutput, get_default_prompt_template
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmConfig, EvalAlgorithmInterface
from fmeval.eval_algorithms.util import (
    validate_dataset,
    aggregate_evaluation_scores,
    generate_output_dataset_path,
    save_dataset,
    get_dataset_configs,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.helper_models import BertscoreModelTypes, BertscoreModel
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

from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.util import create_shared_resource, assert_condition

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
        if self.rouge_type not in ROUGE_TYPES:
            raise EvalAlgorithmClientError(
                f"Invalid rouge_type: {self.rouge_type} requested in SummarizationAccuracyConfig. "
                f"Please choose from acceptable values: {ROUGE_TYPES}."
            )

        if not BertscoreModelTypes.model_is_allowed(self.model_type_for_bertscore):
            raise EvalAlgorithmClientError(
                f"Invalid model_type_for_bertscore: {self.model_type_for_bertscore} requested in "
                f"SummarizationAccuracyConfig. Please choose from acceptable values: "
                f"{BertscoreModelTypes.model_list()}."
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

    def __init__(
        self, eval_algorithm_config: SummarizationAccuracyConfig = SummarizationAccuracyConfig(), use_ray: bool = True
    ):
        """SummarizationAccuracy initializer.

        :param eval_algorithm_config: Summarization Accuracy evaluation algorithm config.
        :param use_ray: Whether to use the Ray distributed computing framework to run the evaluation
            algorithm logic. While using Ray will typically speed up the evaluation, setting this flag
            to False can be useful for debugging purposes or in situations where computational resources
            are limited.
        """
        super().__init__(eval_algorithm_config)
        meteor_transform = MeteorScore(
            target_output_keys=[DatasetColumns.TARGET_OUTPUT.value.name],
            model_output_keys=[DatasetColumns.MODEL_OUTPUT.value.name],
            output_keys=[METEOR_SCORE],
            allow_duplicate_input_keys=False,
        )
        rouge_transform = RougeScore(
            target_output_keys=[DatasetColumns.TARGET_OUTPUT.value.name],
            model_output_keys=[DatasetColumns.MODEL_OUTPUT.value.name],
            output_keys=[ROUGE_SCORE],
            allow_duplicate_input_keys=False,
            rouge_type=eval_algorithm_config.rouge_type,
            use_stemmer=eval_algorithm_config.use_stemmer_for_rouge,
        )
        bertscore_model = BertscoreModel(eval_algorithm_config.model_type_for_bertscore)
        if use_ray:
            bertscore_model = create_shared_resource(bertscore_model)
        bert_transform = BertScore(
            target_output_keys=[DatasetColumns.TARGET_OUTPUT.value.name],
            model_output_keys=[DatasetColumns.MODEL_OUTPUT.value.name],
            output_keys=[BERT_SCORE],
            allow_duplicate_input_keys=False,
            bertscore_model=bertscore_model,
        )
        self.pipeline = TransformPipeline([meteor_transform, rouge_transform, bert_transform])

    @staticmethod
    def create_sample(target_output: str, model_output: str) -> Dict[str, Any]:
        """Create a sample in the record format used by Transforms.

        This function's primary use is to be called by evaluate_sample.

        :param target_output: The target_output parameter passed to evaluate_sample.
        :param model_output: The model_output parameter passed to evaluate_sample.
        """
        return {
            DatasetColumns.TARGET_OUTPUT.value.name: target_output,
            DatasetColumns.MODEL_OUTPUT.value.name: model_output,
        }

    def evaluate_sample(self, target_output: str, model_output: str) -> List[EvalScore]:  # type: ignore[override]
        """Compute summarization accuracy metrics for a single sample.

        :param target_output: The expected/desired model output.
        :param model_output: The actual model output.
        :returns: A list of EvalScore objects, one for each of the summarization accuracy metrics.
        """
        sample = SummarizationAccuracy.create_sample(target_output=target_output, model_output=model_output)
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
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        save: bool = False,
        num_records=100,
    ) -> List[EvalOutput]:
        """Perform summarization accuracy evaluation on a dataset.

        :param model: An instance of ModelRunner representing the model under evaluation.
        :param dataset_config: Configures the single dataset used for evaluation.
            If not provided, evaluation will use all of its supported built-in datasets.
        :param prompt_template: A template used to generate prompts.
            If not provided, defaults will be used.
        :param save: If set to true, prompt responses and scores will be saved to a file.
            The output is written to EvalAlgorithmInterface.EVAL_RESULTS_PATH.
        :param num_records: The number of records to be sampled randomly from the input dataset
            to perform the evaluation
        :return: A list of EvalOutput objects.
        """
        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [DatasetColumns.TARGET_OUTPUT.value.name, DatasetColumns.MODEL_INPUT.value.name])
            dataset_prompt_template = None
            pipeline = self.pipeline

            if DatasetColumns.MODEL_OUTPUT.value.name not in dataset.columns():
                require(model, "No ModelRunner provided. ModelRunner is required for inference on model_inputs")
                dataset_prompt_template = (
                    get_default_prompt_template(dataset_config.dataset_name) if not prompt_template else prompt_template
                )
                gen_prompt = GeneratePrompt(
                    input_keys=[DatasetColumns.MODEL_INPUT.value.name],
                    output_keys=[DatasetColumns.PROMPT.value.name],
                    prompt_template=dataset_prompt_template,
                )
                get_model_outputs = GetModelOutputs(
                    input_to_output_keys={DatasetColumns.PROMPT.value.name: [DatasetColumns.MODEL_OUTPUT.value.name]},
                    model_runner=model,
                )
                pipeline = TransformPipeline([gen_prompt, get_model_outputs, pipeline])

            with timed_block(f"Computing score and aggregation on dataset {dataset_config.dataset_name}", logger):
                dataset = pipeline.execute(dataset)
                dataset_scores, category_scores = aggregate_evaluation_scores(dataset, METRIC_NAMES, agg_method=MEAN)
                eval_outputs.append(
                    EvalOutput(
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                        prompt_template=dataset_prompt_template,
                        dataset_scores=dataset_scores,
                        category_scores=category_scores,
                        output_path=generate_output_dataset_path(
                            path_to_parent_dir=self._eval_results_path,
                            eval_name=self.eval_name,
                            dataset_name=dataset_config.dataset_name,
                        ),
                    )
                )
            if save:
                save_dataset(
                    dataset=dataset,
                    score_names=METRIC_NAMES,
                    path=generate_output_dataset_path(
                        path_to_parent_dir=self._eval_results_path,
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                    ),
                )

        return eval_outputs
