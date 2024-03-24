import logging

from dataclasses import dataclass
from typing import List, Optional

from fmeval.constants import (
    DatasetColumns,
    PREFIX_FOR_DELTA_SCORES,
    BERTSCORE_DEFAULT_MODEL,
    MEAN,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.data_loaders.util import get_dataset
from fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalScore,
    EvalOutput,
    DEFAULT_PROMPT_TEMPLATE,
    get_default_prompt_template,
)
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface
from fmeval.eval_algorithms.semantic_robustness_utils import (
    SemanticRobustnessConfig,
    get_perturbation_transform,
    get_model_responses_from_perturbed_inputs,
)
from fmeval.transforms.semantic_robustness_metrics import MeanDeltaScores
from fmeval.eval_algorithms.summarization_accuracy import (
    SummarizationAccuracy,
    ROUGE_TYPES,
    ROUGE_2,
    ROUGE_SCORE,
    METEOR_SCORE,
    BERT_SCORE,
)
from fmeval.helper_models import BertscoreModelTypes, BertscoreModel
from fmeval.eval_algorithms.util import (
    get_dataset_configs,
    evaluate_dataset,
    create_model_invocation_pipeline,
    validate_dataset,
)
from fmeval.transforms.summarization_accuracy_metrics import MeteorScore, RougeScore, BertScore
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.transforms.util import create_output_key
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.util import require, create_shared_resource, get_eval_results_path

logger = logging.getLogger(__name__)

DELTA_ROUGE_SCORE = PREFIX_FOR_DELTA_SCORES + ROUGE_SCORE
DELTA_METEOR_SCORE = PREFIX_FOR_DELTA_SCORES + METEOR_SCORE
DELTA_BERT_SCORE = PREFIX_FOR_DELTA_SCORES + BERT_SCORE
DELTA_SCORES = [DELTA_METEOR_SCORE, DELTA_ROUGE_SCORE, DELTA_BERT_SCORE]
ORIGINAL_SCORES = [METEOR_SCORE, ROUGE_SCORE, BERT_SCORE]


@dataclass(frozen=True)
class SummarizationAccuracySemanticRobustnessConfig(SemanticRobustnessConfig):
    """Configures the summarization accuracy semantic robustness evaluation algorithm.

    See SemanticRobustnessConfig for the configurable parameters that this config class inherits.

    :param rouge_type: ROUGE metric type.
    :param use_stemmer_for_rouge: Whether to use stemmer when computing ROUGE metric.
    :param model_type_for_bertscore: BERT model type to use for computing BERT score.
    """

    rouge_type: str = ROUGE_2
    use_stemmer_for_rouge: bool = True
    model_type_for_bertscore: str = BERTSCORE_DEFAULT_MODEL

    def __post_init__(self):
        super().__post_init__()
        require(
            self.rouge_type in ROUGE_TYPES,
            f"Invalid rouge_type: {self.rouge_type} requested in SummarizationAccuracySemanticRobustnessConfig. "
            f"Please choose from acceptable values: {ROUGE_TYPES}.",
        )
        require(
            BertscoreModelTypes.model_is_allowed(self.model_type_for_bertscore),
            f"Invalid model_type_for_bertscore: {self.model_type_for_bertscore} requested in "
            f"SummarizationAccuracySemanticRobustnessConfig. Please choose from acceptable values: {BertscoreModelTypes.model_list()}.",
        )


class SummarizationAccuracySemanticRobustness(EvalAlgorithmInterface):
    """Semantic Robustness evaluation algorithm for Summarization Accuracy

    This evaluation measures how much Summarization Accuracy changes as a result of semantic preserving
    perturbations on the input. For example, if we apply the whitespace perturbation (adding extra whitepaces at random) to the input text,
    how much does the quality of the model summary change.

    The output difference is measured by computing the Summarization Accuracy metrics before after perturbing the inputs. We report the absolute value of the difference in scores
    on average over N (`num_perturbations`) perturbed inputs: $$ \frac{1}{P} \sum_{i=1}^{P} |s - \bar{s}_i|,$$
    where $s$ is the score produced by the original metric (i.e., ROUGE, METEOR and BERTScore), and $\bar{s_i}$ is the metric evaluated after the i-th perturbation has been applied.

    For details on the Summarization Accuracy metrics, see the Summarization Accuracy evaluation. For details on perturbations, see the GeneralSemanticRobustness evaluation.
    """

    eval_name = EvalAlgorithm.SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS.value

    def __init__(
        self,
        eval_algorithm_config: SummarizationAccuracySemanticRobustnessConfig = SummarizationAccuracySemanticRobustnessConfig(),
        use_ray: bool = True,
    ):
        """SummarizationAccuracySemanticRobustness initializer.

        :param eval_algorithm_config: Summarization accuracy semantic robustness evaluation algorithm config.
        """
        self.config = eval_algorithm_config
        self.perturbation_transform = get_perturbation_transform(eval_algorithm_config)
        bertscore_model = BertscoreModel(eval_algorithm_config.model_type_for_bertscore)
        if use_ray:  # pragma: no branch
            bertscore_model = create_shared_resource(bertscore_model)
        self.bertscore_model = bertscore_model

    def build_pipeline(
        self,
        model: ModelRunner,
        prompt_template: str,
        use_ray: bool,
    ) -> TransformPipeline:
        """Build the TransformPipeline to be used by `evaluate` and `evaluate_sample`.

        While other evaluation algorithms (ex: Summarization Accuracy) can configure
        their TransformPipeline at algorithm initialization, because the Summarization Accuracy
        Semantic Robustness (SASR) algorithm's evaluation logic depends on the ModelRunner
        and prompt template that are evaluation-specific (i.e. these parameters aren't
        configured at the algorithm level), the pipeline used by the SASR algorithm is built
        when `evaluate` or `evaluate_sample` is called.

        :param model: The ModelRunner representing the model under evaluation.
        :param prompt_template: A template that is used to construct the prompt fed to the model.
        :param use_ray: Whether to create a Ray actor for the BertscoreModel used by this evaluation
            algorithm instance. Currently, `evaluate` will only work if `use_ray` is set to True,
            as the execution of the transform pipeline relies on the BertscoreModel existing
            in shared memory. This flag can be set to False if you only plan on invoking the
            `evaluate_sample` method, which is a computationally cheap operation that does not
            require utilizing Ray for parallel execution.
        :returns: A TransformPipeline that can be used by either `evaluate_sample` or `evaluate`.
        """
        transforms = get_model_responses_from_perturbed_inputs(
            self.perturbation_transform,
            prompt_template,
            model,
        )
        get_perturbed_inputs, gen_perturbed_prompts, get_perturbed_responses = transforms

        meteor_score, rouge_score, bert_score, _ = SummarizationAccuracy.build_pipeline(
            target_output_keys=[DatasetColumns.TARGET_OUTPUT.value.name],
            model_output_keys=[DatasetColumns.MODEL_OUTPUT.value.name],
            meteor_keys=[METEOR_SCORE],
            rouge_keys=[ROUGE_SCORE],
            bertscore_keys=[BERT_SCORE],
            rouge_type=self.config.rouge_type,
            use_stemmer_for_rouge=self.config.use_stemmer_for_rouge,
            bertscore_model=self.bertscore_model,
            use_ray=use_ray,
        )

        perturbed_meteor_score, perturbed_rouge_score, perturbed_bert_score, _ = SummarizationAccuracy.build_pipeline(
            target_output_keys=[DatasetColumns.TARGET_OUTPUT.value.name for _ in range(self.config.num_perturbations)],
            model_output_keys=get_perturbed_responses.output_keys,
            meteor_keys=[
                create_output_key(MeteorScore.__name__, "perturbed", i) for i in range(self.config.num_perturbations)
            ],
            rouge_keys=[
                create_output_key(RougeScore.__name__, "perturbed", i) for i in range(self.config.num_perturbations)
            ],
            bertscore_keys=[
                create_output_key(BertScore.__name__, "perturbed", i) for i in range(self.config.num_perturbations)
            ],
            rouge_type=self.config.rouge_type,
            use_stemmer_for_rouge=self.config.use_stemmer_for_rouge,
            bertscore_model=self.bertscore_model,
            use_ray=use_ray,
        )

        delta_meteor_key = DELTA_METEOR_SCORE
        delta_rouge_key = DELTA_ROUGE_SCORE
        delta_bert_key = DELTA_BERT_SCORE
        mean_delta_scores = MeanDeltaScores(
            {
                meteor_score.output_keys[0]: (perturbed_meteor_score.output_keys, delta_meteor_key),
                rouge_score.output_keys[0]: (perturbed_rouge_score.output_keys, delta_rouge_key),
                bert_score.output_keys[0]: (perturbed_bert_score.output_keys, delta_bert_key),
            }
        )

        transforms = [
            get_perturbed_inputs,
            gen_perturbed_prompts,
            get_perturbed_responses,
            meteor_score,
            rouge_score,
            bert_score,
            perturbed_meteor_score,
            perturbed_rouge_score,
            perturbed_bert_score,
            mean_delta_scores,
        ]
        pipeline = TransformPipeline(transforms)
        return pipeline

    def evaluate_sample(
        self,
        model_input: str,
        target_output: str,
        model: ModelRunner,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> List[EvalScore]:
        """Compute summarization accuracy semantic robustness metrics for a single sample.

        A sample is defined as a model input and model output pair.

        :param model_input: Text input, which will be composed into a prompt that gets fed to the model.
        :param target_output: The expected response from the model.
        :param model: An instance of ModelRunner representing the model under evaluation.
        :param prompt_template: A template used to compose the prompt from `model_input`.
        :return: A list of EvalScores.
        """
        sample = {
            DatasetColumns.MODEL_INPUT.value.name: model_input,
            DatasetColumns.TARGET_OUTPUT.value.name: target_output,
        }
        invoke_model = create_model_invocation_pipeline(model, prompt_template)
        compute_metrics = self.build_pipeline(model, prompt_template, use_ray=False)
        pipeline = TransformPipeline([invoke_model, compute_metrics])
        output_record = pipeline.execute_record(sample)

        original_scores = [
            EvalScore(name=score_name, value=output_record[score_name]) for score_name in ORIGINAL_SCORES
        ]
        delta_scores = [
            EvalScore(name=delta_score_name, value=output_record[delta_score_name]) for delta_score_name in DELTA_SCORES
        ]
        return original_scores + delta_scores

    def evaluate(
        self,
        model: ModelRunner,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        num_records: int = 100,
        save: bool = False,
    ) -> List[EvalOutput]:
        """
        Semantic Robustness evaluate.

        :param model: An instance of ModelRunner which is the model under evaluation
        :param dataset_config: Configures the single dataset used for evaluation. If not provided,
            evaluation will use all of it's supported built-in datasets
        :param prompt_template: A template which can be used to generate prompts, optional, if not provided defaults
            will be used.
        :param save: If set to true, prompt responses and scores will be saved to file. The output is written to
                     EvalAlgorithmInterface.EVAL_RESULTS_PATH
        :param num_records: The number of records to be sampled randomly from the input dataset to perform the
                            evaluation
        :return: List of EvalOutput objects.
        """
        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset_prompt_template = (
                get_default_prompt_template(dataset_config.dataset_name) if not prompt_template else prompt_template
            )
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [DatasetColumns.MODEL_INPUT.value.name, DatasetColumns.TARGET_OUTPUT.value.name])
            eval_output = evaluate_dataset(
                dataset=dataset,
                pipeline=self.build_pipeline(model, dataset_prompt_template, use_ray=True),
                dataset_name=dataset_config.dataset_name,
                eval_name=self.eval_name,
                metric_names=ORIGINAL_SCORES + DELTA_SCORES,
                eval_results_path=get_eval_results_path(),
                model=model,
                prompt_template=dataset_prompt_template,
                agg_method=MEAN,
                save=save,
            )
            eval_outputs.append(eval_output)

        return eval_outputs
