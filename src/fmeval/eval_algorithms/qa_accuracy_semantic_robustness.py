import logging

from typing import List, Optional, Union
from dataclasses import dataclass

from ray.actor import ActorHandle

from fmeval.constants import (
    DatasetColumns,
    MEAN,
    PREFIX_FOR_DELTA_SCORES,
    BERTSCORE_DEFAULT_MODEL,
)
from fmeval.data_loaders.util import get_dataset
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.common import evaluate_dataset
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModelTypes, BertscoreHelperModel
from fmeval.eval_algorithms.save_strategy import SaveStrategy
from fmeval.eval_algorithms.semantic_robustness_utils import (
    SemanticRobustnessConfig,
    get_perturbation_transform,
    get_model_outputs_from_perturbed_inputs,
)
from fmeval.eval_algorithms.util import validate_dataset, create_model_invocation_pipeline, get_dataset_configs
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface
from fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalOutput,
    EvalScore,
    get_default_prompt_template,
    DEFAULT_PROMPT_TEMPLATE,
)
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.eval_algorithms.qa_accuracy import (
    F1_SCORE,
    EXACT_MATCH_SCORE,
    QUASI_EXACT_MATCH_SCORE,
    PRECISION_OVER_WORDS,
    RECALL_OVER_WORDS,
    BERT_SCORE,
    QAAccuracyScores,
    BertScoreWithDelimiter,
    QA_ACCURACY_SCORE_NAMES,
    QAAccuracy,
)

# from fmeval.transforms.qa_accuracy_metrics import BertScoreWithDelimiter

# Changed SCORE_NAMES to QA_ACCURACY_SCORE_NAMES
from fmeval.transforms.semantic_robustness_metrics import MeanDeltaScores
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.transforms.util import create_output_key
from fmeval.util import require, get_eval_results_path, create_shared_resource, cleanup_shared_resource

DELTA_F1_SCORE = PREFIX_FOR_DELTA_SCORES + F1_SCORE
DELTA_EXACT_MATCH_SCORE = PREFIX_FOR_DELTA_SCORES + EXACT_MATCH_SCORE
DELTA_QUASI_EXACT_MATCH_SCORE = PREFIX_FOR_DELTA_SCORES + QUASI_EXACT_MATCH_SCORE
DELTA_PRECISION_OVER_WORDS = PREFIX_FOR_DELTA_SCORES + PRECISION_OVER_WORDS
DELTA_RECALL_OVER_WORDS = PREFIX_FOR_DELTA_SCORES + RECALL_OVER_WORDS
DELTA_BERT_SCORE = PREFIX_FOR_DELTA_SCORES + BERT_SCORE
DELTA_SCORES = [
    DELTA_F1_SCORE,
    DELTA_EXACT_MATCH_SCORE,
    DELTA_QUASI_EXACT_MATCH_SCORE,
    DELTA_PRECISION_OVER_WORDS,
    DELTA_RECALL_OVER_WORDS,
    DELTA_BERT_SCORE,
]
ORIGINAL_SCORES = [
    F1_SCORE,
    EXACT_MATCH_SCORE,
    QUASI_EXACT_MATCH_SCORE,
    PRECISION_OVER_WORDS,
    RECALL_OVER_WORDS,
    BERT_SCORE,
]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QAAccuracySemanticRobustnessConfig(SemanticRobustnessConfig):
    """Configures the QA Accuracy Semantic Robustness evaluation algorithm.

    See SemanticRobustnessConfig for the configurable parameters that this config class inherits.

    :param target_output_delimiter: Target Output can have multiple answers. We expect customer to combine all the
        possible answers into a single string and use the delimiter to separate them. For instance,
        if the answers are ["UK", "England"] and the delimiter="<OR>", then the target_output should be "UK<OR>England".
    :param model_type_for_bertscore: BERT model type to use for computing BERT score.
    """

    target_output_delimiter: Optional[str] = "<OR>"
    model_type_for_bertscore: str = BERTSCORE_DEFAULT_MODEL

    def __post_init__(self):
        super().__post_init__()
        require(
            self.target_output_delimiter != "",
            "Empty target_output_delimiter is provided. "
            "Please either provide a non-empty string, or set it to None.",
        )
        require(
            BertscoreHelperModelTypes.model_is_allowed(self.model_type_for_bertscore),
            f"Invalid model_type_for_bertscore: {self.model_type_for_bertscore} requested in "
            f"QAAccuracyConfig. Please choose from acceptable values: "
            f"{BertscoreHelperModelTypes.model_list()}.",
        )


class QAAccuracySemanticRobustness(EvalAlgorithmInterface):
    """Semantic Robustness evaluation algorithm for QA Accuracy

    This evaluation measures how much QA Accuracy changes as a result of semantic preserving
    perturbations on the input. For example, if we apply the whitespace perturbation (adding extra whitepaces at random) to the input text,
    how much does the quality of the model answer change.

    The output difference is measured by computing the QA Accuracy metrics before after perturbing the inputs. We report the absolute value of the difference in scores
    on average over N (`num_perturbations`) perturbed inputs: $$ \frac{1}{P} \sum_{i=1}^{P} |s - \bar{s}_i|,$$
    where $s$ is the score produced by the original metric (i.e., exact match, quasi-exact match, precision over words, recall over words and F1 over words), and $\bar{s_i}$ is the metric evaluated after the i-th perturbation has been applied.

    For details on the QA Accuracy metrics, see the QA Accuracy evaluation. For details on perturbations, see the GeneralSemanticRobustness evaluation.
    """

    eval_name = EvalAlgorithm.QA_ACCURACY_SEMANTIC_ROBUSTNESS.value

    def __init__(
        self, eval_algorithm_config: QAAccuracySemanticRobustnessConfig = QAAccuracySemanticRobustnessConfig()
    ):
        """QAAccuracySemanticRobustness initializer.

        :param eval_algorithm_config: QA Accuracy Semantic Robustness evaluation algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.config = eval_algorithm_config
        self.perturbation_transform = get_perturbation_transform(eval_algorithm_config)
        self.target_output_delimiter = eval_algorithm_config.target_output_delimiter
        bertscore_model = BertscoreHelperModel(eval_algorithm_config.model_type_for_bertscore)
        self.bertscore_model = bertscore_model

    def _build_pipeline(
        self,
        model: ModelRunner,
        prompt_template: str,
        bertscore_model: Union[BertscoreHelperModel, ActorHandle],
    ) -> TransformPipeline:
        """Build the TransformPipeline to be used by `evaluate` and `evaluate_sample`.

        While other evaluation algorithms (ex: QA Accuracy) can configure
        their TransformPipeline at algorithm initialization, because the QA Accuracy
        Semantic Robustness algorithm's evaluation logic depends on the ModelRunner
        and prompt template that are evaluation-specific (i.e. these parameters aren't
        configured at the algorithm level), the pipeline used by this algorithm is built
        when `evaluate` or `evaluate_sample` is called.

        :param model: The ModelRunner representing the model under evaluation.
        :param prompt_template: A template that is used to construct the prompt fed to the model.
        :param bertscore_model: Either a BertscoreHelperModel instance or a Ray actor handle corresponding
            to a BertscoreHelperModel (i.e. a shared resource).
        :returns: A TransformPipeline that can be used by either `evaluate_sample` or `evaluate`.
        """
        transforms = get_model_outputs_from_perturbed_inputs(
            self.perturbation_transform,
            prompt_template,
            model,
        )
        get_perturbed_inputs, gen_perturbed_prompts, get_perturbed_outputs = transforms

        # Scores for the original QAAccuracy metrics (exact match, f1, precision, recall, etc)
        original_scores = QAAccuracyScores(target_output_delimiter=self.target_output_delimiter)

        # Perturbed scores for the original QAAccuracy metrics
        perturbed_scores = [
            QAAccuracyScores(
                model_output_key=perturbed_output_key,
                target_output_delimiter=self.target_output_delimiter,
                output_keys=[
                    create_output_key(score_name, perturbed_output_key) for score_name in QA_ACCURACY_SCORE_NAMES
                ],
            )
            for perturbed_output_key in get_perturbed_outputs.output_keys
        ]

        # bert score (with the target output delimiter)
        bert_score = BertScoreWithDelimiter(
            target_output_keys=[DatasetColumns.TARGET_OUTPUT.value.name],
            model_output_keys=[DatasetColumns.MODEL_OUTPUT.value.name],
            output_keys=[BERT_SCORE],
            allow_duplicate_input_keys=True,
            bertscore_model=bertscore_model,
            target_output_delimiter=self.target_output_delimiter,
        )

        perturbed_bert_score = BertScoreWithDelimiter(
            target_output_keys=[DatasetColumns.TARGET_OUTPUT.value.name for _ in range(self.config.num_perturbations)],
            model_output_keys=get_perturbed_outputs.output_keys,
            output_keys=[
                create_output_key(BertScoreWithDelimiter.__name__, "perturbed", i)
                for i in range(self.config.num_perturbations)
            ],
            allow_duplicate_input_keys=True,
            bertscore_model=bertscore_model,
            target_output_delimiter=self.target_output_delimiter,
        )

        # key mapping for the QA accuracy scores
        key_mapping = {
            original_score_name: (
                [perturbed_score_transform.output_keys[i] for perturbed_score_transform in perturbed_scores],
                DELTA_SCORES[i],
            )
            for i, original_score_name in enumerate(QA_ACCURACY_SCORE_NAMES)
        }

        mean_delta_scores = MeanDeltaScores(key_mapping)

        # key mapping + mean of bert scores
        mean_delta_bert_scores = MeanDeltaScores(
            {
                bert_score.output_keys[0]: (perturbed_bert_score.output_keys, DELTA_BERT_SCORE),
            }
        )

        transforms = [
            get_perturbed_inputs,
            gen_perturbed_prompts,
            get_perturbed_outputs,
            original_scores,
            bert_score,
            TransformPipeline(perturbed_scores),
            perturbed_bert_score,
            mean_delta_scores,
            mean_delta_bert_scores,
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
        """Compute question answering accuracy semantic robustness metrics for a single sample.

        A sample is defined as a model input and target output pair.

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
        compute_metrics = self._build_pipeline(model, prompt_template, self.bertscore_model)
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
        dataset_config: Optional[Union[DataConfig, List[DataConfig]]] = None,
        prompt_template: Optional[str] = None,
        num_records: int = 100,
        save: bool = False,
        save_strategy: Optional[SaveStrategy] = None,
    ) -> List[EvalOutput]:
        """Compute QA accuracy semantic robustness metrics on one or more datasets.

        :param model: An instance of ModelRunner representing the model under evaluation.
            This is a required argument, as even if the dataset contains model outputs,
            semantic robustness algorithms rely on invoking a model on perturbed inputs
            to see how the model outputs from the perturbed inputs differ from the original
            model outputs.
        :param dataset_config: Configures a single dataset or list of datasets used for the
            evaluation. If not provided, this method will run evaluations using all of its
            supported built-in datasets.
        :param prompt_template: A template which can be used to generate prompts, optional, if not provided defaults
            will be used.
        :param num_records: The number of records to be sampled randomly from the input dataset to perform the
                            evaluation
        :param save: If set to true, prompt responses and scores will be saved to a file.
        :param save_strategy: Specifies the strategy to use the save the localized outputs of the evaluations. If not
            specified, it will save it to the path that can be configured by the EVAL_RESULTS_PATH environment variable.
            If that environment variable is also not configured, it will be saved to the default path `/tmp/eval_results/`.
        :returns: A List of EvalOutput objects.
        """
        # Create a shared resource to be used during the evaluation.
        bertscore_shared_resource = create_shared_resource(self.bertscore_model)

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
                pipeline=self._build_pipeline(model, dataset_prompt_template, bertscore_shared_resource),
                dataset_name=dataset_config.dataset_name,
                eval_name=self.eval_name,
                metric_names=ORIGINAL_SCORES + DELTA_SCORES,
                eval_results_path=get_eval_results_path(),
                model=model,
                prompt_template=dataset_prompt_template,
                agg_method=MEAN,
                save=save,
                save_strategy=save_strategy,
            )
            eval_outputs.append(eval_output)
        cleanup_shared_resource(bertscore_shared_resource)
        return eval_outputs
