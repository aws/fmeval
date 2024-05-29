import logging
import warnings

from typing import Callable, List, Optional, Union
from dataclasses import dataclass

from fmeval.constants import (
    DatasetColumns,
    MEAN,
)
from fmeval.data_loaders.util import get_dataset
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.common import evaluate_dataset
from fmeval.eval_algorithms.save_strategy import SaveStrategy
from fmeval.eval_algorithms.semantic_robustness_utils import (
    SemanticRobustnessConfig,
    get_perturbation_transform,
    get_model_outputs_from_perturbed_inputs,
)
from fmeval.eval_algorithms.util import (
    get_dataset_configs,
    validate_dataset,
    create_model_invocation_pipeline,
)
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface
from fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalOutput,
    EvalScore,
    get_default_prompt_template,
    DEFAULT_PROMPT_TEMPLATE,
)
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.eval_algorithms.classification_accuracy import (
    convert_model_output_to_label,
    CLASSIFICATION_ACCURACY_SCORE,
    UNIQUENESS_FACTOR,
    ClassificationAccuracyScores,
    CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME,
)
from fmeval.transforms.semantic_robustness_metrics import MeanDeltaScores
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.util import get_eval_results_path

PREFIX_FOR_DELTA_SCORES = "delta_"
DELTA_CLASSIFICATION_ACCURACY_SCORE = PREFIX_FOR_DELTA_SCORES + CLASSIFICATION_ACCURACY_SCORE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationAccuracySemanticRobustnessConfig(SemanticRobustnessConfig):
    """Configures the Classification Accuracy Semantic Robustness evaluation algorithm.

    See SemanticRobustnessConfig for the configurable parameters that this config class inherits.

    :param valid_labels: A list of valid labels.
    :param converter_fn: Function to process model output to labels. Defaults to simple integer conversion.
    """

    valid_labels: Optional[List[str]] = None
    converter_fn: Callable[[str, List[str]], str] = convert_model_output_to_label

    def __post_init__(self):
        super().__post_init__()
        if self.valid_labels:
            for i, label in enumerate(self.valid_labels):
                if not isinstance(label, str):
                    warnings.warn("Valid labels should be strings, casting.")
                    self.valid_labels[i] = str(label)


class ClassificationAccuracySemanticRobustness(EvalAlgorithmInterface):
    """Semantic Robustness evaluation algorithm for Classification Accuracy

    This evaluation measures how much Classification Accuracy changes as a result of semantic preserving
    perturbations on the input. For example, if we apply the whitespace perturbation (adding extra whitepaces at random) to the input text,
    how much does this impact the ability of the model to correctly classify this text.

    The output difference is measured by computing the Classification Accuracy metrics before after perturbing the inputs. We report the absolute value of the difference in scores
    on average over N (`num_perturbations`) perturbed inputs: $$ \frac{1}{P} \sum_{i=1}^{P} |s - \bar{s}_i|,$$
    where $s$ is the score produced by the original metric (i.e., accuracy, precision, recall and balanced accuracy), and $\bar{s_i}$ is the metric evaluated after the i-th perturbation has been applied.

    For details on the Classification Accuracy metrics, see the Classification Accuracy evaluation. For details on perturbations, see the GeneralSemanticRobustness evaluation.
    """

    eval_name = EvalAlgorithm.CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS.value

    def __init__(
        self,
        eval_algorithm_config: ClassificationAccuracySemanticRobustnessConfig = ClassificationAccuracySemanticRobustnessConfig(),
    ):
        """ClassificationAccuracySemanticRobustness initializer.

        :param eval_algorithm_config: Classification Accuracy Semantic Robustness evaluation algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.config = eval_algorithm_config
        self.perturbation_transform = get_perturbation_transform(eval_algorithm_config)
        self.valid_labels = eval_algorithm_config.valid_labels
        self.converter_fn = eval_algorithm_config.converter_fn

    def _build_pipeline(
        self,
        model: ModelRunner,
        prompt_template: str,
        valid_labels: Optional[List[str]],
    ) -> TransformPipeline:
        """Build the TransformPipeline to be used by `evaluate` and `evaluate_sample`.

        While other evaluation algorithms (ex: Classification Accuracy) can configure
        their TransformPipeline at algorithm initialization, because the Classification Accuracy
        Semantic Robustness algorithm's evaluation logic depends on the ModelRunner
        and prompt template that are evaluation-specific (i.e. these parameters aren't
        configured at the algorithm level), the pipeline used by this algorithm is built
        when `evaluate` or `evaluate_sample` is called.

        :param model: The ModelRunner representing the model under evaluation.
        :param prompt_template: A template that is used to construct the prompt fed to the model.
        :param valid_labels: A list of valid labels for the classified model output.
        :returns: A TransformPipeline that can be used by either `evaluate_sample` or `evaluate`.
        """
        get_perturbed_inputs, gen_perturbed_prompts, get_perturbed_outputs = get_model_outputs_from_perturbed_inputs(
            self.perturbation_transform,
            prompt_template,
            model,
        )

        original_scores = ClassificationAccuracyScores(valid_labels=valid_labels, converter_fn=self.converter_fn)
        perturbed_scores = [
            ClassificationAccuracyScores(
                valid_labels=valid_labels,
                model_output_key=perturbed_output_key,
                classified_model_output_key=f"{CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME}_perturbed_{i}",
                classification_accuracy_score_key=f"{CLASSIFICATION_ACCURACY_SCORE}_perturbed_{i}",
                converter_fn=self.converter_fn,
            )
            for i, perturbed_output_key in enumerate(get_perturbed_outputs.output_keys)
        ]

        perturbed_score_keys = [
            perturbed_score_transform.classification_accuracy_score_key
            for perturbed_score_transform in perturbed_scores
        ]
        mean_delta_scores = MeanDeltaScores(
            {CLASSIFICATION_ACCURACY_SCORE: (perturbed_score_keys, DELTA_CLASSIFICATION_ACCURACY_SCORE)}
        )

        transforms = [
            get_perturbed_inputs,
            gen_perturbed_prompts,
            get_perturbed_outputs,
            original_scores,
            TransformPipeline(perturbed_scores),
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
        """Compute classification accuracy semantic robustness metrics for a single sample.

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
        compute_metrics = self._build_pipeline(model, prompt_template, self.valid_labels)
        pipeline = TransformPipeline([invoke_model, compute_metrics])
        output_record = pipeline.execute_record(sample)

        original_score = EvalScore(
            name=CLASSIFICATION_ACCURACY_SCORE, value=output_record[CLASSIFICATION_ACCURACY_SCORE]
        )
        delta_score = EvalScore(
            name=DELTA_CLASSIFICATION_ACCURACY_SCORE, value=output_record[DELTA_CLASSIFICATION_ACCURACY_SCORE]
        )
        return [original_score, delta_score]

    def evaluate(
        self,
        model: ModelRunner,
        dataset_config: Optional[Union[DataConfig, List[DataConfig]]] = None,
        prompt_template: Optional[str] = None,
        num_records: int = 100,
        save: bool = False,
        save_strategy: Optional[SaveStrategy] = None,
    ) -> List[EvalOutput]:
        """Compute classification accuracy semantic robustness metrics on one or more datasets.

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
        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        eval_outputs: List[EvalOutput] = []

        for dataset_config in dataset_configs:
            dataset_prompt_template = (
                get_default_prompt_template(dataset_config.dataset_name) if not prompt_template else prompt_template
            )
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [DatasetColumns.TARGET_OUTPUT.value.name, DatasetColumns.MODEL_INPUT.value.name])

            valid_labels = (
                self.valid_labels
                if self.valid_labels
                else dataset.unique(column=DatasetColumns.TARGET_OUTPUT.value.name)
            )
            row_count = dataset.count()
            if len(valid_labels) / (row_count + 1) < UNIQUENESS_FACTOR:  # pragma: no cover
                logger.warning(
                    f"The number of classes: {len(valid_labels)} in the dataset is too large "
                    f"for the number of rows in the dataset: {row_count}",
                )

            eval_output = evaluate_dataset(
                dataset=dataset,
                pipeline=self._build_pipeline(model, dataset_prompt_template, valid_labels),
                dataset_name=dataset_config.dataset_name,
                eval_name=self.eval_name,
                metric_names=[CLASSIFICATION_ACCURACY_SCORE, DELTA_CLASSIFICATION_ACCURACY_SCORE],
                eval_results_path=get_eval_results_path(),
                model=model,
                prompt_template=dataset_prompt_template,
                agg_method=MEAN,
                save=save,
                save_strategy=save_strategy if save_strategy else None,
            )
            eval_outputs.append(eval_output)

        return eval_outputs
