import itertools
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from fmeval.constants import (
    DatasetColumns,
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
from fmeval.eval_algorithms.common import evaluate_dataset
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface
from fmeval.eval_algorithms.save_strategy import SaveStrategy
from fmeval.eval_algorithms.semantic_robustness_utils import (
    SemanticRobustnessConfig,
    get_perturbation_transform,
    get_model_outputs_from_perturbed_inputs,
)
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModelTypes, BertscoreHelperModel
from fmeval.transforms.common import GeneratePrompt, GetModelOutputs
from fmeval.eval_algorithms.util import (
    validate_dataset,
    verify_model_determinism,
    get_dataset_configs,
)
from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.constants import BERTSCORE_DEFAULT_MODEL
from fmeval.transforms.summarization_accuracy_metrics import BertScore
from fmeval.transforms.semantic_robustness_metrics import BertScoreDissimilarity, WER
from fmeval.transforms.transform import Transform
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.transforms.util import create_output_key
from fmeval.util import create_shared_resource, require, get_eval_results_path, cleanup_shared_resource

logger = logging.getLogger(__name__)


WER_SCORE = "word_error_rate"
BERT_SCORE_DISSIMILARITY = "bertscore_dissimilarity"
BASELINE_SUFFIX = "baseline"
BASELINE_WER_SCORE = f"{WER_SCORE}_{BASELINE_SUFFIX}"
BASELINE_BERT_SCORE_DISSIMILARITY = f"{BERT_SCORE_DISSIMILARITY}_{BASELINE_SUFFIX}"


@dataclass(frozen=True)
class GeneralSemanticRobustnessConfig(SemanticRobustnessConfig):
    """Configures the general semantic robustness evaluation algorithm.

    :param num_baseline_samples: Only used for non-deterministic models. Number of times we generate
        the model output with the same input to compute the "baseline" change in model output. We
        compute differences between all pairs of outputs, i.e. between comb(num_baseline_samples, 2) pairs.

    :param model_type_for_bertscore: Model type to use for BERT score.
    """

    num_baseline_samples: int = 4
    model_type_for_bertscore: str = BERTSCORE_DEFAULT_MODEL

    def __post_init__(self):
        super().__post_init__()
        require(
            BertscoreHelperModelTypes.model_is_allowed(self.model_type_for_bertscore),
            f"Invalid model_type_for_bertscore: {self.model_type_for_bertscore} requested in "
            f"GeneralSemanticRobustnessConfig, please choose from acceptable values: {BertscoreHelperModelTypes.model_list()}.",
        )
        require(
            self.num_baseline_samples >= 2,
            f"Invalid num_baseline_samples: {self.num_baseline_samples} in GeneralSemanticRobustnessConfig. "
            f"The value should be at least 2.",
        )


class GeneralSemanticRobustness(EvalAlgorithmInterface):
    """Semantic Robustness evaluation algorithm for general task LLMs.

    This evaluation measures how much the model output changes as a result of semantic preserving
    perturbations. Given the input, e.g., "A quick brown fox jumps over the lazy dog", the
    evaluation creates a perturbation that preserves the semantic meaning of the input e.g.,
    whitespace perturbation that changes the input text to "A q uick bro wn fox ju mps overthe lazy
    dog". The evaluation then measures how much the model output changes when prompted with the
    original vs. perturbed input.

    The output difference is measured using two metrics: the [Word Error Rate](https://huggingface.co/spaces/evaluate-metric/wer)
    and the BERTScore Dissimilarity, which is
    1 - [BERTScore](https://huggingface.co/spaces/evaluate-metric/bertscore), between the original
    and the perturbed outputs. Word Error Rate measures syntactic differences, that is, changes in
    the words, whereas BERTScore Dissimilarity measures semantic differences. Semantic differences
    account of cases when the precise words in the output change but the meaning is the same, e.g.,
    consider the outputs "it is pouring down today" vs. "it is very rainy today".

    Note: When the model generation strategy is non-deterministic (e.g., with non-zero temperature),
    the output can change even if the input is the same. In such scenarios, reporting differences
    (using Word Error Rate or BERTScore Dissimilarity) between the model output on the original input
    and perturbed inputs might show artificially low robustness since the model output changes even
    without a change in the input. So this evaluation normalizes the robustness score to account for
    the baseline non-determinism. Specifically, if d is a score (Word Error Rate or BERTScore
    Dissimilarity), then the evaluation reports max(0, d - d_base) where d_base measures the
    differences between the model output on the same input.
    """

    eval_name = EvalAlgorithm.GENERAL_SEMANTIC_ROBUSTNESS.value

    def __init__(
        self,
        eval_algorithm_config: GeneralSemanticRobustnessConfig = GeneralSemanticRobustnessConfig(),
    ):
        """GeneralSemanticRobustness initializer.

        :param eval_algorithm_config: General semantic robustness evaluation algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.num_perturbations = eval_algorithm_config.num_perturbations
        self.num_baseline_samples = eval_algorithm_config.num_baseline_samples
        self.perturbation_transform = get_perturbation_transform(eval_algorithm_config)
        self.bertscore_model = BertscoreHelperModel(eval_algorithm_config.model_type_for_bertscore)

    def _build_pipeline(
        self,
        model: ModelRunner,
        prompt_template: str,
        is_deterministic: bool,
    ) -> TransformPipeline:
        """Build the TransformPipeline to be used by `evaluate` and `evaluate_sample`.

        While other evaluation algorithms (e.g. Summarization Accuracy) can configure
        their TransformPipeline at algorithm initialization, because the General
        Semantic Robustness algorithm's evaluation logic depends on the ModelRunner
        and prompt template that are evaluation-specific (i.e. these parameters aren't
        configured at the algorithm level), the pipeline used by the GSR algorithm is built
        when `evaluate` or `evaluate_sample` is called.

        :param model: The ModelRunner representing the model under evaluation.
        :param prompt_template: A template that is used to construct the prompt fed to the model.
        :param is_deterministic: Whether `model` produces deterministic results.
            In `evaluate_sample`, this is computed by invoking the model with the
            same input twice, and checking if the model output is the same.
            In `evaluate`, similar logic is used, but instead of using just a single input,
            multiple inputs from the dataset are used.
        :returns: A TransformPipeline that can be used by either `evaluate_sample` or `evaluate`.
        """
        (
            get_perturbed_inputs,
            gen_perturbed_prompts,
            get_perturbed_responses,
        ) = get_model_outputs_from_perturbed_inputs(
            self.perturbation_transform,
            prompt_template,
            model,
        )

        original_model_output_key = DatasetColumns.MODEL_OUTPUT.value.name
        # Compute BERTScores with target_output = the original model output
        # and model_output = the output from invoking the model with the perturbed prompt.
        get_bert_scores = BertScore(
            target_output_keys=[original_model_output_key for _ in range(self.num_perturbations)],
            model_output_keys=get_perturbed_responses.output_keys,
            output_keys=[create_output_key(BertScore.__name__, i) for i in range(self.num_perturbations)],
            allow_duplicate_input_keys=True,
            bertscore_model=self.bertscore_model,
        )

        compute_bertscore_dissimilarity = BertScoreDissimilarity(
            bert_score_keys=get_bert_scores.output_keys,
            output_key=BERT_SCORE_DISSIMILARITY,
        )

        compute_wer_metric = WER(
            prediction_keys=get_perturbed_responses.output_keys,
            reference_keys=[original_model_output_key for _ in range(self.num_perturbations)],
            output_key=WER_SCORE,
        )

        transforms = [
            get_perturbed_inputs,
            gen_perturbed_prompts,
            get_perturbed_responses,
            get_bert_scores,
            compute_bertscore_dissimilarity,
            compute_wer_metric,
        ]

        pipeline = TransformPipeline(transforms)

        # If the model is not deterministic, we execute additional steps
        # to compute baseline scores for both BERTScore and WER.
        if not is_deterministic:
            # Invoke the model with the original (i.e. unperturbed) prompt
            # self.num_baseline_samples - 1 times.
            baseline_response_keys = [
                create_output_key(GeneratePrompt.__name__, BASELINE_SUFFIX, i)
                for i in range(self.num_baseline_samples - 1)
            ]
            get_baseline_outputs = GetModelOutputs(
                input_to_output_keys={DatasetColumns.PROMPT.value.name: baseline_response_keys},
                model_runner=model,
            )

            # Get every possible pair of model outputs.
            # The first output in the pair is treated as the target output
            # and the second output is treated as the model output
            # when computing the BERTScore.
            baseline_keys = baseline_response_keys + [DatasetColumns.MODEL_OUTPUT.value.name]
            all_pairs = itertools.combinations(baseline_keys, 2)
            first_output_keys, second_output_keys = zip(*all_pairs)

            # Compute baseline BERTScores and then compute BERTScore Dissimilarity using these BERTScores.
            get_baseline_bert_scores = BertScore(
                target_output_keys=list(first_output_keys),
                model_output_keys=list(second_output_keys),
                output_keys=[
                    create_output_key(BertScore.__name__, BASELINE_SUFFIX, i) for i in range(len(first_output_keys))
                ],
                allow_duplicate_input_keys=True,
                bertscore_model=self.bertscore_model,
            )
            compute_baseline_bertscore_dissimilarity = BertScoreDissimilarity(
                bert_score_keys=get_baseline_bert_scores.output_keys,
                output_key=BASELINE_BERT_SCORE_DISSIMILARITY,
            )

            # Compute WER metric using the baseline model outputs.
            compute_baseline_wer_metric = WER(
                prediction_keys=list(first_output_keys),
                reference_keys=list(second_output_keys),
                output_key=BASELINE_WER_SCORE,
            )
            # Update BERTScore Dissimilarity and WER metrics
            # given the new baseline scores that have been computed.
            update_scores = UpdateRobustnessScores()

            # Extend the pipeline with these additional steps.
            additional_steps = TransformPipeline(
                [
                    get_baseline_outputs,
                    get_baseline_bert_scores,
                    compute_baseline_bertscore_dissimilarity,
                    compute_baseline_wer_metric,
                    update_scores,
                ]
            )
            pipeline = TransformPipeline([pipeline, additional_steps])

        return pipeline

    def evaluate_sample(
        self,
        model_input: str,
        model: ModelRunner,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> List[EvalScore]:  # type: ignore[override]
        """Compute general semantic robustness metrics for a single sample.

        :param model_input: Text input for model.
        :param model: An instance of ModelRunner representing the model under evaluation.
        :param prompt_template: A template that is used in conjunction with `model_input`
            to construct the prompt that is fed to the model.
        :returns: A list of EvalScore objects, one for each of the robustness metrics.
        """
        # Determine whether model produces deterministic outputs, as this affects
        # what steps will be included in the TransformPipeline.
        prompt_composer = PromptComposer(prompt_template)
        prompt = prompt_composer.compose(model_input)
        model_output = model.predict(prompt)[0]
        is_deterministic = model_output == model.predict(prompt)[0]

        sample = {
            DatasetColumns.MODEL_INPUT.value.name: model_input,
            DatasetColumns.PROMPT.value.name: prompt,
            DatasetColumns.MODEL_OUTPUT.value.name: model_output,
        }
        pipeline = self._build_pipeline(model, prompt_template, is_deterministic=is_deterministic)
        output_record = pipeline.execute_record(sample)

        bert_score_dissimilarity_value = output_record[BERT_SCORE_DISSIMILARITY]
        wer_value = output_record[WER_SCORE]
        return [
            EvalScore(name=BERT_SCORE_DISSIMILARITY, value=bert_score_dissimilarity_value),
            EvalScore(name=WER_SCORE, value=wer_value),
        ]

    def evaluate(
        self,
        model: ModelRunner,
        dataset_config: Optional[Union[DataConfig, List[DataConfig]]] = None,
        prompt_template: Optional[str] = None,
        num_records: int = 100,
        save: bool = False,
        save_strategy: Optional[SaveStrategy] = None,
    ) -> List[EvalOutput]:
        """Compute general semantic robustness metrics on one or more datasets.

        :param model: An instance of ModelRunner representing the model under evaluation.
            This is a required argument, as even if the dataset contains model outputs,
            semantic robustness algorithms rely on invoking a model on perturbed inputs
            to see how the model outputs from the perturbed inputs differ from the original
            model outputs.
        :param dataset_config: Configures a single dataset or list of datasets used for the
            evaluation. If not provided, this method will run evaluations using all of its
            supported built-in datasets.
        :param prompt_template: A template used to generate prompts that are fed to the model.
            If not provided, defaults will be used.
        :param num_records: The number of records to be sampled randomly from the input dataset
            used to perform the evaluation.
        :param save: If set to true, prompt responses and scores will be saved to a file.
        :param save_strategy: Specifies the strategy to use the save the localized outputs of the evaluations. If not
            specified, it will save it to the path that can be configured by the EVAL_RESULTS_PATH environment variable.
            If that environment variable is also not configured, it will be saved to the default path `/tmp/eval_results/`.

        :return: A list of EvalOutput objects.
        """
        # Create a shared resource to be used during the evaluation.
        bertscore_shared_resource = create_shared_resource(self.bertscore_model)
        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [DatasetColumns.MODEL_INPUT.value.name])
            dataset_prompt_template = (
                get_default_prompt_template(dataset_config.dataset_name) if not prompt_template else prompt_template
            )
            is_deterministic = verify_model_determinism(model, dataset, dataset_prompt_template)
            eval_output = evaluate_dataset(
                dataset=dataset,
                pipeline=self._build_pipeline(model, dataset_prompt_template, is_deterministic=is_deterministic),
                dataset_name=dataset_config.dataset_name,
                eval_name=self.eval_name,
                metric_names=[BERT_SCORE_DISSIMILARITY, WER_SCORE],
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


class UpdateRobustnessScores(Transform):
    """Used by General Semantic Robustness when the model under evaluation is not deterministic.

    See the class documentation for GeneralSemanticRobustness for details on how baseline scores
    are computed and used. This transform simply updates the data corresponding to the
    WER_SCORE and BERT_SCORE_DISSIMILARITY keys after baseline scores have been computed.
    """

    def __init__(self):
        super().__init__()
        self.register_input_output_keys(
            input_keys=[WER_SCORE, BERT_SCORE_DISSIMILARITY, BASELINE_WER_SCORE, BASELINE_BERT_SCORE_DISSIMILARITY],
            output_keys=[WER_SCORE, BERT_SCORE_DISSIMILARITY],
        )

    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Update the values corresponding to the keys WER_SCORE and BERT_SCORE_DISSIMILARITY.

        This method does not add new keys, but rather mutates the data corresponding to existing
        keys (WER_SCORE and BERT_SCORE_DISSIMILARITY) in the input record.

        :param record: The input record.
        :returns: The input record with updated WER_SCORE and BERT_SCORE_DISSIMILARITY values.
        """
        bert_score_dissimilarity_value = record[BERT_SCORE_DISSIMILARITY]
        wer_value = record[WER_SCORE]
        baseline_bert_score_dissimilarity_value = record[BASELINE_BERT_SCORE_DISSIMILARITY]
        baseline_wer_value = record[BASELINE_WER_SCORE]

        record[BERT_SCORE_DISSIMILARITY] = max(
            0, bert_score_dissimilarity_value - baseline_bert_score_dissimilarity_value
        )
        record[WER_SCORE] = max(0, wer_value - baseline_wer_value)
        return record
