from typing import Optional, List, Dict, Any
import logging
from dataclasses import dataclass

from fmeval.constants import (
    DatasetColumns,
    MEAN,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.data_loaders.util import get_dataset
from fmeval.eval_algorithms import EvalAlgorithm, EvalScore, EvalOutput
from fmeval.eval_algorithms.common import evaluate_dataset
from fmeval.eval_algorithms.eval_algorithm import (
    EvalAlgorithmInterface,
    EvalAlgorithmConfig,
)
from fmeval.eval_algorithms.save_strategy import SaveStrategy
from fmeval.eval_algorithms.util import get_dataset_configs, validate_dataset
from fmeval.model_runners.bedrock_model_runner import BedrockModelRunner
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.transform import Transform
from fmeval.transforms.util import validate_call
from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.util import get_eval_results_path

logger = logging.getLogger(__name__)

CONTEXT_PRECISION_SCORE = "context_precision_score"

DEFAULT_CONTEXT_PRECISION_PROMPT_TEMPLATE = (
    "Given a question, answer, and context, verify if the context was useful in "
    "arriving at the given answer. Give verdict as 1 if useful and 0 if not. "
    "question: $model_input, answer: $target_output, context: $retrieved_context."
    "The verdict should only contain an integer, either 1 or 0, do not give an explanation."
)

BINARY_SCORE_VALUES = ["0", "1"]

# TODO: Pending judge model selection
def get_default_judge_model() -> BedrockModelRunner:  # pragma: no cover
    return BedrockModelRunner(
        model_id="anthropic.claude-v2",
        output="completion",
        content_template='{"prompt": $prompt, "max_tokens_to_sample": 10000, "temperature": 0.1}',
    )


@dataclass(frozen=True)
class ContextQualityConfig(EvalAlgorithmConfig):
    """Configures the Context Quality evaluation algorithm.

    :param target_output_delimiter: There can be multiple target outputs for a given input.
        This delimiter is used to combine all retrieved contexts into a single string.
    :param retrieved_context_delimiter: There can be multiple retrieved contexts for a given input.
        This delimiter is used to combine all retrieved contexts into a single string.
    """

    target_output_delimiter: Optional[str] = "<OR>"
    retrieved_context_delimiter: Optional[str] = "<AND>"


class ContextQuality(EvalAlgorithmInterface):
    eval_name = EvalAlgorithm.CONTEXT_QUALITY.value

    def __init__(self, eval_algorithm_config: ContextQualityConfig = ContextQualityConfig()):
        """ContextQuality initializer

        :param eval_algorithm_config: Context Quality evaluation algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.eval_algorithm_config = eval_algorithm_config

    def evaluate(
        self,
        judge_model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        num_records: int = 100,
        save: bool = False,
        save_strategy: Optional[SaveStrategy] = None,
        context_precision_prompt_template: str = DEFAULT_CONTEXT_PRECISION_PROMPT_TEMPLATE,
    ) -> List[EvalOutput]:
        """Compute context quality metrics on one or more datasets.

        :param judge_model: An instance of ModelRunner representing the judge model to be used.
            If this argument is None, default judge model will be provided.
        :param dataset_config: Configures the single dataset used for evaluation.
            If not provided, evaluations will be run on all of this algorithm's built-in datasets.
        :param num_records: The number of records to be sampled randomly from the input dataset(s)
            used to perform the evaluation(s).
        :param save: If set to true, prompt responses and scores will be saved to a file.
        :param save_strategy: Specifies the strategy to use the save the localized outputs of the evaluations. If not
            specified, it will save it to the path that can be configured by the EVAL_RESULTS_PATH environment variable.
            If that environment variable is also not configured, it will be saved to the default path `/tmp/eval_results/`.
        :param context_precision_prompt_template: The string prompt template for context precision prompts.
        :return: A list of EvalOutput objects.
        """
        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        if not judge_model:  # pragma: no cover
            judge_model = get_default_judge_model()
        pipeline = ContextQuality._build_pipeline(
            judge_model=judge_model,
            context_quality_config=self.eval_algorithm_config,
            context_precision_prompt_template=context_precision_prompt_template,
        )

        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(
                dataset,
                [
                    DatasetColumns.MODEL_INPUT.value.name,
                    DatasetColumns.TARGET_OUTPUT.value.name,
                    DatasetColumns.RETRIEVED_CONTEXT.value.name,
                ],
            )
            eval_output = evaluate_dataset(
                dataset=dataset,
                pipeline=pipeline,
                dataset_name=dataset_config.dataset_name,
                eval_name=self.eval_name,
                metric_names=[CONTEXT_PRECISION_SCORE],
                eval_results_path=get_eval_results_path(),
                agg_method=MEAN,
                save=save,
                save_strategy=save_strategy,
                validate_columns=False,
            )
            eval_outputs.append(eval_output)
        return eval_outputs

    def evaluate_sample(
        self,
        model_input: str,
        target_output: str,
        retrieved_context: str,
        judge_model: Optional[ModelRunner] = None,
        context_precision_prompt_template: str = DEFAULT_CONTEXT_PRECISION_PROMPT_TEMPLATE,
    ) -> List[EvalScore]:
        """
        :param model_input: The input that is composed with a prompt template and passed to the model.
        :param target_output: The referenced "ground truth" answer.
        :param retrieved_context: The context retrieved from the RAG system.
        :param judge_model: An instance of ModelRunner representing the judge model to be used.
            If this argument is None, default judge model will be provided.
        :param context_precision_prompt_template: The string prompt template for context precision prompts.
        :return: A list of EvalScores corresponding to the context quality metrics.
        """
        if not judge_model:  # pragma: no cover
            judge_model = get_default_judge_model()
        sample = {
            DatasetColumns.MODEL_INPUT.value.name: model_input,
            DatasetColumns.TARGET_OUTPUT.value.name: target_output,
            DatasetColumns.RETRIEVED_CONTEXT.value.name: retrieved_context,
        }
        pipeline = ContextQuality._build_pipeline(
            judge_model=judge_model,
            context_quality_config=self.eval_algorithm_config,
            context_precision_prompt_template=context_precision_prompt_template,
        )
        result = pipeline.execute_record(sample)
        context_precision_score = EvalScore(name=CONTEXT_PRECISION_SCORE, value=result[CONTEXT_PRECISION_SCORE])
        return [context_precision_score]

    @staticmethod
    def _build_pipeline(
        judge_model: ModelRunner, context_quality_config: ContextQualityConfig, context_precision_prompt_template: str
    ) -> TransformPipeline:
        """Builds a transform pipeline to compute context quality scores.

        :param judge_model: An instance of ModelRunner representing the judge model to be used.
        :param context_quality_config: An instance of ContextQualityConfig.
        :param context_precision_prompt_template: The string prompt template for context precision prompts.
        :return: A TransformPipeline containing Transforms to compute context quality scores.
        """
        precision_score = ContextPrecisionScore(
            judge_model=judge_model,
            context_precision_prompt_template=context_precision_prompt_template,
            target_output_delimiter=context_quality_config.target_output_delimiter,
            retrieved_context_delimiter=context_quality_config.retrieved_context_delimiter,
        )
        pipeline = TransformPipeline([precision_score])
        return pipeline


class ContextPrecisionScore(Transform):
    def __init__(
        self,
        judge_model: ModelRunner,
        context_precision_prompt_template: str,
        model_input_key: str = DatasetColumns.MODEL_INPUT.value.name,
        target_output_key: str = DatasetColumns.TARGET_OUTPUT.value.name,
        retrieved_context_key: str = DatasetColumns.RETRIEVED_CONTEXT.value.name,
        context_precision_score_key: str = CONTEXT_PRECISION_SCORE,
        target_output_delimiter: Optional[str] = "<OR>",
        retrieved_context_delimiter: Optional[str] = "<AND>",
    ):
        """Context Precision score initializer.

        :param judge_model: A ModelRunner instance for the context judge model.
        :param context_precision_prompt_template: The prompt template for context precision.
        :param model_input_key: The key corresponding to the model input.
        :param target_output_key: The key corresponding to the target output.
        :param retrieved_context_key: The key corresponding to the retrieved context.
        :param context_precision_score_key: The key corresponding to the context precision score.
        :param target_output_delimiter: The delimiter used to concatenate multiple target outputs.
        :param retrieved_context_delimiter: The delimiter used to concatenate multiple retrieved contexts.
        """
        super().__init__(
            judge_model,
            context_precision_prompt_template,
            model_input_key,
            target_output_key,
            retrieved_context_key,
            context_precision_score_key,
            target_output_delimiter,
            retrieved_context_delimiter,
        )
        self.register_input_output_keys(
            input_keys=[model_input_key, target_output_key, retrieved_context_key],
            output_keys=[context_precision_score_key],
        )
        self.model_input_key = model_input_key
        self.target_output_key = target_output_key
        self.retrieved_context_key = retrieved_context_key
        self.judge_model = judge_model
        self.context_precision_prompt_template = context_precision_prompt_template
        self.context_precision_score_key = context_precision_score_key
        self.target_output_delimiter = target_output_delimiter
        self.retrieved_context_delimiter = retrieved_context_delimiter
        self.context_precision_prompt_composer = PromptComposer(self.context_precision_prompt_template)

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with the computed context quality scores.

        :param record: the input record.
        :return: The input record with the context quality scores added.
        """
        model_input = record[self.model_input_key]
        target_output = record[self.target_output_key]
        retrieved_context = record[self.retrieved_context_key]
        record[self.context_precision_score_key] = self._get_score(model_input, target_output, retrieved_context)
        return record

    def _get_score(self, model_input: str, target_output: str, retrieved_context: str) -> float:
        """Computes the context precision score by composing a prompt on each (context chunk, model input,
        target output) triple and doing inference on the judge model to get a verdict of 0 or 1, then
        calculating the score using the judge model verdicts.

        For multiple target contexts, we do a "logical or" operation on the model verdicts, so if the
        model verdict for a given context chunk is 1 for any target output, the model verdict for that
        context chunk will be 1.

        :param model_input: The input that is composed with a prompt template and passed to the model.
        :param target_output: The referenced "ground truth" answer.
        :param retrieved_context: The context retrieved from the RAG system.
        :return: the context precision score.
        """
        target_outputs = target_output.split(self.target_output_delimiter)
        context_chunks = retrieved_context.split(self.retrieved_context_delimiter)
        results = []
        for context_chunk in context_chunks:
            model_verdict = 0
            for target_output in target_outputs:
                prompt = self.context_precision_prompt_composer.compose(
                    model_input, {"target_output": target_output, "retrieved_context": context_chunk}
                )
                model_output = self.judge_model.predict(prompt)[0]
                model_verdict = ContextPrecisionScore._parse_model_output(model_output)
                if model_verdict:
                    break
            results.append(model_verdict)
        score = ContextPrecisionScore._compute_metric(results)
        return score

    @staticmethod
    def _parse_model_output(model_output: str) -> int:
        """Parses the model output to a binary score of 0 or 1

        :param model_output: The output of the judge model.
        :return: Score of 0 or 1.
        """
        response_words = model_output.split(" ")
        predicted_labels = [
            word.lower().strip() for word in response_words if word.lower().strip() in BINARY_SCORE_VALUES
        ]
        string_label = predicted_labels[0] if predicted_labels else "0"
        return int(string_label)

    @staticmethod
    def _compute_metric(model_verdicts: List[int]) -> float:
        """Computes the context precision score for one record given judge model verdicts

        :param model_verdicts: A list of judge model verdicts of 0 or 1.
        :return: the context precision score for the record.
        """
        denominator = sum(model_verdicts) + 1e-10
        numerator = sum(
            [(sum(model_verdicts[: i + 1]) / (i + 1)) * model_verdicts[i] for i in range(len(model_verdicts))]
        )
        score = numerator / denominator
        return score
