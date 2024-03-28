import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from fmeval.constants import (
    DatasetColumns,
    MEAN,
)
from fmeval.data_loaders.util import get_dataset
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.util import get_dataset_configs, evaluate_dataset
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface, EvalAlgorithmConfig
from fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalOutput,
    EvalScore,
)
from fmeval.eval_algorithms.util import validate_dataset
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.transform import Transform
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.transforms.util import validate_call
from fmeval.util import get_eval_results_path

logger = logging.getLogger(__name__)

FACTUAL_KNOWLEDGE = EvalAlgorithm.FACTUAL_KNOWLEDGE.value


class FactualKnowledgeScore(Transform):
    """This transform augments its input record with the computed factual knowledge score.

    See the docstring for `FactualKnowledge` for more details regarding the score itself.
    """

    def __init__(
        self,
        target_output_key: str = DatasetColumns.TARGET_OUTPUT.value.name,
        model_output_key: str = DatasetColumns.MODEL_OUTPUT.value.name,
        output_key: str = FACTUAL_KNOWLEDGE,
        target_output_delimiter: Optional[str] = "<OR>",
    ):
        """FactualKnowledgeScore initializer.

        :param target_output_key: The record key corresponding to the target output.
        :param model_output_key: The record key corresponding to the model output.
        :param output_key: The key corresponding to the factual knowledge score that
            will be added to the input record.
        :param target_output_delimiter: See the docstring in `FactualKnowledgeConfig`.
        """
        super().__init__(target_output_key, model_output_key, output_key, target_output_delimiter)
        self.register_input_output_keys(
            input_keys=[target_output_key, model_output_key],
            output_keys=[output_key],
        )
        self.target_output_key = target_output_key
        self.model_output_key = model_output_key
        self.output_key = output_key
        self.target_output_delimiter = target_output_delimiter

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with the computed factual knowledge score.

        :param record: The input record.
        :returns: The input record, with the factual knowledge score added in.
        """
        target_output = record[self.target_output_key]
        model_output = record[self.model_output_key]
        record[self.output_key] = self._get_score(target_output, model_output)
        return record

    def _get_score(self, target_output: str, model_output: str) -> int:
        """Compute the factual knowledge score for a target output and model output pair.

        :param target_output: Target output.
        :param model_output: Model output.
        :returns: Either 0 or 1. See the docstring for `FactualKnowledge` for more details
            on what these numerical values represent.
        """
        possible_targets = target_output.split(self.target_output_delimiter)
        model_output_lower_case = model_output.lower()
        return int(any([t.lower() in model_output_lower_case for t in possible_targets]))


@dataclass(frozen=True)
class FactualKnowledgeConfig(EvalAlgorithmConfig):
    """Configures the factual knowledge evaluation algorithm.

    :param target_output_delimiter: There can be multiple valid target outputs for a given question.
        This delimiter is used to combine all possible target outputs into a single string.
        For example, if valid answers are ["UK", "England"] and the delimiter is "<OR>", then the
        target output text will be "UK<OR>England".
    """

    target_output_delimiter: Optional[str] = "<OR>"

    def __post_init__(self):
        if self.target_output_delimiter == "":
            raise EvalAlgorithmClientError(
                "Empty target_output_delimiter is provided. Please either provide a non-empty string, or set it to None."
            )


class FactualKnowledge(EvalAlgorithmInterface):
    """
    This evaluation measures the ability of language models to reproduce facts about the real world and was proposed
    by [Petroni et al.](https://arxiv.org/pdf/1909.01066.pdf). The evaluation queries the model with prompts like
    'Berlin is the capital of' and 'Tata Motors is a subsidiary of' and compares the model generation with one or more
    target answers. The prompts are divided into different knowledge categories like capitals, subsidiaries, etc.

    This evaluation outputs a single binary metric. The metric value is 1 if the lower-cased expected answer is
    contained anywhere within the lower-cased model response. For instance, consider the prompt
    'Berlin is the capital of' with the expected answer 'Germany'.
    If the model generation is 'Germany, and is also its most populous city', then the metric evaluates to 1.

    If there is more than one correct target answer, answers are seperated by the `target_output_delimiter` which can be
    configured inside the `FactualKnowledgeConfig`. It defaults to `<OR>`, i.e, the target answer in this example could
    be Germany<OR>Berlin (since Berlin is its own federal state).
    """

    eval_name = EvalAlgorithm.FACTUAL_KNOWLEDGE.value

    def __init__(self, eval_algorithm_config: FactualKnowledgeConfig = FactualKnowledgeConfig()):
        """FactualKnowledge initializer.

        :param eval_algorithm_config: Factual knowledge evaluation algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.pipeline = TransformPipeline(
            [FactualKnowledgeScore(target_output_delimiter=eval_algorithm_config.target_output_delimiter)]
        )

    def evaluate_sample(self, target_output: str, model_output: str) -> List[EvalScore]:  # type: ignore[override]
        """Compute the factual knowledge score on a single sample.

        :param target_output: The expected responses from the model.
        :param model_output: The output of the model being evaluated.
        :return: A single-element list containing an EvalScore corresponding to the factual knowledge score.
        """
        sample = {
            DatasetColumns.TARGET_OUTPUT.value.name: target_output,
            DatasetColumns.MODEL_OUTPUT.value.name: model_output,
        }
        result = self.pipeline.execute_record(sample)
        return [EvalScore(name=FACTUAL_KNOWLEDGE, value=result[FACTUAL_KNOWLEDGE])]

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        num_records: int = 300,
        save: bool = False,
    ) -> List[EvalOutput]:
        """Compute the factual knowledge score on one or more datasets.

        :param model: An instance of ModelRunner representing the model under evaluation.
            If this argument is None, the `dataset_config` argument must not be None,
            and must correspond to a dataset that already contains a column with model outputs.
        :param dataset_config: Configures the single dataset used for evaluation.
            If not provided, evaluations will be run on all of this algorithm's built-in datasets.
        :param prompt_template: A template used to generate prompts that are fed to the model.
            If not provided, defaults will be used. If provided, `model` must not be None.
        :param num_records: The number of records to be sampled randomly from the input dataset(s)
            used to perform the evaluation(s). Note that the default value is 300, rather than
            100, as it is for the rest of the built-in algorithms. This is because there
            are 15 categories for factual knowledge, and if only 100 samples are used, there
            will be categories with very few samples.
        :param save: If set to true, prompt responses and scores will be saved to a file.
            The path that this file is stored at can be configured by the EVAL_RESULTS_PATH
            environment variable.

        :return: A list of EvalOutput objects.
        """
        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [DatasetColumns.TARGET_OUTPUT.value.name])
            eval_output = evaluate_dataset(
                dataset=dataset,
                pipeline=self.pipeline,
                dataset_name=dataset_config.dataset_name,
                eval_name=self.eval_name,
                metric_names=[FACTUAL_KNOWLEDGE],
                eval_results_path=get_eval_results_path(),
                model=model,
                prompt_template=prompt_template,
                agg_method=MEAN,
                save=save,
            )
            eval_outputs.append(eval_output)
        return eval_outputs
