import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable

from fmeval.constants import (
    DatasetColumns,
    MEAN,
)
from fmeval.data_loaders.util import get_dataset
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.common import evaluate_dataset
from fmeval.eval_algorithms.save_strategy import SaveStrategy
from fmeval.eval_algorithms.util import get_dataset_configs, normalize_text_quac_protocol
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

FACTUAL_KNOWLEDGE = EvalAlgorithm.FACTUAL_KNOWLEDGE.value
FACTUAL_KNOWLEDGE_QUASI_EXACT = "factual_knowledge_quasi_exact"
SCORE_NAMES = [FACTUAL_KNOWLEDGE, FACTUAL_KNOWLEDGE_QUASI_EXACT]

logger = logging.getLogger(__name__)


def _exact_inclusion_score(model_output: str, target_output: str) -> float:
    """
    Given the model output and the target output, _exact_inclusion_score checks whether the target output
    is contained in the model output after converting both strings to lowercase. If so, the function returns
    1.0. Otherwise, returns 0.

    :param model_output: The output of a model that we want to evaluate.
    :param target_output: The reference or the "ground truth" output.
    :returns: The exact_inclusion score.
    """
    model_output_lower_case = model_output.lower()
    return float(target_output.lower() in model_output_lower_case)


def _quasi_exact_inclusion_score(model_output: str, target_output: str) -> float:
    """
    Inspired by HELM: https://github.com/stanford-crfm/helm/blob/62f817eb695a31e8389e3f7be30609d3f0871837/src/helm/benchmark/metrics/basic_metrics.py#L144
    Computes if the target_output is contained in the model_output after normalizing both strings. If so, the
    function returns 1.0. Otherwise, returns 0.

    Normalization: Given a text, normalize it using the SQUAD/QUAC protocol (remove punctuations, excess spaces,
    and articles) and return the lowercased tokens.
    SQUAD (https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/) and
    QuAC benchmarks (https://s3.amazonaws.com/my89public/quac/scorer.py) use this protocol to normalize text before
    evaluating it. Can learn more at fmeval/src/fmeval/eval_algorithms/util.py

    :param model_output: The output of a model that we want to evaluate.
    :param target_output: The reference or the "ground truth" output.
    :returns: The quasi_exact_inclusion score (1 if the target_output is contained in model_output
    after normalization, else 0).
    """
    return float(
        normalize_text_quac_protocol(target_output.strip()) in normalize_text_quac_protocol(model_output.strip())
    )


FACTUAL_KNOWLEDGE_SCORES_TO_FUNCS: Dict[str, Callable[..., float]] = {
    FACTUAL_KNOWLEDGE: _exact_inclusion_score,
    FACTUAL_KNOWLEDGE_QUASI_EXACT: _quasi_exact_inclusion_score,
}


class FactualKnowledgeScores(Transform):
    """This transform augments its input record with the computed factual knowledge scores.

    See the docstring for `FactualKnowledge` for more details regarding the score itself.
    """

    def __init__(
        self,
        target_output_key: str = DatasetColumns.TARGET_OUTPUT.value.name,
        model_output_key: str = DatasetColumns.MODEL_OUTPUT.value.name,
        output_keys: List[str] = SCORE_NAMES,
        target_output_delimiter: Optional[str] = "<OR>",
        logical_operator: str = "OR",
    ):
        """FactualKnowledgeScores initializer.

        :param target_output_key: The record key corresponding to the target output.
        :param model_output_key: The record key corresponding to the model output.
        :param output_key: The key corresponding to the factual knowledge score that
            will be added to the input record.
        :param target_output_delimiter: This delimiter is used to combine all possible target outputs into
            a single string. For example, if valid answers are ["UK", "England"] and the delimiter is "<OR>",
            then the target output text will be "UK<OR>England". This can be useful to account for multiple
            valid target outputs or to ensure that multiple target outputs are contained in the model output
            (which can be configured using the logical_operator).
        :param logical_operator: The logical operator can be set to "OR" (default) or "AND". When the logical operator
            is "OR" (the default behavior), at least one of the possible target outputs (separated by the
            target_output_delimiter) must be contained in the model output for the answer to be correct. When the logical
            operator is "AND", ALL possible target outputs (separated by the target_output_delimiter) must be contained in
            the model output in order for the answer to be correct.
        """
        super().__init__(target_output_key, model_output_key, output_keys, target_output_delimiter, logical_operator)
        self.register_input_output_keys(
            input_keys=[target_output_key, model_output_key],
            output_keys=output_keys,
        )
        self.target_output_key = target_output_key
        self.model_output_key = model_output_key
        self.output_keys = output_keys
        self.target_output_delimiter = target_output_delimiter
        self.logical_operator = logical_operator

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with the computed factual knowledge scores.

        :param record: The input record.
        :returns: The input record, with the factual knowledge scores added in.
        """
        target_output = record[self.target_output_key]
        model_output = record[self.model_output_key]
        for output_key, score_name in zip(self.output_keys, SCORE_NAMES):
            record[output_key] = self._get_score(
                target_output=target_output,
                model_output=model_output,
                score_fn=FACTUAL_KNOWLEDGE_SCORES_TO_FUNCS[score_name],
            )
        return record

    def _get_score(
        self,
        target_output: str,
        model_output: str,
        score_fn: Callable[..., float],
        **fn_kwargs,
    ) -> float:
        """Compute a factual knowledge score for a target output and model output pair based
        on the score function.

        :param target_output: Target output.
        :param model_output: Model output.
        :param score_fn: One of the functions in FACTUAL_KNOWLEDGE_SCORES_TO_FUNCS.
        :returns: A computed factual knowledge score (0 or 1). See the docstring for
        `FactualKnowledge` for more details on what these numerical values represent.
        """
        possible_targets = target_output.split(self.target_output_delimiter)
        if self.logical_operator == "OR":
            return max([score_fn(model_output, target, **fn_kwargs) for target in possible_targets])
        else:  # self.logical_operator is "AND"
            # checks that every target is in model_output, otherwise returns 0.0
            return min([score_fn(model_output, target, **fn_kwargs) for target in possible_targets])


@dataclass(frozen=True)
class FactualKnowledgeConfig(EvalAlgorithmConfig):
    """Configures the factual knowledge evaluation algorithm.

    :param target_output_delimiter: This delimiter is used to combine all possible target outputs into
        a single string. For example, if valid answers are ["UK", "England"] and the delimiter is "<OR>",
        then the target output text will be "UK<OR>England". This can be useful to account for multiple
        valid target outputs or to ensure that multiple target outputs are contained in the model output
        (which can be configured using the logical_operator).
    :param logical_operator: The logical operator can be set to "OR" (default) or "AND". When the logical operator
        is "OR" (the default behavior), at least one of the possible target outputs (separated by the
        target_output_delimiter) must be contained in the model output for the answer to be correct. When the logical
        operator is "AND", ALL possible target outputs (separated by the target_output_delimiter) must be contained in
        the model output in order for the answer to be correct.
    """

    target_output_delimiter: Optional[str] = "<OR>"
    logical_operator: str = "OR"

    def __post_init__(self):
        if self.target_output_delimiter == "":
            raise EvalAlgorithmClientError(
                "Empty target_output_delimiter is provided. Please either provide a non-empty string, "
                "or set it to None."
            )
        if self.logical_operator not in ["OR", "AND"]:
            raise EvalAlgorithmClientError(
                'Invalid logical_operator is provided. The only valid inputs are strings "OR" and "AND".'
            )
        if self.target_output_delimiter in ["<OR>", "<AND>"] and self.target_output_delimiter != "<{}>".format(
            self.logical_operator
        ):
            logger.warning(
                f"The target_output_delimiter `{self.target_output_delimiter}` and logical_operator"
                f" `{self.logical_operator}` are not consistent."
            )


class FactualKnowledge(EvalAlgorithmInterface):
    """
    This evaluation measures the ability of language models to reproduce facts about the real world and was proposed
    by [Petroni et al.](https://arxiv.org/pdf/1909.01066.pdf). The evaluation queries the model with prompts like
    'Berlin is the capital of' and 'Tata Motors is a subsidiary of' and compares the model generation with one or more
    target answers. The prompts are divided into different knowledge categories like capitals, subsidiaries, etc.

    This evaluation outputs two binary metrics.
    The first is the "exact_inclusion" score: the metric value is 1 if the lower-cased expected answer is
    contained anywhere within the lower-cased model response. For instance, consider the prompt
    'Berlin is the capital of' with the expected answer 'Germany'.
    If the model generation is 'Germany, and is also its most populous city', then the metric evaluates to 1.

    The second metric is the "quasi_exact_inclusion" score: the metric value is 1 if the target output is contained
    in the model output after both strings are normalized.
    Inspired by HELM: https://github.com/stanford-crfm/helm/blob/62f817eb695a31e8389e3f7be30609d3f0871837/src/helm/benchmark/metrics/basic_metrics.py#L144

    If there is more than one correct target answer, the `logical_operator` can be set to "OR" (default) and
    answers are seperated by the `target_output_delimiter`, both of which are configured inside the
    `FactualKnowledgeConfig`. The `target_output_delimiter` defaults to `<OR>`, i.e, the target answer in this
    example could be Germany<OR>Berlin (since Berlin is its own federal state).

    If there are multiple correct target answers that must be included in the model output,
    the `logical_operator` can be set to "AND". For example, consider the prompt 'What are the three primary colors?'.
    The target answer would be Red<AND>Yellow<AND>Blue" (note that the target_output_delimiter could be anything,
    but it is "<AND>" here for the sake of consistency with the logical_operator value).Red, yellow, and blue must
    all be contained in the model generation for the answer to be correct under this configuration.
    """

    eval_name = EvalAlgorithm.FACTUAL_KNOWLEDGE.value

    def __init__(self, eval_algorithm_config: FactualKnowledgeConfig = FactualKnowledgeConfig()):
        """FactualKnowledge initializer.

        :param eval_algorithm_config: Factual knowledge evaluation algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.pipeline = TransformPipeline(
            [
                FactualKnowledgeScores(
                    target_output_delimiter=eval_algorithm_config.target_output_delimiter,
                    logical_operator=eval_algorithm_config.logical_operator,
                )
            ]
        )

    def evaluate_sample(self, target_output: str, model_output: str) -> List[EvalScore]:  # type: ignore[override]
        """Computes the factual knowledge metrics for a single sample.

        :param target_output: The expected responses from the model.
        :param model_output: The output of the model being evaluated.
        :return: A list of EvalScore objects, one for each of the Factual Knowledge metrics
        ("exact_inclusion" and "quasi_exact_inclusion").
        """
        sample = {
            DatasetColumns.TARGET_OUTPUT.value.name: target_output,
            DatasetColumns.MODEL_OUTPUT.value.name: model_output,
        }
        result = self.pipeline.execute_record(sample)
        return [EvalScore(name=score_name, value=result[score_name]) for score_name in SCORE_NAMES]

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[Union[DataConfig, List[DataConfig]]] = None,
        prompt_template: Optional[str] = None,
        num_records: int = 300,
        save: bool = False,
        save_strategy: Optional[SaveStrategy] = None,
    ) -> List[EvalOutput]:
        """Compute the factual knowledge scores on one or more datasets.

        :param model: An instance of ModelRunner representing the model under evaluation.
            If this argument is None, the `dataset_config` argument must not be None,
            and must correspond to a dataset that already contains a column with model outputs.
        :param dataset_config: Configures a single dataset or list of datasets used for the
            evaluation. If not provided, this method will run evaluations using all of its
            supported built-in datasets.
        :param prompt_template: A template used to generate prompts that are fed to the model.
            If not provided, defaults will be used. If provided, `model` must not be None.
        :param num_records: The number of records to be sampled randomly from the input dataset(s)
            used to perform the evaluation(s). Note that the default value is 300, rather than
            100, as it is for the rest of the built-in algorithms. This is because there
            are 15 categories for factual knowledge, and if only 100 samples are used, there
            will be categories with very few samples.
        :param save: If set to true, prompt responses and scores will be saved to a file.
        :param save_strategy: Specifies the strategy to use the save the localized outputs of the evaluations. If not
            specified, it will save it to the path that can be configured by the EVAL_RESULTS_PATH environment variable.
            If that environment variable is also not configured, it will be saved to the default path `/tmp/eval_results/`.

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
                metric_names=SCORE_NAMES,
                eval_results_path=get_eval_results_path(),
                model=model,
                prompt_template=prompt_template,
                agg_method=MEAN,
                save=save,
                save_strategy=save_strategy,
            )
            eval_outputs.append(eval_output)
        return eval_outputs
