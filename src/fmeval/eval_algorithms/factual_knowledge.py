import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


from fmeval.constants import (
    DatasetColumns,
    MEAN,
)
import fmeval.util as util
from fmeval.data_loaders.util import get_dataset
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.util import save_dataset, generate_output_dataset_path
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface, EvalAlgorithmConfig
from fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalOutput,
    EvalScore,
    EVAL_DATASETS,
    DATASET_CONFIGS,
    get_default_prompt_template,
)
from fmeval.eval_algorithms.util import (
    generate_model_predict_response_for_dataset,
    generate_prompt_column_for_dataset,
    aggregate_evaluation_scores,
    validate_dataset,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.perf_util import timed_block

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactualKnowledgeConfig(EvalAlgorithmConfig):
    """
    Configuration for the factual knowledge eval algorithm

    :param target_output_delimiter: Target Output can have multiple answers. We expect customer to combine all the
        possible answers into a single string and use the delimiter to separate them. For instance,
        if the answers are ["UK", "England"] and the delimiter="<OR>", then the target_output should be "UK<OR>England".
    """

    target_output_delimiter: Optional[str] = "<OR>"

    def __post_init__(self):
        if self.target_output_delimiter == "":
            raise EvalAlgorithmClientError(
                "Empty target_output_delimiter is provided. Please either provide a non-empty string, or set it to None."
            )


FACTUAL_KNOWLEDGE = EvalAlgorithm.FACTUAL_KNOWLEDGE.value


class FactualKnowledge(EvalAlgorithmInterface):
    """
    Factual Knowledge Eval algorithm
    """

    def __init__(self, eval_algorithm_config: FactualKnowledgeConfig = FactualKnowledgeConfig()):
        """Default constructor

        :param eval_algorithm_config: Factual knowledge eval algorithm config.
        """
        self.eval_name = FACTUAL_KNOWLEDGE
        self._eval_algorithm_config = eval_algorithm_config

    def evaluate_sample(self, target_output: str, model_output: str) -> List[EvalScore]:  # type: ignore[override]
        """
        Factual knowledge evaluate sample.

        Given an input prompt e.g., "London is the capital of" and expected answers(target_output) like
        ["United Kingdom", "England"], if the model is able to arrive at the correct completion(model_output).
        Generating any of the expected answers is considered a correct completion.
        Since models might generate long outputs, this evaluation does not look for an exact match.
        It considers the completion to be correct if the answer is contained within the model output generated.

        :param target_output: The expected responses from the model
        :param model_output: The output of a model that we want to evaluate.
        :return: list of EvalScore object
        """
        if target_output is None:
            raise EvalAlgorithmClientError(
                "Missing required input: target_output, for FactualKnowledge evaluate_sample"
            )
        if model_output is None:
            raise EvalAlgorithmClientError("Missing required input: model_output, for FactualKnowledge evaluate_sample")

        return [
            EvalScore(
                name=self.eval_name,
                value=self._get_score(target_output=target_output, model_output=model_output),
            )
        ]

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        save: bool = False,
        num_records=300,
    ) -> List[EvalOutput]:
        """
        Factual knowledge evaluate.

        Given an input prompt e.g., "London is the capital of" and expected answers(target_output) like
        ["United Kingdom", "England"], if the model is able to arrive at the correct completion(model_output).
        Generating any of the expected answers is considered a correct completion.
        Since models might generate long outputs, this evaluation does not look for an exact match.
        It considers the completion to be correct if the answer is contained within the model output generated.

        :param model: An instance of ModelRunner which is the model under evaluation
        :param dataset_config: The config to load the dataset to use for evaluation. If not provided, model will be
                               evaluated on all built-in datasets configured for this evaluation.
        :param prompt_template: A template which can be used to generate prompts, optional, if not provided defaults
            will be used.
        :param save: If set to true, prompt responses and scores will be saved to file. The output is written to
                     EvalAlgorithmInterface.EVAL_RESULTS_PATH
        :param num_records: The number of records to be sampled randomly from the input dataset to perform the
                            evaluation
        :return: List of EvalOutput objects. Current implementation returns only one score.
        """
        if dataset_config:
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[self.eval_name]]

        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [DatasetColumns.TARGET_OUTPUT.value.name, DatasetColumns.MODEL_INPUT.value.name])
            dataset_prompt_template = None
            if DatasetColumns.MODEL_OUTPUT.value.name not in dataset.columns():
                util.require(model, "No ModelRunner provided. ModelRunner is required for inference on model_inputs")
                dataset_prompt_template = (
                    get_default_prompt_template(dataset_config.dataset_name) if not prompt_template else prompt_template
                )
                dataset = generate_prompt_column_for_dataset(
                    dataset_prompt_template,
                    dataset,
                    DatasetColumns.MODEL_INPUT.value.name,
                    DatasetColumns.PROMPT.value.name,
                )
                assert model  # to satisfy mypy
                dataset = generate_model_predict_response_for_dataset(
                    model=model,
                    data=dataset,
                    model_input_column_name=DatasetColumns.PROMPT.value.name,
                    model_output_column_name=DatasetColumns.MODEL_OUTPUT.value.name,
                )
            with timed_block(f"Computing score and aggregation on dataset {dataset_config.dataset_name}", logger):

                def _generate_eval_scores(row: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
                    """
                    Map function generating the scores for every input record in input dataset
                    """
                    row[FACTUAL_KNOWLEDGE] = self._get_score(
                        row[DatasetColumns.TARGET_OUTPUT.value.name],
                        row[DatasetColumns.MODEL_OUTPUT.value.name],
                    )
                    return row

                dataset = dataset.map(_generate_eval_scores)
                dataset_scores, category_scores = aggregate_evaluation_scores(
                    dataset, [FACTUAL_KNOWLEDGE], agg_method=MEAN
                )
                eval_outputs.append(
                    EvalOutput(
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                        prompt_template=dataset_prompt_template,
                        dataset_scores=dataset_scores,
                        category_scores=category_scores,
                        output_path=generate_output_dataset_path(
                            path_to_parent_dir=util.get_eval_results_path(),
                            eval_name=self.eval_name,
                            dataset_name=dataset_config.dataset_name,
                        ),
                    )
                )
            if save:
                save_dataset(
                    dataset=dataset,
                    score_names=[FACTUAL_KNOWLEDGE],
                    path=generate_output_dataset_path(
                        path_to_parent_dir=util.get_eval_results_path(),
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                    ),
                )

        return eval_outputs

    def _get_score(self, target_output: str, model_output: str) -> int:
        """
        Method to generate accuracy score for a target_output and model_output

        :param target_output: Target output
        :param model_output: Model output
        :returns: Accuracy score i.e. O or 1
        """
        possible_targets = target_output.split(self._eval_algorithm_config.target_output_delimiter)
        model_output_lower_case = model_output.lower()

        return int(any([t.lower() in model_output_lower_case for t in possible_targets]))
