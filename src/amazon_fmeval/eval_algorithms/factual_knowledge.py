import logging
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd

from amazon_fmeval.constants import (
    MODEL_OUTPUT_COLUMN_NAME,
    TARGET_OUTPUT_COLUMN_NAME,
    MODEL_INPUT_COLUMN_NAME,
    MODEL_LOG_PROBABILITY_COLUMN_NAME,
    MEAN,
)
import amazon_fmeval.util as util
from amazon_fmeval.data_loaders.util import get_dataset
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.eval_algorithms.util import save_dataset, generate_output_dataset_path
from amazon_fmeval.eval_algorithms.eval_algorithm import (
    EvalAlgorithmInterface,
    EvalAlgorithmConfig,
)
from amazon_fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalOutput,
    EvalScore,
    EVAL_DATASETS,
    EVAL_PROMPT_TEMPLATES,
    DATASET_CONFIGS,
)
from amazon_fmeval.eval_algorithms.util import (
    generate_model_predict_response_for_dataset,
    generate_prompt_column_for_dataset,
    aggregate_evaluation_scores,
    validate_dataset,
)
from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.model_runner import ModelRunner
from amazon_fmeval.perf_util import timed_block

PROMPT_COLUMN_NAME = "prompt"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactualKnowledgeConfig(EvalAlgorithmConfig):
    """
    Configuration for the factual knowledge eval algorithm

    :param target_output_delimiter: Target Output can have multiple answers. We expect customer to combine all the
        possible answers into a single string and use the delimiter to separate them. For instance,
        if the answers are ["UK", "England"] and the delimiter="<OR>", then the target_output should be "UK<OR>England".
    """

    target_output_delimiter: str


SCORE_NAME = "factual_knowledge"


class FactualKnowledge(EvalAlgorithmInterface):
    """
    Factual Knowledge Eval algorithm
    """

    def __init__(self, eval_algorithm_config: FactualKnowledgeConfig):
        """Default constructor

        :param eval_algorithm_config: Factual knowledge eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.eval_name = EvalAlgorithm.FACTUAL_KNOWLEDGE.value
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
        :param model_output: An instance of ModelOutput which contains the responses from the model needed for this
                             evaluation
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
        :param prompt_template: A template which can be used to generate prompts, optional for the built-in datasets.
        :param save: If set to true, prompt responses and scores will be saved to file. The output is written to
                     EvalAlgorithmInterface.EVAL_RESULTS_PATH
        :return: List of EvalOutput objects. Current implementation returns only one score.
        """
        is_custom_dataset_evaluation = False
        if dataset_config:
            is_custom_dataset_evaluation = True
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[self.eval_name]]

        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config)
            validate_dataset(dataset, [TARGET_OUTPUT_COLUMN_NAME, MODEL_INPUT_COLUMN_NAME])
            if MODEL_OUTPUT_COLUMN_NAME not in dataset.columns():
                util.require(model, "No ModelRunner provided. ModelRunner is required for inference on model_inputs")
                if is_custom_dataset_evaluation:
                    # TODO when user provide built-in DataConfig, we should provide default prompt_template
                    util.require(
                        prompt_template,
                        f"Missing required input: prompt_template for evaluating custom dataset : {dataset_config}",
                    )
                else:
                    prompt_template = EVAL_PROMPT_TEMPLATES[self.eval_name, dataset_config.dataset_name]
                    util.assert_condition(
                        prompt_template is not None,
                        f"No Prompt Template configured for ({self.eval_name}, {dataset_config.dataset_name})",
                    )
                assert prompt_template  # to satisfy mypy
                dataset = generate_prompt_column_for_dataset(
                    prompt_template, dataset, MODEL_INPUT_COLUMN_NAME, PROMPT_COLUMN_NAME
                )
                assert model  # to satisfy mypy
                dataset = generate_model_predict_response_for_dataset(
                    model=model,
                    data=dataset,
                    model_input_column_name=PROMPT_COLUMN_NAME,
                    model_output_column_name=MODEL_OUTPUT_COLUMN_NAME,
                )
            with timed_block(f"Computing score and aggregation on dataset {dataset_config.dataset_name}", logger):

                def _generate_eval_scores(df: pd.DataFrame) -> pd.Series:  # pragma: no cover
                    """
                    Map function generating the scores for every input record in input dataset
                    """
                    return pd.Series(
                        data=[
                            self._get_score(row[TARGET_OUTPUT_COLUMN_NAME], row[MODEL_OUTPUT_COLUMN_NAME])
                            for index, row in df.iterrows()
                        ]
                    )

                dataset = dataset.add_column(SCORE_NAME, _generate_eval_scores)
                dataset_scores, category_scores = aggregate_evaluation_scores(dataset, [SCORE_NAME], agg_method=MEAN)
                eval_outputs.append(
                    EvalOutput(
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                        prompt_template=prompt_template,
                        dataset_scores=dataset_scores,
                        category_scores=category_scores,
                        output_path=self._eval_results_path,
                    )
                )
            if save:
                save_dataset(
                    dataset=dataset,
                    score_names=[SCORE_NAME],
                    path=generate_output_dataset_path(
                        path_to_parent_dir=self._eval_results_path,
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
