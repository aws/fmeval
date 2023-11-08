import logging
from dataclasses import dataclass
from typing import Optional, List

import ray
from ray.data import Dataset

from fmeval import util
from fmeval.constants import MODEL_INPUT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME, MEAN
from fmeval.data_loaders.data_config import DataConfig
from fmeval.data_loaders.util import get_dataset
from fmeval.eval_algorithms import (
    EvalOutput,
    EvalScore,
    EvalAlgorithm,
    DATASET_CONFIGS,
    EVAL_DATASETS,
    get_default_prompt_template,
)
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmConfig, EvalAlgorithmInterface
from fmeval.eval_algorithms.helper_models.helper_model import ToxigenHelperModel, DetoxifyHelperModel
from fmeval.eval_algorithms.util import (
    validate_dataset,
    generate_prompt_column_for_dataset,
    generate_model_predict_response_for_dataset,
    aggregate_evaluation_scores,
    save_dataset,
    generate_output_dataset_path,
)
from fmeval.util import get_num_actors
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.perf_util import timed_block

TOXIGEN_MODEL = "toxigen"
DETOXIFY_MODEL = "detoxify"
DEFAULT_MODEL_TYPE = DETOXIFY_MODEL
MODEL_TYPES_SUPPORTED = [TOXIGEN_MODEL, DETOXIFY_MODEL]

TOXICITY_HELPER_MODEL_MAPPING = {TOXIGEN_MODEL: ToxigenHelperModel, DETOXIFY_MODEL: DetoxifyHelperModel}

PROMPT_COLUMN_NAME = "prompt"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToxicityConfig(EvalAlgorithmConfig):
    """
    Configuration for the toxicity eval algorithm

    :param model_type: model to use for toxicity eval
    """

    model_type: str = DEFAULT_MODEL_TYPE

    def __post_init__(self):
        if not self.model_type in MODEL_TYPES_SUPPORTED:
            raise EvalAlgorithmClientError(
                f"Invalid model_type: {self.model_type} requested in ToxicityConfig, "
                f"please choose from acceptable values: {MODEL_TYPES_SUPPORTED}"
            )


TOXICITY = EvalAlgorithm.TOXICITY.value


class Toxicity(EvalAlgorithmInterface):
    """
    Toxicity eval algorithm
    """

    def __init__(self, eval_algorithm_config: ToxicityConfig = ToxicityConfig()):
        """Default constructor

        :param eval_algorithm_config: Toxicity eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.eval_name = TOXICITY
        self._eval_algorithm_config = eval_algorithm_config
        self._helper_model = TOXICITY_HELPER_MODEL_MAPPING[self._eval_algorithm_config.model_type]()

    def evaluate_sample(self, model_output: str) -> List[EvalScore]:  # type: ignore[override]
        """
        Toxicity evaluate sample

        :param model_output: The output of a model that we want to evaluate.
        :return: list of EvalScore objects
        """
        util.require(model_output, "Missing required input: target_output, for Toxicity evaluate_sample")
        scores = self._helper_model.get_helper_scores([model_output])
        return [EvalScore(name=key, value=value[0]) for key, value in scores.items()]

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        save: bool = False,
        num_records: int = 100,
    ) -> List[EvalOutput]:
        """
        Toxicity evaluate

        :param model: An instance of ModelRunner which is the model under evaluation
        :param dataset_config: The config to load the dataset to use for evaluation. If not provided, model will be
                               evaluated on all built-in datasets configured for this evaluation.
        :param prompt_template: A template which can be used to generate prompts, optional, if not provided defaults
            will be used.
        :param save: If set to true, prompt responses and scores will be saved to file. The output is written to
                     EvalAlgorithmInterface.EVAL_RESULTS_PATH
        :param num_records: The number of records to be sampled randomly from the input dataset to perform the
                            evaluation
        :return: List of EvalOutput objects.
        """
        if dataset_config:
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[self.eval_name]]

        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [MODEL_INPUT_COLUMN_NAME])
            dataset_prompt_template = None
            if MODEL_OUTPUT_COLUMN_NAME not in dataset.columns():
                util.require(model, "No ModelRunner provided. ModelRunner is required for inference on model_inputs")
                dataset_prompt_template = (
                    get_default_prompt_template(dataset_config.dataset_name) if not prompt_template else prompt_template
                )
                dataset = generate_prompt_column_for_dataset(
                    dataset_prompt_template, dataset, MODEL_INPUT_COLUMN_NAME, PROMPT_COLUMN_NAME
                )
                assert model  # to satisfy mypy
                dataset = generate_model_predict_response_for_dataset(
                    model=model,
                    data=dataset,
                    model_input_column_name=PROMPT_COLUMN_NAME,
                    model_output_column_name=MODEL_OUTPUT_COLUMN_NAME,
                )
            with timed_block(f"Computing score and aggregation on dataset {dataset_config.dataset_name}", logger):

                dataset = self.__add_scores(dataset)
                dataset_scores, category_scores = aggregate_evaluation_scores(
                    dataset, self._helper_model.get_score_names(), agg_method=MEAN
                )
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
                    score_names=self._helper_model.get_score_names(),
                    path=generate_output_dataset_path(
                        path_to_parent_dir=self._eval_results_path,
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                    ),
                )

        return eval_outputs

    def __add_scores(self, dataset: Dataset) -> Dataset:  # pragma: no cover
        """
        Private method to encapsulate making map_batch call to helper model

        :param dataset: input dataset with prompts
        :returns: Materialised ray dataset with score columns added
        """
        return dataset.map_batches(
            fn=TOXICITY_HELPER_MODEL_MAPPING[self._eval_algorithm_config.model_type],
            fn_constructor_args=(MODEL_OUTPUT_COLUMN_NAME,),
            compute=ray.data.ActorPoolStrategy(size=get_num_actors()),
        ).materialize()
