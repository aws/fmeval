import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from fmeval import util
from fmeval.constants import DatasetColumns
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms import (
    EvalOutput,
    EvalScore,
    EvalAlgorithm,
)
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface, EvalAlgorithmConfig
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.helper_models import ToxigenModel, DetoxifyModel, ToxicityDetector
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.toxicity import ToxicityScores
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.util import create_shared_resource, require, get_eval_results_path
from fmeval.eval_algorithms import util

TOXIGEN_MODEL = "toxigen"
DETOXIFY_MODEL = "detoxify"
DEFAULT_MODEL_TYPE = DETOXIFY_MODEL
MODEL_TYPES_SUPPORTED = [TOXIGEN_MODEL, DETOXIFY_MODEL]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToxicityConfig(EvalAlgorithmConfig):
    """
    Configuration for the toxicity eval algorithm

    :param model_type: model to use for toxicity eval
    """

    model_type: str = DEFAULT_MODEL_TYPE

    def __post_init__(self):
        if self.model_type not in MODEL_TYPES_SUPPORTED:
            raise EvalAlgorithmClientError(
                f"Invalid model_type: {self.model_type} requested in ToxicityConfig, "
                f"please choose from acceptable values: {MODEL_TYPES_SUPPORTED}"
            )

    def get_model(self) -> ToxicityDetector:
        if self.model_type == TOXIGEN_MODEL:
            return ToxigenModel()
        # add here new models (note that checks are done on __post_init__)
        else:
            return DetoxifyModel()


class Toxicity(EvalAlgorithmInterface):
    """
    Toxicity evaluation algorithm
    """

    eval_name = EvalAlgorithm.TOXICITY.value

    def __init__(self, config: ToxicityConfig = ToxicityConfig(), use_ray: bool = True):
        """Toxicity evaluation initializer.

        :param config: toxicity evaluation config.
        :param use_ray: Whether to create a Ray actor for the toxicity detector model used by this evaluation
            algorithm instance. Currently, `evaluate` will only work if `use_ray` is set to True,
            as the execution of the transform pipeline relies on the BertscoreModel existing
            in shared memory. This flag can be set to False if you only plan on invoking the
            `evaluate_sample` method, which is a computationally cheap operation that does not
            require utilizing Ray for parallel execution.
        """
        self.use_ray = use_ray
        model = config.get_model()
        self.score_names = model.SCORE_NAMES
        if self.use_ray:
            model = create_shared_resource(model)
        self.toxicity_scores = ToxicityScores(
            text_keys=[DatasetColumns.MODEL_OUTPUT.value.name], score_names=self.score_names, toxicity_detector=model
        )
        self.pipeline = TransformPipeline([self.toxicity_scores])

    @staticmethod
    def create_sample(model_output: str) -> Dict[str, Any]:
        """Create a sample in the record format used by Transforms.

        This function's primary use is to be called by evaluate_sample.

        :param model_output: The model_output parameter passed to evaluate_sample.
        """
        return {
            DatasetColumns.MODEL_OUTPUT.value.name: [model_output],
            # here we put a list around because the toxicity scores transform is batched
        }

    def evaluate_sample(self, model_output: str) -> List[EvalScore]:  # type: ignore[override]
        """
        Toxicity evaluate sample

        :param model_output: The output of a model that we want to evaluate.
        :return: list of EvalScore objects
        """
        sample = Toxicity.create_sample(model_output)
        output_record = self.pipeline.execute_record(sample)
        return [EvalScore(name=metric_name, value=output_record[metric_name]) for metric_name in self.score_names]

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        num_records: int = 100,
        save: bool = False,
    ) -> List[EvalOutput]:
        """Compute toxicity scores on one or more datasets.

        :param model: An instance of ModelRunner representing the model under evaluation.
        :param dataset_config: Configures the single dataset used for evaluation.
            If not provided, evaluation will use all of its supported built-in datasets.
        :param prompt_template: A template used to generate prompts that are fed to the model.
            If not provided, defaults will be used.
        :param num_records: The number of records to be sampled randomly from the input dataset
            used to perform the evaluation.
        :param save: If set to true, prompt responses and scores will be saved to a file.
            The path that this file is stored at is configured by `eval_results_path`.

        :return: A list of EvalOutput objects.
        """
        require(
            self.use_ray,
            "The use_ray instance attribute of SummarizationAccuracy must be True in order "
            "for the evaluate method to run successfully.",
        )
        return util.run_evaluation(
            eval_name=self.eval_name,
            pipeline=self.pipeline,
            metric_names=self.toxicity_scores.output_keys,
            required_columns=[DatasetColumns.MODEL_INPUT.value.name],
            eval_results_path=get_eval_results_path(),
            model=model,
            dataset_config=dataset_config,
            prompt_template=prompt_template,
            num_records=num_records,
            save=save,
        )
