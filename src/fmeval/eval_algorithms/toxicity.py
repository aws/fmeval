import logging
from dataclasses import dataclass
from typing import Optional, List, Union, Dict

import ray
from ray.actor import ActorHandle
import numpy as np
from fmeval.constants import DatasetColumns, MEAN
from fmeval.data_loaders.data_config import DataConfig
from fmeval.data_loaders.util import get_dataset
from fmeval.eval_algorithms import (
    EvalOutput,
    EvalScore,
    EvalAlgorithm,
)
from fmeval.eval_algorithms.common import evaluate_dataset
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface, EvalAlgorithmConfig
from fmeval.eval_algorithms.helper_models.helper_model import ToxigenHelperModel, DetoxifyHelperModel, BaseHelperModel
from fmeval.eval_algorithms.save_strategy import SaveStrategy
from fmeval.eval_algorithms.util import (
    get_dataset_configs,
)
from fmeval.transforms.batched_transform import BatchedTransform
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.transforms.util import validate_call
from fmeval.util import get_eval_results_path, create_shared_resource, cleanup_shared_resource
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner

TOXIGEN_MODEL = "toxigen"
DETOXIFY_MODEL = "detoxify"
DEFAULT_MODEL_TYPE = DETOXIFY_MODEL
MODEL_TYPES_SUPPORTED = [TOXIGEN_MODEL, DETOXIFY_MODEL]
TOXICITY_HELPER_MODEL_MAPPING = {TOXIGEN_MODEL: ToxigenHelperModel, DETOXIFY_MODEL: DetoxifyHelperModel}
TOXICITY_BATCH_SIZE = 64

logger = logging.getLogger(__name__)


class ToxicityScores(BatchedTransform):
    """This transform computes toxicity scores on a batch of records at a time using a helper model.

    This transform augments the input batch with the computed scores.
    """

    def __init__(
        self,
        input_key: str,
        toxicity_helper_model: Union[ToxigenHelperModel, DetoxifyHelperModel, ActorHandle],
    ):
        """ToxicityScores initializer.

        :param input_key: The key corresponding to the batch data to be processed by this transform.
        :param toxicity_helper_model: A toxicity helper model instance (see MODEL_TYPES_SUPPORTED
            for the supported helper models) or a Ray actor handle for a helper model.
        """
        super().__init__(input_key, toxicity_helper_model)
        score_names = (
            toxicity_helper_model.get_score_names()
            if (
                isinstance(toxicity_helper_model, ToxigenHelperModel)
                or isinstance(toxicity_helper_model, DetoxifyHelperModel)
            )
            else ray.get(toxicity_helper_model.get_score_names.remote())  # type: ignore
        )
        self.register_input_output_keys(
            input_keys=[input_key],
            output_keys=score_names,
        )
        self.input_key = input_key
        self.toxicity_helper_model = toxicity_helper_model

    @property
    def batch_size(self) -> int:
        """The batch size to use when invoking the toxicity helper model."""
        return TOXICITY_BATCH_SIZE  # pragma: no cover

    @validate_call
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Augment the input batch with toxicity scores computed by the helper model.

        :param batch: The input batch.
        :returns: The input batch with toxicity scores added in.
        """
        text_input: List[str] = batch[self.input_key].tolist()
        scores = (
            self.toxicity_helper_model.get_helper_scores(text_input)
            if isinstance(self.toxicity_helper_model, BaseHelperModel)
            else ray.get(self.toxicity_helper_model.get_helper_scores.remote(text_input))
        )
        for key, value in scores.items():
            batch.update({key: np.array(value)})
        return batch


@dataclass(frozen=True)
class ToxicityConfig(EvalAlgorithmConfig):
    """
    Configuration for the toxicity eval algorithm

    :param model_type: Which toxicity detector to use. Choose between "toxigen" and "detoxify".
    """

    model_type: str = DEFAULT_MODEL_TYPE

    def __post_init__(self):
        if self.model_type not in MODEL_TYPES_SUPPORTED:
            raise EvalAlgorithmClientError(
                f"Invalid model_type: {self.model_type} requested in ToxicityConfig, "
                f"please choose from acceptable values: {MODEL_TYPES_SUPPORTED}"
            )


TOXICITY = EvalAlgorithm.TOXICITY.value


class Toxicity(EvalAlgorithmInterface):
    """
    This evaluation measures whether a model outputs toxic content, and it can be performed over any task that involves the generation of content (including open-ended generation, summarization and question answering). The toxicity score is given by one of two built-in toxicity detectors, "toxigen" and "detoxify". Configure which one to use inside the `ToxicityConfig`.

    Disclaimer: the concept of toxicity is cultural and context dependent. As this evaluation employs a model to score generated passages, the various scores represent the “view” of the toxicity detector used.
    """

    eval_name = TOXICITY

    def __init__(self, eval_algorithm_config: ToxicityConfig = ToxicityConfig()):
        """Toxicity initializer.

        :param eval_algorithm_config: Toxicity evaluation algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self._helper_model = TOXICITY_HELPER_MODEL_MAPPING[eval_algorithm_config.model_type]()

    def evaluate_sample(self, model_output: str) -> List[EvalScore]:  # type: ignore[override]
        """Evaluate toxicity on a single datapoint.

        :param model_output: The output of the model under evaluation.
        :returns: A list of EvalScore objects representing the computed toxicity scores.
        """
        scores = self._helper_model.get_helper_scores([model_output])
        return [EvalScore(name=key, value=value[0]) for key, value in scores.items()]

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[Union[DataConfig, List[DataConfig]]] = None,
        prompt_template: Optional[str] = None,
        num_records: int = 100,
        save: bool = False,
        save_strategy: Optional[SaveStrategy] = None,
    ) -> List[EvalOutput]:
        """Compute toxicity metrics on one or more datasets.

        :param model: An instance of ModelRunner representing the model under evaluation.
            If this argument is None, the `dataset_config` argument must not be None,
            and must correspond to a dataset that already contains a column with model outputs.
        :param dataset_config: Configures a single dataset or list of datasets used for the
            evaluation. If not provided, this method will run evaluations using all of its
            supported built-in datasets.
        :param prompt_template: A template used to generate prompts that are fed to the model.
            If not provided, defaults will be used. If provided, `model` must not be None.
        :param num_records: The number of records to be sampled randomly from the input dataset(s)
            used to perform the evaluation(s).
        :param save: If set to true, prompt responses and scores will be saved to a file.
        :param save_strategy: Specifies the strategy to use the save the localized outputs of the evaluations. If not
            specified, it will save it to the path that can be configured by the EVAL_RESULTS_PATH environment variable.
            If that environment variable is also not configured, it will be saved to the default path `/tmp/eval_results/`.

        :return: A list of EvalOutput objects.
        """
        toxicity_model_shared_resource = create_shared_resource(self._helper_model)
        pipeline = TransformPipeline(
            [
                ToxicityScores(
                    input_key=DatasetColumns.MODEL_OUTPUT.value.name,
                    toxicity_helper_model=toxicity_model_shared_resource,
                )
            ]
        )
        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            eval_output = evaluate_dataset(
                dataset=dataset,
                pipeline=pipeline,
                dataset_name=dataset_config.dataset_name,
                eval_name=self.eval_name,
                metric_names=self._helper_model.get_score_names(),
                eval_results_path=get_eval_results_path(),
                model=model,
                prompt_template=prompt_template,
                agg_method=MEAN,
                save=save,
                save_strategy=save_strategy,
            )
            eval_outputs.append(eval_output)
        cleanup_shared_resource(toxicity_model_shared_resource)
        return eval_outputs
