from abc import ABC, abstractmethod
from collections import UserDict
from typing import List, Optional, Type, Union
from dataclasses import dataclass
from util import EVAL_ALGORITHMS, EVAL_DATASETS, DATASET_CONFIGS, camel_to_snake
from ..exceptions import UserError, AlgorithmError
from ..model_runners.model_runner import ModelRunner
from ..data_loaders.data_loader import DataLoader
from ..data_loaders.util import DataConfig

class EvalAlgorithmConfig:
    """Config class to be used or extended to provide eval algorithm specific parameters."""


@dataclass(frozen=True)
class CategoryScore:
    """The class that contains the aggregated scores computed across specific categories in the dataset.

    :param name: The name of the category.
    :param score: The aggregated score computed for the given category.
    """

    name: str
    score: float


@dataclass(frozen=True)
class EvalOutput:
    """
    The class that contains evaluation scores from `EvalAlgorithmInterface`.

    :param eval_type: The name of the evaluation
    :param dataset_score: The aggregated score computed across the whole dataset.
    :param sample_scores: The score(s) for each individual sample (row) in the dataset.
    :param categories: The category for each individual sample (row) in the dataset.
    :param category_scores: A list of CategoryScore object that contain the scores for each category in the dataset.
    """

    eval_name: str
    dataset_score: float
    category_scores: Optional[List[CategoryScore]] = None
    output_path: str = None

class EvalAlgorithmInterface(ABC):
    """
    Interface class for eval algorithms.

    All the eval algorithm classes inheriting this interface will be registered in the registry.
    """

    eval_name: str

    def __init__(self, eval_algorithm_config: EvalAlgorithmConfig):
        """Initialize an instance of a subclass of EvalAlgorithmConfig

        :param eval_algorithm_config: An instance of the subclass of EvalAlgorithmConfig specific to the
                                            current evaluation.
        """
        self.eval_algorithm_config = eval_algorithm_config

    def __init_subclass__(cls, **kwargs):
        """
        Method to register algorithms in the EvalAlgorithmRegistry.

        :raises DuplicateEvalNameError if the name of the evaluation being initialized already exists.
        """
        super().__init_subclass__(**kwargs)
        cls.eval_name = camel_to_snake(cls.__name__)
        if cls.eval_name in EVAL_ALGORITHMS.keys():
            raise AlgorithmError(
                f"Eval algorithm with name {cls.eval_name} already exists. " f"Please rename the class {cls.__name__}"
            )

    def evaluate(self, model: Optional[ModelRunner], dataset_name: Optional[str] = None,
             custom_dataset_config: Optional[DataConfig] = None, promtp_template: str = None, save: bool = False) -> EvalOutput:
        dataset_name = custom_dataset_config.dataset_name
        if dataset_name:
            if dataset_name and dataset_name not in EVAL_DATASETS[self.eval_name]:
                raise UserError()
            dataset = DATASET_CONFIGS[dataset_name]
        dataset = DataLoader.get_dataset(dataset_name)
        dataset_score = dataset.map(self.evaluate_sample).avg()

        if save:
            ## TODO: Write model input, model output, sample scores to local file
            pass

        return EvalOutput(
            eval_name=self.EVAL_NAME,
            dataset_score=dataset_score,
        )

    @abstractmethod
    def evaluate_sample(
        self, model_input: Optional[str] = None, target_output: Optional[str] = None, model_output: Optional[str] = None
    ) -> float:
        """
        Computes the evaluation score for each instance.

        :param model_output: An instance of ModelOutput which contains the responses from the model needed for this
                             evaluation
        :param model_input: An instance of ModelInput which contains the prompts on which the model needs to be
                            evaluated on
        :param target_output: The expected responses for the prompts in model_input
        :return: evaluation score for the sample.
        """


