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

    :param eval_name: The name of the evaluation
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
        Method to register algorithms.

        :raises DuplicateEvalNameError if the name of the evaluation being initialized already exists.
        """
        super().__init_subclass__(**kwargs)
        cls.eval_name = camel_to_snake(cls.__name__)
        if cls.eval_name in EVAL_ALGORITHMS.keys():
            raise AlgorithmError(
                f"Eval algorithm with name {cls.eval_name} already exists. " f"Please rename the class {cls.__name__}"
            )

    @abstractmethod
    def evaluate(self, model: Optional[ModelRunner], custom_dataset_config: Optional[DataConfig] = None,
                 prompt_template: str = None, save: bool = False) -> EvalOutput:
        """
        Computes the evaluation score for dataset(s).

        :param model: An instance of ModelRunner which is the model under evaluation
        :param custom_dataset_config: Custom dataset for evaluation. If not provided, evaluation will use
                                      the built-in datasets.
        :param prompt_template: A template which can be used to generate prompts which is optional for
                                      the built-in datasets.
        :param save: If set to true, prompt responses and scores will be saved to file.
        :return: evaluation scores for the dataset.
        """

    @abstractmethod
    def evaluate_sample(
        self, model_input: Optional[str] = None, target_output: Optional[str] = None, model_output: Optional[str] = None
    ) -> float:
        """
        Computes the evaluation score for one instance.

        :param model_output: An instance of ModelOutput which contains the responses from the model needed for this
                             evaluation
        :param model_input: An instance of ModelInput which contains the prompts on which the model needs to be
                            evaluated on
        :param target_output: The expected responses for the prompts in model_input
        :return: evaluation score for the sample.
        """


