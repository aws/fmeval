from abc import ABC, abstractmethod
from collections import UserDict
from typing import List, Optional, Type, Union
from dataclasses import dataclass

from data_loaders.dataset_factory import DataLoader
from eval_algorithms.exceptions import DuplicateEvalNameError
from eval_algorithms.util import camel_to_snake
from model_runners.model_runner import ModelRunner

class EvalAlgorithmRegistry(UserDict[str, Type["EvalAlgorithmInterface"]]):
    """
    Eval algorithm registry.

    This is a singleton class maintains a list of all evals that are available - that is, it maintains a list of
    all class that implement the `EvalAlgorithmInterface`.
    """

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(EvalAlgorithmRegistry, cls).__new__(cls)
        return cls.instance

    def __setitem__(self, key: str, value: Type["EvalAlgorithmInterface"]):
        """
        Set eval algorithms.

        :param key: eval algorithm name
        :param value: eval algorithm class
        """
        if "EvalAlgorithmInterface" in (c.__name__ for c in value.__mro__):
            super().__setitem__(key, value)
        else:
            raise ValueError("Please inherit the eval algorithm from EvalAlgorithmInterface")

    def get_eval_algorithm(self, eval_type: str) -> Type["EvalAlgorithmInterface"]:
        """
        Get eval algorithm class with name

        :param eval_type: eval algorithm name
        :return: eval algorithm class
        """
        if eval_type in self:
            return self[eval_type]
        else:
            raise KeyError(f"Unknown eval algorithm {eval_type}")


ALGORITHM_REGISTRY = EvalAlgorithmRegistry()


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
    The class that contains aggregate scores computed in the `aggregate()` function of `EvalAlgorithmInterface`.

    :param eval_type: The name of the evaluation
    :param dataset_score: The aggregation score computed across the whole dataset.
    :param category_scores: A list of CategoryScore object that contain the scores for each category in the dataset.
    """

    eval_type: str
    dataset_score: float
    category_scores: Optional[List[CategoryScore]] = None


@dataclass(frozen=True)
class AggregationInput:
    """
    Dataclass for inputs for the `aggregate()` method of `EvalAlgorithmInterface`.

    :param eval_outputs: List of EvalOutput instances returned by `evaluate()` method of EvalAlgorithmInterface
    :param categories: List of categories corresponding to EvalOutput, if provided `aggregate()` method will return
    both dataset score and category based scores.
    """

    eval_outputs: List[float]
    categories: Optional[List[str]]

@dataclass
class DataConfig:
    dataset_name: str
    dataset_uri: str
    prompt_template: str
    model_input_jmespaths: List[str]
    model_output_jmespaths: List[str]
    target_output_jmespath: Optional[str]


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
        self.model = model

    def __init_subclass__(cls, **kwargs):
        """
        Method to register algorithms in the EvalAlgorithmRegistry.

        :raises DuplicateEvalNameError if the name of the evaluation being initialized already exists.
        """
        super().__init_subclass__(**kwargs)
        cls.eval_name = camel_to_snake(cls.__name__)
        if cls.eval_name in ALGORITHM_REGISTRY.keys():
            raise DuplicateEvalNameError(
                f"Eval algorithm with name {cls.eval_name} already exists. " f"Please rename the class {cls.__name__}"
            )
        ALGORITHM_REGISTRY[cls.eval_name] = cls

    def evaluate(self, model: Optional[ModelRunner], dataset_name: Optional[str] = None,
             custom_dataset_config: Optional[DataConfig] = None):
        dataset = custom_dataset_config
        if dataset_name:
            if dataset_name and dataset_name not in EVAL_DATASETS[self.eval_name]:
                raise EvalAlgorithmClientError()
            dataset = DATASET_CONFIGS[dataset_name]
        data = DataLoader.get_dataset(dataset)



    @abstractmethod
    def eval_sample(
        self, model_input: Optional[str] = None, target_output: Optional[str] = None, model_output: Optional[str] = None
    ) -> float:
        """
        Computes the evaluation score for each instance.

        :param model_output: An instance of ModelOutput which contains the responses from the model needed for this
                             evaluation
        :param model_input: An instance of ModelInput which contains the prompts on which the model needs to be
                            evaluated on
        :param target_output: The expected responses for the prompts in model_input
        :return: an instance of EvalOutput that contains the score computed for prompts and responses.
        """

    @abstractmethod
    def aggregate(self, eval_outputs: List[float], categories: List[str]) -> EvalOutput:
        """
        Computes the aggregate score for the model based on a list of individual evaluate responses.

        :param eval_outputs: A list of EvalOutput objects obtained from the evaluate function of the same class
        :return: an instance of AggregationOutput that contains the aggregation result
        """
