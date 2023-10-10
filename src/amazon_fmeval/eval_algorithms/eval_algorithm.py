from abc import ABC, abstractmethod
from typing import List, Optional
from amazon_fmeval.data_loaders.util import DataConfig
from amazon_fmeval.eval_algorithms import EvalScore, EvalOutput
from amazon_fmeval.model_runners.model_runner import ModelRunner

from amazon_fmeval.util import get_eval_results_path, camel_to_snake


class EvalAlgorithmConfig:
    """Config class to be used or extended to provide eval algorithm specific parameters."""


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
        self._eval_results_path = get_eval_results_path()

    def __init_subclass__(cls, **kwargs):
        """
        Method to register algorithms.

        :raises DuplicateEvalNameError if the name of the evaluation being initialized already exists.
        """
        super().__init_subclass__(**kwargs)
        cls.eval_name = camel_to_snake(cls.__name__)

    @abstractmethod
    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        save: bool = False,
    ) -> List[EvalOutput]:
        """
        Computes the evaluation score for dataset(s).

        :param model: An instance of ModelRunner which is the model under evaluation
        :param dataset_config: Configures the single dataset used for evaluation. If not provided,
            evaluation will use all of it's supported built-in datasets
        :param prompt_template: A template which can be used to generate prompts, optional for the built-in datasets.
        :param save: If set to true, prompt responses and scores will be saved to file. The output is written to
                     EvalAlgorithmInterface.EVAL_RESULTS_PATH
        :return: List of EvalOutput objects.
        """

    @abstractmethod
    def evaluate_sample(
        self, model_input: Optional[str] = None, target_output: Optional[str] = None, model_output: Optional[str] = None
    ) -> List[EvalScore]:
        """
        Computes the evaluation score for one instance.

        :param model_output: An instance of ModelOutput which contains the responses from the model needed for this
                             evaluation
        :param model_input: An instance of ModelInput which contains the prompts on which the model needs to be
                            evaluated on
        :param target_output: The expected responses for the prompts in model_input
        :return: list evaluation scores for the sample.
        """
