from abc import ABC, abstractmethod
from typing import List, Optional

from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms import EvalScore, EvalOutput
from fmeval.model_runners.model_runner import ModelRunner


class EvalAlgorithmConfig:
    """Configuration class to be inherited from to provide evaluation algorithm-specific parameters."""


class EvalAlgorithmInterface(ABC):
    """Interface for evaluation algorithms.

    This interface defines two required methods that all evaluation algorithms must implement.
    """

    def __init__(self, eval_algorithm_config: EvalAlgorithmConfig):
        """Initialize an evaluation algorithm instance.

        :param eval_algorithm_config: Contains all configurable parameters for the evaluation algorithm.
        """

    @abstractmethod
    def evaluate_sample(
        self,
        model_input: Optional[str] = None,
        target_output: Optional[str] = None,
        model_output: Optional[str] = None,
    ) -> List[EvalScore]:
        """Compute metrics for a single sample, where a sample is defined by the particular algorithm.

        The `evaluate_sample` method implemented by different algorithms should use a subset of
        these input parameters, but not all of them are required.

        :param model_input: The input passed to `model`. If this parameter is not None,
            `model` should likewise not be None.
        :param target_output: The reference output that `model_output` will be compared against.
        :param model_output: The output from invoking a model.
        :returns: A list of EvalScore objects, where each EvalScore represents a single
            score/metric that is computed by the evaluation algorithm.
        """

    @abstractmethod
    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        num_records: int = 100,
        save: bool = False,
    ) -> List[EvalOutput]:
        """Compute metrics on all samples in one or more datasets.

        :param model: An instance of ModelRunner representing the model being evaluated.
        :param dataset_config: Configures the single dataset used for the evaluation.
            If not provided, this method will run evaluations using all of its supported
            built-in datasets.
        :param prompt_template: A template used to generate prompts from raw text inputs.
            This parameter is not required if you with to run evaluations using the built-in
            datasets, as they have their own default prompt templates pre-configured.
        :param save: If True, model responses and scores will be saved to a file.
            By default, the directory that this output file gets written to is
            DEFAULT_EVAL_RESULTS_PATH, but this directory can be configured through
            the EVAL_RESULTS_PATH environment variable.
        :param num_records: The number of records to be randomly sampled from the input dataset
            that is used for the evaluation.

        :returns: A list of EvalOutput objects, where an EvalOutput encapsulates
        the EvalScores (and optionally, CategoryScores) generated by the evaluation,
        as well as additional metadata regarding the evaluation.
        """
