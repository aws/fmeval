from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from constants import MIME_TYPE_JSON, MIME_TYPE_JSONLINES
from data_loaders.util import DataConfig
from eval_algorithms import EvalAlgorithm, EvalScore, EvalOutput
from model_runners.model_runner import ModelRunner

# Constants for Built-in dataset names
from util import get_eval_results_path, camel_to_snake

TREX = "trex"
BOOLQ = "boolq"
TRIVIA_QA = "trivia_qa"
NATURAL_QUESTIONS = "natural_questions"
CROW_PAIRS = "crow-pairs"
CNN_DAILY_MAIL = "cnn_daily_mail"
XSUM = "xsum"


class EvalAlgorithmConfig:
    """Config class to be used or extended to provide eval algorithm specific parameters."""


# Mapping of Eval algorithms and corresponding Built-in datasets
EVAL_DATASETS: Dict[str, List[str]] = {
    EvalAlgorithm.FACTUAL_KNOWLEDGE.value: [TREX],
    EvalAlgorithm.QA_ACCURACY.value: [BOOLQ, TRIVIA_QA, NATURAL_QUESTIONS],
    EvalAlgorithm.PROMPT_STEREOTYPING.value: [CROW_PAIRS],
    EvalAlgorithm.SUMMARIZATION_ACCURACY.value: [CNN_DAILY_MAIL, XSUM],
}

# Mapping of Default Prompt Template corresponding to eval, built-in dataset pair
# TODO: To be correctly populated
EVAL_PROMPT_TEMPLATES: Dict[Tuple[str, str], str] = {
    (EvalAlgorithm.FACTUAL_KNOWLEDGE.value, TREX): "Answer: $feature",
    (EvalAlgorithm.QA_ACCURACY.value, BOOLQ): "$feature",
    (EvalAlgorithm.QA_ACCURACY.value, TRIVIA_QA): "$feature",
    (EvalAlgorithm.QA_ACCURACY.value, NATURAL_QUESTIONS): "$feature",
    (EvalAlgorithm.PROMPT_STEREOTYPING.value, CROW_PAIRS): "$feature",
    (EvalAlgorithm.SUMMARIZATION_ACCURACY.value, CNN_DAILY_MAIL): "Summarise: $feature",
    (EvalAlgorithm.SUMMARIZATION_ACCURACY.value, XSUM): "Summarise: $feature",
}

# Mapping of Built-in dataset names and their DataConfigs
# TODO: To be updated once datasets are uploaded in S3, update Configs accordingly
DATASET_CONFIGS: Dict[str, DataConfig] = {
    TREX: DataConfig(
        dataset_name=TREX,
        dataset_uri="tba",
        dataset_mime_type=MIME_TYPE_JSON,
        model_input_location="tba",
        target_output_location="tba",
        model_output_location=None,
        category_location="tba",
    ),
    BOOLQ: DataConfig(
        dataset_name="boolq",
        dataset_uri="tba",
        dataset_mime_type=MIME_TYPE_JSON,
        model_input_location="tba",
        target_output_location="tba",
        model_output_location=None,
        category_location="tba",
    ),
    TRIVIA_QA: DataConfig(
        dataset_name="trivia_qa",
        dataset_uri="tba",
        dataset_mime_type=MIME_TYPE_JSON,
        model_input_location="tba",
        target_output_location="tba",
        model_output_location=None,
        category_location="tba",
    ),
    NATURAL_QUESTIONS: DataConfig(
        dataset_name="natural_questions",
        dataset_uri="tba",
        dataset_mime_type=MIME_TYPE_JSON,
        model_input_location="tba",
        target_output_location="tba",
        model_output_location=None,
        category_location="tba",
    ),
    CROW_PAIRS: DataConfig(
        dataset_name=CROW_PAIRS,
        dataset_uri="tba",
        dataset_mime_type=MIME_TYPE_JSONLINES,
        sent_more_input_location="sent_more",
        sent_less_input_location="sent_less",
    ),
    CNN_DAILY_MAIL: DataConfig(
        dataset_name="cnn_daily_mail",
        dataset_uri="tba",
        dataset_mime_type=MIME_TYPE_JSON,
        model_input_location="tba",
        target_output_location="tba",
        model_output_location=None,
        category_location="tba",
    ),
    XSUM: DataConfig(
        dataset_name="xsum",
        dataset_uri="tba",
        dataset_mime_type=MIME_TYPE_JSON,
        model_input_location="tba",
        target_output_location="tba",
        model_output_location=None,
        category_location="tba",
    ),
}


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
