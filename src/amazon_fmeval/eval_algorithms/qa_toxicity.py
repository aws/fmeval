import logging

from amazon_fmeval.eval_algorithms import (
    EvalAlgorithm,
)
from amazon_fmeval.eval_algorithms.helper_models.helper_model import ToxigenHelperModel, DetoxifyHelperModel
from amazon_fmeval.eval_algorithms.toxicity import Toxicity, ToxicityConfig

TOXIGEN_MODEL = "toxigen"
DETOXIFY_MODEL = "detoxify"
DEFAULT_MODEL_TYPE = DETOXIFY_MODEL
MODEL_TYPES_SUPPORTED = [TOXIGEN_MODEL, DETOXIFY_MODEL]

TOXICITY_HELPER_MODEL_MAPPING = {TOXIGEN_MODEL: ToxigenHelperModel, DETOXIFY_MODEL: DetoxifyHelperModel}

PROMPT_COLUMN_NAME = "prompt"

logger = logging.getLogger(__name__)

QA_TOXICITY = EvalAlgorithm.QA_TOXICITY.value


class QAToxicity(Toxicity):
    """
    QA Toxicity eval algorithm

    Note: This separate eval algo implementation is for mapping QA Toxicity specific built-in datasets. For consuming
    toxicity eval algo with your custom dataset please refer and use Toxicity eval algo
    """

    def __init__(self, eval_algorithm_config: ToxicityConfig):
        """Default constructor

        :param eval_algorithm_config: Toxicity eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.eval_name = QA_TOXICITY
        self._eval_algorithm_config = eval_algorithm_config
        self._helper_model = TOXICITY_HELPER_MODEL_MAPPING[self._eval_algorithm_config.model_type]()
