from fmeval.eval_algorithms import (
    EvalAlgorithm,
)
from fmeval.eval_algorithms.helper_models.helper_model import ToxigenHelperModel, DetoxifyHelperModel
from fmeval.eval_algorithms.toxicity import Toxicity, ToxicityConfig

TOXIGEN_MODEL = "toxigen"
DETOXIFY_MODEL = "detoxify"

TOXICITY_HELPER_MODEL_MAPPING = {TOXIGEN_MODEL: ToxigenHelperModel, DETOXIFY_MODEL: DetoxifyHelperModel}

SUMMARIZATION_TOXICITY = EvalAlgorithm.SUMMARIZATION_TOXICITY.value


class SummarizationToxicity(Toxicity):
    """
    Summarization Toxicity eval algorithm

    Note: This separate eval algo implementation is for mapping Summarization Toxicity specific built-in datasets.
    For consuming toxicity eval algo with your custom dataset please refer and use Toxicity eval algo
    """

    def __init__(self, eval_algorithm_config: ToxicityConfig = ToxicityConfig()):
        """Default constructor

        :param eval_algorithm_config: Toxicity eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.eval_name = SUMMARIZATION_TOXICITY
        self._eval_algorithm_config = eval_algorithm_config
        self._helper_model = TOXICITY_HELPER_MODEL_MAPPING[self._eval_algorithm_config.model_type]()
