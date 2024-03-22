from fmeval.eval_algorithms import (
    EvalAlgorithm,
)
from fmeval.eval_algorithms.helper_models.helper_model import ToxigenHelperModel, DetoxifyHelperModel
from fmeval.eval_algorithms.toxicity import Toxicity, ToxicityConfig

TOXIGEN_MODEL = "toxigen"
DETOXIFY_MODEL = "detoxify"

TOXICITY_HELPER_MODEL_MAPPING = {TOXIGEN_MODEL: ToxigenHelperModel, DETOXIFY_MODEL: DetoxifyHelperModel}

QA_TOXICITY = EvalAlgorithm.QA_TOXICITY.value


class QAToxicity(Toxicity):
    """
    Toxicity evaluation specific to the QA task on our built-in dataset. As for the general toxicity evaluation, the toxicity score is given by one of two built-in toxicity detectors, "toxigen" and "detoxify". Configure which one to use inside the `ToxicityConfig`.

    Disclaimer: the concept of toxicity is cultural and context dependent. As this evaluation employs a model to score generated passages, the various scores represent the “view” of the toxicity detector used.

    Note: This separate eval algo implementation is for use with the built-in QA datasets. For consuming
    toxicity eval algo with your custom dataset please refer and use the general Toxicity eval algo.
    """

    def __init__(self, eval_algorithm_config: ToxicityConfig = ToxicityConfig()):
        """Default constructor

        :param eval_algorithm_config: Toxicity eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.eval_name = QA_TOXICITY
        self._eval_algorithm_config = eval_algorithm_config
        self._helper_model = TOXICITY_HELPER_MODEL_MAPPING[self._eval_algorithm_config.model_type]()
