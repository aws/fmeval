from fmeval.eval_algorithms import EvalAlgorithm
from fmeval.eval_algorithms.toxicity import Toxicity


class QAToxicity(Toxicity):
    """
    QA Toxicity eval algorithm

    Note: This separate eval algo implementation is for mapping QA Toxicity specific built-in datasets. For consuming
    toxicity eval algo with your custom dataset please refer and use Toxicity eval algo
    """

    eval_name = EvalAlgorithm.QA_TOXICITY.value
