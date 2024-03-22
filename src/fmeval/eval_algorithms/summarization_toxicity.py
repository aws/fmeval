from fmeval.eval_algorithms import EvalAlgorithm
from fmeval.eval_algorithms.toxicity import Toxicity


class SummarizationToxicity(Toxicity):
    """
    Summarization Toxicity eval algorithm

    Note: This separate eval algo implementation is for mapping Summarization Toxicity specific built-in datasets.
    For consuming toxicity eval algo with your custom dataset please refer and use Toxicity eval algo
    """

    eval_name = EvalAlgorithm.SUMMARIZATION_TOXICITY.value
