from typing import Type

from amazon_fmeval.eval_algo_mapping import EVAL_ALGORITHMS
from amazon_fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface
from amazon_fmeval.exceptions import EvalAlgorithmClientError


def get_eval_algorithm(eval_name: str) -> Type[EvalAlgorithmInterface]:
    """
    Get eval algorithm class with name

    :param eval_name: eval algorithm name
    :return: eval algorithm class
    """
    if eval_name in EVAL_ALGORITHMS:
        return EVAL_ALGORITHMS[eval_name]
    else:
        raise EvalAlgorithmClientError(f"Unknown eval algorithm {eval_name}")
