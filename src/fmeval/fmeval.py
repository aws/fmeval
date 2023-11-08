import inspect
import json
from typing import Dict, Optional, Union

from fmeval.eval_algo_mapping import EVAL_ALGORITHMS
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface, EvalAlgorithmConfig
from fmeval.exceptions import EvalAlgorithmClientError


def get_eval_algorithm(
    eval_name: str, eval_algorithm_config: Optional[Union[Dict, EvalAlgorithmConfig]] = None
) -> EvalAlgorithmInterface:
    """
    Get eval algorithm class with name

    :param eval_name: eval algorithm name
    :return: eval algorithm class
    """
    if eval_name in EVAL_ALGORITHMS:
        if isinstance(eval_algorithm_config, EvalAlgorithmConfig):
            eval_algorithm_config = json.loads(json.dumps(eval_algorithm_config, default=vars))
        try:
            config_parameters = inspect.signature(EVAL_ALGORITHMS[eval_name]).parameters.get("eval_algorithm_config")
            return (
                EVAL_ALGORITHMS[eval_name](config_parameters.annotation(**eval_algorithm_config))
                if eval_algorithm_config and config_parameters
                else EVAL_ALGORITHMS[eval_name]()
            )
        except TypeError as e:
            raise EvalAlgorithmClientError(
                f"Unable to create algorithm for eval_name {eval_name} with config {eval_algorithm_config}: Error {e}"
            )
    else:
        raise EvalAlgorithmClientError(f"Unknown eval algorithm {eval_name}")
