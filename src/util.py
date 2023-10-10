import os
import re

from constants import EVAL_RESULTS_PATH, DEFAULT_EVAL_RESULTS_PATH
from exceptions import EvalAlgorithmInternalError, EvalAlgorithmClientError


def require(expression, msg: str):
    """
    Raise EvalAlgorithmClientError if expression is not True
    """
    if not expression:
        raise EvalAlgorithmClientError(msg)


def assert_condition(expression, msg: str):
    """
    Raise EvalAlgorithmInternalError if expression is not True
    """
    if not expression:
        raise EvalAlgorithmInternalError(msg)


def project_root(current_file: str) -> str:
    """
    :return: project root
    """
    curpath = os.path.abspath(os.path.dirname(current_file))

    def is_project_root(path: str) -> bool:
        return os.path.exists(os.path.join(path, ".root"))

    while not is_project_root(curpath):  # pragma: no cover
        parent = os.path.abspath(os.path.join(curpath, os.pardir))
        if parent == curpath:
            raise EvalAlgorithmInternalError("Got to the root and couldn't find a parent folder with .root")
        curpath = parent
    return curpath


def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def get_eval_results_path():
    """
    Util method to return results path for eval_algos. This method looks for EVAL_RESULTS_PATH environment variable,
    if present returns that else default path
    :returns: Local directory path of eval algo results
    """
    if os.environ.get(EVAL_RESULTS_PATH) is not None:
        os.makedirs(os.environ[EVAL_RESULTS_PATH], exist_ok=True)
        return os.environ[EVAL_RESULTS_PATH]
    else:
        os.makedirs(DEFAULT_EVAL_RESULTS_PATH, exist_ok=True)
        return DEFAULT_EVAL_RESULTS_PATH
