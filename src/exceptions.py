"""
Error classes for exceptions
"""


class EvalAlgorithmClientError(ValueError):
    """
    Client Error when using Eval Algorithm
    """


class EvalAlgorithmInternalError(Exception):
    """
    Algorithm error when using Eval Algorithm
    """


class DuplicateEvalNameError(EvalAlgorithmClientError):
    """
    Evaluation name already exists.
    """
