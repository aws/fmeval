"""
Error classes for exceptions
"""


class EvalAlgorithmClientError(ValueError):
    """
    Client Error when using Eval Algorithm
    """


class EvalAlgorithmSystemError(Exception):
    """
    System Error when using Eval Algorithm
    """


class DuplicateEvalNameError(EvalAlgorithmClientError):
    """
    Evaluation name already exists.
    """
