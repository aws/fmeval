import re
from abc import ABC, abstractmethod
from collections import UserDict
from typing import List, Optional, Type, Union
from dataclasses import dataclass


EVAL_ALGORITHMS = {

}

EVAL_DATASETS = {

}

DATASET_CONFIGS = {

}

def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

def get_eval_algorithm(self, eval_type: str) -> Type["EvalAlgorithmInterface"]:
    """
    Get eval algorithm class with name

    :param eval_type: eval algorithm name
    :return: eval algorithm class
    """
    if eval_type in self:
        return self[eval_type]
    else:
        raise KeyError(f"Unknown eval algorithm {eval_type}")
