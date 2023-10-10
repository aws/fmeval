from enum import Enum
from typing import Dict, Type

from eval_algorithms.eval_algorithm import EvalAlgorithmInterface
from eval_algorithms import EvalAlgorithm
from eval_algorithms.factual_knowledge import FactualKnowledge

EVAL_ALGORITHMS: Dict[str, Type["EvalAlgorithmInterface"]] = {EvalAlgorithm.FACTUAL_KNOWLEDGE.value: FactualKnowledge}


def get_eval_algorithm(eval_name: str) -> Type["EvalAlgorithmInterface"]:  # pragma: no cover
    """
    Get eval algorithm class with name

    :param eval_name: eval algorithm name
    :return: eval algorithm class
    """
    # TODO: Add unit tests for this util once we updated the concerned maps.
    if eval_name in EVAL_ALGORITHMS:
        return EVAL_ALGORITHMS[eval_name]
    else:
        raise KeyError(f"Unknown eval algorithm {eval_name}")


class ModelTask(Enum):
    """The different types of tasks that are supported by the evaluations.

    The model tasks are used to determine the evaluation metrics for the
    model.
    """

    NO_TASK = "no_task"
    CLASSIFICATION = "classification"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"


# These mappings are not to be consumed for any use cases and is for representational purposes.
# NO_TASK should have all keys from EvalAlgorithm
MODEL_TASK_EVALUATION_MAP = {
    ModelTask.NO_TASK: [
        EvalAlgorithm.PROMPT_STEREOTYPING,
        EvalAlgorithm.FACTUAL_KNOWLEDGE,
        EvalAlgorithm.TOXICITY,
        EvalAlgorithm.SEMANTIC_ROBUSTNESS,
    ],
    ModelTask.CLASSIFICATION: [
        EvalAlgorithm.SEMANTIC_ROBUSTNESS,
    ],
    ModelTask.QUESTION_ANSWERING: [
        EvalAlgorithm.TOXICITY,
        EvalAlgorithm.SEMANTIC_ROBUSTNESS,
    ],
    ModelTask.SUMMARIZATION: [
        EvalAlgorithm.TOXICITY,
        EvalAlgorithm.SEMANTIC_ROBUSTNESS,
    ],
}
