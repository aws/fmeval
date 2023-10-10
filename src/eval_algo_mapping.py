from typing import Dict, Type

from eval_algorithms import EvalAlgorithm
from eval_algorithms.eval_algorithm import EvalAlgorithmInterface
from eval_algorithms.factual_knowledge import FactualKnowledge
from eval_algorithms.prompt_stereotyping import PromptStereotyping
from eval_algorithms.qa_accuracy import QAAccuracy
from eval_algorithms.summarization_accuracy import SummarizationAccuracy
from exceptions import EvalAlgorithmClientError

EVAL_ALGORITHMS: Dict[str, Type["EvalAlgorithmInterface"]] = {
    EvalAlgorithm.FACTUAL_KNOWLEDGE.value: FactualKnowledge,
    EvalAlgorithm.QA_ACCURACY.value: QAAccuracy,
    EvalAlgorithm.SUMMARIZATION_ACCURACY.value: SummarizationAccuracy,
    EvalAlgorithm.PROMPT_STEREOTYPING.value: PromptStereotyping,
}


def get_eval_algorithm(eval_name: str) -> Type["EvalAlgorithmInterface"]:
    """
    Get eval algorithm class with name

    :param eval_name: eval algorithm name
    :return: eval algorithm class
    """
    if eval_name in EVAL_ALGORITHMS:
        return EVAL_ALGORITHMS[eval_name]
    else:
        raise EvalAlgorithmClientError(f"Unknown eval algorithm {eval_name}")
