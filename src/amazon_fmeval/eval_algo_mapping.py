from typing import Dict, Type

from amazon_fmeval.eval_algorithms import EvalAlgorithm
from amazon_fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface
from amazon_fmeval.eval_algorithms.factual_knowledge import FactualKnowledge
from amazon_fmeval.eval_algorithms.prompt_stereotyping import PromptStereotyping
from amazon_fmeval.eval_algorithms.qa_accuracy import QAAccuracy
from amazon_fmeval.eval_algorithms.summarization_accuracy import SummarizationAccuracy
from amazon_fmeval.eval_algorithms.classification_accuracy import ClassificationAccuracy

EVAL_ALGORITHMS: Dict[str, Type["EvalAlgorithmInterface"]] = {
    EvalAlgorithm.FACTUAL_KNOWLEDGE.value: FactualKnowledge,
    EvalAlgorithm.QA_ACCURACY.value: QAAccuracy,
    EvalAlgorithm.SUMMARIZATION_ACCURACY.value: SummarizationAccuracy,
    EvalAlgorithm.PROMPT_STEREOTYPING.value: PromptStereotyping,
    EvalAlgorithm.CLASSIFICATION_ACCURACY.value: ClassificationAccuracy,
}
