from typing import Dict, Type

from amazon_fmeval.eval_algorithms import EvalAlgorithm
from amazon_fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface
from amazon_fmeval.eval_algorithms.factual_knowledge import FactualKnowledge
from amazon_fmeval.eval_algorithms.general_semantic_robustness import GeneralSemanticRobustness
from amazon_fmeval.eval_algorithms.prompt_stereotyping import PromptStereotyping
from amazon_fmeval.eval_algorithms.qa_accuracy import QAAccuracy
from amazon_fmeval.eval_algorithms.summarization_accuracy import SummarizationAccuracy
from amazon_fmeval.eval_algorithms.classification_accuracy import ClassificationAccuracy
from amazon_fmeval.eval_algorithms.summarization_accuracy_semantic_robustness import (
    SummarizationAccuracySemanticRobustness,
)
from amazon_fmeval.eval_algorithms.toxicity import Toxicity

EVAL_ALGORITHMS: Dict[str, Type["EvalAlgorithmInterface"]] = {
    EvalAlgorithm.FACTUAL_KNOWLEDGE.value: FactualKnowledge,
    EvalAlgorithm.QA_ACCURACY.value: QAAccuracy,
    EvalAlgorithm.SUMMARIZATION_ACCURACY.value: SummarizationAccuracy,
    EvalAlgorithm.PROMPT_STEREOTYPING.value: PromptStereotyping,
    EvalAlgorithm.CLASSIFICATION_ACCURACY.value: ClassificationAccuracy,
    EvalAlgorithm.GENERAL_SEMANTIC_ROBUSTNESS.value: GeneralSemanticRobustness,
    EvalAlgorithm.SUMMARIZATION_ACCURACY_SEMANTIC_ROBUSTNESS.value: SummarizationAccuracySemanticRobustness,
    EvalAlgorithm.TOXICITY.value: Toxicity,
}
