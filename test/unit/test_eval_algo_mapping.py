import pytest

from eval_algo_mapping import get_eval_algorithm
from eval_algorithms.factual_knowledge import FactualKnowledge
from exceptions import EvalAlgorithmClientError


def test_get_eval_algorithm():
    assert get_eval_algorithm("factual_knowledge") == FactualKnowledge
    with pytest.raises(EvalAlgorithmClientError, match="Unknown eval algorithm accuracy"):
        get_eval_algorithm("accuracy")
