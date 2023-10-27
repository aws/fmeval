import pytest

from amazon_fmeval.fmeval import get_eval_algorithm
from amazon_fmeval.eval_algorithms.factual_knowledge import FactualKnowledge
from amazon_fmeval.exceptions import EvalAlgorithmClientError


def test_get_eval_algorithm():
    assert get_eval_algorithm("factual_knowledge") == FactualKnowledge
    with pytest.raises(EvalAlgorithmClientError, match="Unknown eval algorithm accuracy"):
        get_eval_algorithm("accuracy")
