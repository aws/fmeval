import re
from typing import NamedTuple, Optional, Union, Dict, Type

import pytest

from fmeval.eval_algorithms import EvalAlgorithm
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmInterface, EvalAlgorithmConfig
from fmeval.eval_algorithms.prompt_stereotyping import PromptStereotyping
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.fmeval import get_eval_algorithm
from fmeval.eval_algorithms.factual_knowledge import FactualKnowledge, FactualKnowledgeConfig


class TestCaseGetEvalAlgo(NamedTuple):
    eval_name: str
    eval: EvalAlgorithmInterface = None
    eval_algorithm_config: Optional[Union[Dict, EvalAlgorithmConfig]] = None
    error: Type[Exception] = None
    error_message: str = None


@pytest.mark.parametrize(
    "test_case",
    [
        TestCaseGetEvalAlgo(eval_name=EvalAlgorithm.FACTUAL_KNOWLEDGE.value, eval=FactualKnowledge()),
        TestCaseGetEvalAlgo(
            eval_name=EvalAlgorithm.FACTUAL_KNOWLEDGE.value,
            eval=FactualKnowledge(FactualKnowledgeConfig(target_output_delimiter="<OR>")),
            eval_algorithm_config={"target_output_delimiter": "<OR>"},
        ),
        TestCaseGetEvalAlgo(
            eval_name=EvalAlgorithm.FACTUAL_KNOWLEDGE.value,
            eval=FactualKnowledge(FactualKnowledgeConfig(target_output_delimiter="<OR>")),
            eval_algorithm_config=FactualKnowledgeConfig(target_output_delimiter="<OR>"),
        ),
        TestCaseGetEvalAlgo(
            eval_name=EvalAlgorithm.FACTUAL_KNOWLEDGE.value,
            eval_algorithm_config={"invalid_parameter": "<OR>"},
            error=EvalAlgorithmClientError,
            error_message="Unable to create algorithm for eval_name factual_knowledge with config "
            "{'invalid_parameter': '<OR>'}: Error FactualKnowledgeConfig.__init__() got an unexpected "
            "keyword argument 'invalid_parameter'",
        ),
        TestCaseGetEvalAlgo(eval_name=EvalAlgorithm.PROMPT_STEREOTYPING.value, eval=PromptStereotyping()),
        TestCaseGetEvalAlgo(
            eval_name=EvalAlgorithm.PROMPT_STEREOTYPING.value,
            eval=PromptStereotyping(),
            eval_algorithm_config={"invalid_parameter": "<OR>"},
        ),
        TestCaseGetEvalAlgo(
            eval_name="invalid_algo",
            eval_algorithm_config={"invalid_parameter": "<OR>"},
            error=EvalAlgorithmClientError,
            error_message="Unknown eval algorithm invalid_algo",
        ),
    ],
)
def test_get_eval_algorithm(test_case: TestCaseGetEvalAlgo):
    if not test_case.error_message:
        assert type(
            get_eval_algorithm(eval_name=test_case.eval_name, eval_algorithm_config=test_case.eval_algorithm_config)
        ) == type(test_case.eval)
    else:
        with pytest.raises(test_case.error, match=re.escape(test_case.error_message)):
            get_eval_algorithm(eval_name=test_case.eval_name, eval_algorithm_config=test_case.eval_algorithm_config)
