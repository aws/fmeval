from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Optional, Dict

from eval_algorithms.eval_algorithm_registry import (
    EvalAlgorithmInterface,
    EvalOutput,
    AggregationOutput,
    ModelOutput,
    ModelInput,
    TargetOutput,
    EvalAlgorithmConfig,
    AggregationInput,
    CategoryScore,
)
from eval_algorithms.exceptions import EvalAlgorithmClientError


@dataclass(frozen=True)
class FactualKnowledgeModelInput(ModelInput):
    """
    Dataclass for ModelInputs, for Factual Knowledge eval algorithm

    :param prompt: Input prompt sent to LLM
    """

    prompt: str


@dataclass(frozen=True)
class FactualKnowledgeModelOutput(ModelOutput):
    """
    Dataclass for ModelOutputs, for Factual Knowledge eval algorithm

    :param prompt_response: LLM response for an input prompt
    """

    prompt_response: str


@dataclass(frozen=True)
class FactualKnowledgeConfig(EvalAlgorithmConfig):
    """
    Configuration for the factual knowledge eval algorithm

    :param target_output_delimiter: Target Output can have multiple answers. We expect customer to combine all the
        possible answers into a single string and use the delimiter to separate them. For instance,
        if the answers are ["UK", "England"] and the delimiter="<OR>", then the target_output should be "UK<OR>England".
    """

    target_output_delimiter: str


class FactualKnowledge(EvalAlgorithmInterface):
    """
    Factual Knowledge Eval algorithm
    """

    EVAL_NAME = "factual_knowledge"

    def __init__(self, eval_algorithm_config: FactualKnowledgeConfig):
        """Default constructor

        :param eval_algorithm_config: Factual knowledge eval algorithm config.
        """
        self.eval_algorithm_config = eval_algorithm_config

    def evaluate(
        self,
        model_output: FactualKnowledgeModelOutput,
        model_input: Optional[FactualKnowledgeModelInput],
        target_output: Optional[TargetOutput] = None,
    ) -> EvalOutput:
        """
        Factual knowledge evaluation.

        Given an input prompt e.g., "London is the capital of" and expected answers(target_output) like
        ["United Kingdom", "England"], if the model is able to arrive at the correct completion(model_output).
        Generating any of the expected answers is considered a correct completion.
        Since models might generate long outputs, this evaluation does not look for an exact match.
        It considers the completion to be correct if the answer is contained within the model output generated.

        :param model_output: An instance of ModelOutput which contains the responses from the model needed for this
                             evaluation
        :param model_input: An instance of ModelInput which contains the prompts on which the model needs to be
                            evaluated on
        :param target_output: The expected responses for the prompts in model_input
        :return: an instance of EvalOutput that contains the score computed for prompts and responses.
        """
        if target_output is None or target_output.expected_response is None:
            raise EvalAlgorithmClientError("Missing required input: target_output, for FactualKnowledge eval algorithm")
        if model_input is None or model_input.prompt is None:
            raise EvalAlgorithmClientError(
                "Missing required input: prompt in model_input, for FactualKnowledge eval algorithm"
            )
        if model_output.prompt_response is None:
            raise EvalAlgorithmClientError(
                "Missing required input: prompt_response in model_output, for FactualKnowledge eval algorithm"
            )

        possible_targets = target_output.expected_response.split(self.eval_algorithm_config.target_output_delimiter)
        prompt_response_lower_case = model_output.prompt_response.lower()
        return EvalOutput(
            eval_type=self.EVAL_NAME,
            eval_score=int(any([t.lower() in prompt_response_lower_case for t in possible_targets])),
        )

    def aggregate(self, aggregation_input: AggregationInput) -> AggregationOutput:
        """
        Factual knowledge eval algo aggregation method.

        :param aggregation_input: An instance of AggregationInput containing list of EvalOutputs and categories
        :returns: an instance of AggregationOutput that contains the aggregation result
        """
        AggregationInput.validate_aggregation_input(aggregation_input)
        dataset_score = mean([eval_output.eval_score for eval_output in aggregation_input.eval_outputs])
        category_scores = None
        if aggregation_input.categories:
            category_scores_mapping: Dict = defaultdict(lambda: {"sum_score": 0, "count": 0})
            for eval_output, category in zip(aggregation_input.eval_outputs, aggregation_input.categories):
                category_scores_mapping[category]["sum_score"] += eval_output.eval_score
                category_scores_mapping[category]["count"] += 1

            category_scores = [
                CategoryScore(category, category_score_data["sum_score"] / category_score_data["count"])
                for category, category_score_data in category_scores_mapping.items()
            ]

        return AggregationOutput(eval_type=self.EVAL_NAME, dataset_score=dataset_score, category_scores=category_scores)
