from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Optional, Dict, List
from ..exceptions import UserError
from eval_algorithm import EvalOutput, EvalAlgorithmConfig, EvalAlgorithmInterface

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

    def evaluate_sample(
        self,
        prompt: str,
        prompt_response: Optional[str],
        expected_response: Optional[str],
    ) -> List[int]:
        """
        Factual knowledge evaluation for one sample.

        Given an input prompt e.g., "London is the capital of" and expected answers(target_output) like
        ["United Kingdom", "England"], if the model is able to arrive at the correct completion(model_output).
        Generating any of the expected answers is considered a correct completion.
        Since models might generate long outputs, this evaluation does not look for an exact match.
        It considers the completion to be correct if the answer is contained within the model output generated.
        """
        if expected_response is None:
            raise UserError("Missing required input: expected_response, for FactualKnowledge algorithm")
        if prompt is None:
            raise UserError(
                "Missing required input: prompt in model input, for FactualKnowledge algorithm"
            )
        if prompt_response is None:
            raise UserError(
                "Missing required input: prompt_response in model output, for FactualKnowledge algorithm"
            )

        possible_targets = expected_response.split(self.eval_algorithm_config.target_output_delimiter)
        prompt_response_lower_case = prompt_response.lower()
        eval_score = int(any([t.lower() in prompt_response_lower_case for t in possible_targets]))
        return eval_score

    def evaluate(self, model: Optional[ModelRunner], custom_dataset_config: Optional[DataConfig] = None,
                 prompt_template: str = None, save: bool = False) -> EvalOutput:
        """
        Factual knowledge eval algo for one dataset.
        """
        if custom_dataset_config:
            dataset_config = custom_dataset_config
            dataset_name = custom_dataset_config.dataset_name
        else:
            dataset_name = EVAL_DATASETS[self.eval_name]
            dataset_config = DATASET_CONFIGS[dataset_name]

        dataset = DataLoader.get_dataset(dataset_config)
        if 'prompt_response' not in dataset.colnames():
            dataset['prompt_response'] = model.predict(dataset['prompt'])
        sample_scores = dataset.map(lambda x: self.evaluate_sample(x['prompt'], x['prompt_response'], x['expected_output']))
        dataset_score = sample_scores.mean()

        category_scores_mapping: Dict = defaultdict(lambda: {"sum_score": 0, "count": 0})
        for eval_output, category in zip(sample_scores, dataset.categories):
            category_scores_mapping[category]["sum_score"] += eval_output.eval_score
            category_scores_mapping[category]["count"] += 1

        category_scores = [
            CategoryScore(category, category_score_data["sum_score"] / category_score_data["count"])
            for category, category_score_data in category_scores_mapping.items()
        ]

        if save:
            ## TODO: Write model input, model output, sample scores to local file
            pass

        return EvalOutput(
            eval_name=self.EVAL_NAME,
            dataset_score=dataset_score,
            category_scores=category_scores
        )