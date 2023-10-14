import itertools

import evaluate as hf_evaluate
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd

from amazon_fmeval import util
from amazon_fmeval.constants import MODEL_INPUT_COLUMN_NAME, MEAN
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.data_loaders.util import get_dataset
from amazon_fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalScore,
    EvalOutput,
    DATASET_CONFIGS,
    EVAL_DATASETS,
    EVAL_PROMPT_TEMPLATES,
)
from amazon_fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmConfig, EvalAlgorithmInterface
from amazon_fmeval.eval_algorithms.helper_models.semantic_preserving_perturbations import (
    ButterFinger,
    RandomUpperCase,
    WhitespaceAddRemove,
    ButterFingerConfig,
    RandomUpperCaseConfig,
    WhitespaceAddRemoveConfig,
)
from amazon_fmeval.eval_algorithms.util import (
    validate_dataset,
    save_dataset,
    aggregate_evaluation_scores,
    generate_output_dataset_path,
    generate_prompt_column_for_dataset,
)
from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.composers.composers import PromptComposer
from amazon_fmeval.model_runners.model_runner import ModelRunner

# All the perturbation types supported by this eval algo
BUTTER_FINGER = "butter_finger"
RANDOM_UPPER_CASE = "random_upper_case"
WHITESPACE_ADD_REMOVE = "whitespace_add_remove"

PERTURBATION_TYPE_TO_HELPER_CLASS = {
    BUTTER_FINGER: ButterFinger,
    RANDOM_UPPER_CASE: RandomUpperCase,
    WHITESPACE_ADD_REMOVE: WhitespaceAddRemove,
}

WER_SCORE = "word_error_rate"

PROMPT_COLUMN_NAME = "prompt"
PERTURBED_INPUT_COLUMN_NAME = "perturbed_input"
PERTURBED_INPUT_PROMPT_COLUMN_NAME = "perturbed_input_prompt"
PERTURBED_INPUT_MODEL_OUTPUT_COLUMN_NAME = "perturbed_input_model_output"


@dataclass(frozen=True)
class GeneralSemanticRobustnessConfig(EvalAlgorithmConfig):
    """
    Configuration for the general semantic robustness eval algorithm.

    :param perturbation_type: perturbation type for generating perturbed inputs
    :param num_perturbations: Number of perturbed inputs to be generated for robustness evaluation
    :param seed: Seed to be configured for generating perturbations
    :param butter_finger_perturbation_prob: The probability that a given character will be perturbed. Used for
        butter_finger perturbation_type
    :param random_uppercase_corrupt_proportion: Fraction of characters to be changed to uppercase. Used for
        random_upper_case perturbation_type
    :param whitespace_remove_prob: Given a whitespace, remove it with this much probability. Used for
        whitespace_add_remove perturbation_type
    :param whitespace_add_prob: Given a non-whitespace, add a whitespace before it with this probability. Used for
        whitespace_add_remove perturbation_type
    """

    perturbation_type: str = BUTTER_FINGER
    num_perturbations: int = 5
    seed: int = 5
    butter_finger_perturbation_prob: Optional[float] = 0.1
    random_uppercase_corrupt_proportion: Optional[float] = 0.1
    whitespace_remove_prob: Optional[float] = 0.1
    whitespace_add_prob: Optional[float] = 0.05

    def __post_init__(self):
        if self.perturbation_type not in PERTURBATION_TYPE_TO_HELPER_CLASS.keys():
            raise EvalAlgorithmClientError(
                f"Invalid perturbation type '{self.perturbation_type} requested, please "
                f"choose from acceptable values: {PERTURBATION_TYPE_TO_HELPER_CLASS.keys()}"
            )


class GeneralSemanticRobustness(EvalAlgorithmInterface):
    """
    Semantic Robustness Eval algorithm for General task LLMs

    This evaluation measures how much the model output changes as a result of semantic preserving
    perturbations. Given the input, e.g., "A quick brown fox jumps over the lazy dog", the
    evaluation creates a perturbation that preserves the semantic meaning of the input e.g.,
    whitespace perturbation that changes the input text to "A q uick bro wn fox ju mps overthe lazy
    dog". The evaluation then measures how much the model output changes when prompted with the
    original vs. perturbed input. The output difference is measured using Word Error Rate (WER).
    https://huggingface.co/spaces/evaluate-metric/wer
    """

    def __init__(self, eval_algorithm_config: GeneralSemanticRobustnessConfig):
        """Default constructor

        :param eval_algorithm_config: General Semantic Robustness eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.eval_name = EvalAlgorithm.GENERAL_SEMANTIC_ROBUSTNESS.value
        self._eval_algorithm_config = eval_algorithm_config

        if self._eval_algorithm_config.perturbation_type is BUTTER_FINGER:
            self._perturbation_config = ButterFingerConfig(self._eval_algorithm_config.butter_finger_perturbation_prob)
        elif self._eval_algorithm_config.perturbation_type is RANDOM_UPPER_CASE:
            self._perturbation_config = RandomUpperCaseConfig(
                self._eval_algorithm_config.random_uppercase_corrupt_proportion
            )
        else:
            self._perturbation_config = WhitespaceAddRemoveConfig(
                self._eval_algorithm_config.whitespace_remove_prob, self._eval_algorithm_config.whitespace_add_prob
            )

    def evaluate_sample(
        self, model_input: str, model: ModelRunner, prompt_template: str = "$feature"
    ) -> List[EvalScore]:  # type: ignore[override]
        """
        Semantic Robustness evaluate sample.

        :param model_input: text input for model
        :param model: An instance of ModelRunner which is the model under evaluation
        :param prompt_template: A template which can be used to compose prompt using model_input
        :return: list of EvalScore object
        """
        util.require(model_input, "Missing required input: model_input, for GeneralSemanticRobustness evaluate_sample")
        util.require(
            model, "Missing required input: model i.e. ModelRunner, for GeneralSemanticRobustness " "evaluate_sample"
        )

        prompt_composer = PromptComposer(prompt_template)
        original_prompt = prompt_composer.compose(model_input)
        original_model_output = model.predict(original_prompt)[0]

        # Check if predictor is deterministic
        if model.predict(original_prompt)[0] != original_model_output:
            raise EvalAlgorithmClientError("For evaluating semantic robustness, the model should be deterministic.")

        perturbation = PERTURBATION_TYPE_TO_HELPER_CLASS[self._eval_algorithm_config.perturbation_type](
            seed=self._eval_algorithm_config.seed
        )
        perturbed_inputs = perturbation.perturb(
            text=model_input,
            config=self._perturbation_config,
            num_perturbations=self._eval_algorithm_config.num_perturbations,
        )
        perturbed_input_prompts = [prompt_composer.compose(perturbed_input) for perturbed_input in perturbed_inputs]
        perturbed_input_outputs = [model.predict(prompt)[0] for prompt in perturbed_input_prompts]

        wer = hf_evaluate.load("wer")

        return [
            EvalScore(
                name=WER_SCORE,
                value=wer.compute(
                    predictions=perturbed_input_outputs,
                    references=list(
                        itertools.repeat(original_model_output, self._eval_algorithm_config.num_perturbations)
                    ),
                ),
            )
        ]

    def evaluate(  # type: ignore[override]
        self,
        model: ModelRunner,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        save: bool = False,
    ) -> List[EvalOutput]:
        """
        Semantic Robustness evaluate.

        :param model: An instance of ModelRunner which is the model under evaluation
        :param dataset_config: Configures the single dataset used for evaluation. If not provided,
            evaluation will use all of it's supported built-in datasets
        :param prompt_template: A template which can be used to generate prompts, optional for the built-in datasets.
        :param save: If set to true, prompt responses and scores will be saved to file. The output is written to
                     EvalAlgorithmInterface.EVAL_RESULTS_PATH
        :return: List of EvalOutput objects.
        """
        util.require(
            model, "Missing required input: model i.e. ModelRunner, for GeneralSemanticRobustness evaluate method"
        )
        is_custom_dataset_evaluation = False
        if dataset_config:
            is_custom_dataset_evaluation = True
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[self.eval_name]]

        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config)
            validate_dataset(dataset, [MODEL_INPUT_COLUMN_NAME])
            if is_custom_dataset_evaluation:
                # TODO when user provide built-in DataConfig, we should provide default prompt_template
                util.require(
                    prompt_template,
                    f"Missing required input: prompt_template for evaluating custom dataset : {dataset_config}",
                )
            else:
                prompt_template = EVAL_PROMPT_TEMPLATES[self.eval_name, dataset_config.dataset_name]
                util.assert_condition(
                    prompt_template is not None,
                    f"No Prompt Template configured for ({self.eval_name}, {dataset_config.dataset_name})",
                )

            assert prompt_template  # to satisfy mypy
            dataset = generate_prompt_column_for_dataset(
                prompt_template, dataset, MODEL_INPUT_COLUMN_NAME, PROMPT_COLUMN_NAME
            )

            def _generate_general_semantic_robustness_score(df: pd.DataFrame) -> pd.Series:  # pragma: no cover
                """
                Map function generating the scores for every input record in input dataset
                """
                return pd.Series(
                    data=[
                        self.evaluate_sample(row[MODEL_INPUT_COLUMN_NAME], model, prompt_template)[0].value
                        for index, row in df.iterrows()
                    ]
                )

            dataset = dataset.add_column(WER_SCORE, _generate_general_semantic_robustness_score)

            dataset_scores, category_scores = aggregate_evaluation_scores(dataset, [WER_SCORE], agg_method=MEAN)
            eval_outputs.append(
                EvalOutput(
                    eval_name=self.eval_name,
                    dataset_name=dataset_config.dataset_name,
                    prompt_template=prompt_template,
                    dataset_scores=dataset_scores,
                    category_scores=category_scores,
                    output_path=self._eval_results_path,
                )
            )
            if save:
                save_dataset(
                    dataset=dataset,
                    score_names=[WER_SCORE],
                    path=generate_output_dataset_path(
                        path_to_parent_dir=self._eval_results_path,
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                    ),
                )

        return eval_outputs
