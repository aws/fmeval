from dataclasses import dataclass
from typing import Tuple

from fmeval.constants import BUTTER_FINGER, RANDOM_UPPER_CASE, WHITESPACE_ADD_REMOVE, DatasetColumns
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.common import GeneratePrompt, GetModelOutputs
from fmeval.transforms.semantic_perturbations import (
    SemanticPerturbation,
    ButterFinger,
    RandomUppercase,
    AddRemoveWhitespace,
)
from fmeval.transforms.util import create_output_key
from fmeval.util import require

SEMANTIC_PERTURBATIONS = {
    BUTTER_FINGER: ButterFinger,
    RANDOM_UPPER_CASE: RandomUppercase,
    WHITESPACE_ADD_REMOVE: AddRemoveWhitespace,
}


@dataclass(frozen=True)
class SemanticRobustnessConfig:
    """Configures the semantic robustness evaluation algorithms.

    :param perturbation_type: Perturbation type for generating perturbed inputs.
        Either BUTTER_FINGER, RANDOM_UPPER_CASE, or WHITESPACE_ADD_REMOVE.
    :param num_perturbations: Number of perturbed outputs to be generated for robustness evaluation.
    :param butter_finger_perturbation_prob: The probability that a given character will be perturbed.
        Used when perturbation_type is BUTTER_FINGER.
    :param random_uppercase_corrupt_proportion: Fraction of characters to be changed to uppercase.
        Used when perturbation_type is RANDOM_UPPER_CASE.
    :param whitespace_remove_prob: The probability of removing a whitespace character.
        Used when perturbation_type is WHITESPACE_ADD_REMOVE.
    :param whitespace_add_prob: The probability of adding a whitespace character after a non-whitespace character.
        Used when perturbation_type is WHITESPACE_ADD_REMOVE.
    """

    perturbation_type: str = BUTTER_FINGER
    num_perturbations: int = 5
    butter_finger_perturbation_prob: float = 0.1
    random_uppercase_corrupt_proportion: float = 0.1
    whitespace_add_prob: float = 0.05
    whitespace_remove_prob: float = 0.1

    def __post_init__(self):
        require(
            self.perturbation_type in SEMANTIC_PERTURBATIONS,
            f"Invalid perturbation type '{self.perturbation_type} requested, please "
            f"choose from acceptable values: {SEMANTIC_PERTURBATIONS.keys()}",
        )


def get_perturbation_transform(config: SemanticRobustnessConfig) -> SemanticPerturbation:
    if config.perturbation_type == BUTTER_FINGER:
        return ButterFinger(
            input_key=DatasetColumns.MODEL_INPUT.value.name,
            output_keys=[
                create_output_key(ButterFinger.__name__, DatasetColumns.MODEL_INPUT.value.name, i)
                for i in range(config.num_perturbations)
            ],
            num_perturbations=config.num_perturbations,
            perturbation_prob=config.butter_finger_perturbation_prob,
        )
    elif config.perturbation_type == RANDOM_UPPER_CASE:
        return RandomUppercase(
            input_key=DatasetColumns.MODEL_INPUT.value.name,
            output_keys=[
                create_output_key(RandomUppercase.__name__, DatasetColumns.MODEL_INPUT.value.name, i)
                for i in range(config.num_perturbations)
            ],
            num_perturbations=config.num_perturbations,
            uppercase_fraction=config.random_uppercase_corrupt_proportion,
        )
    else:
        return AddRemoveWhitespace(
            input_key=DatasetColumns.MODEL_INPUT.value.name,
            output_keys=[
                create_output_key(AddRemoveWhitespace.__name__, DatasetColumns.MODEL_INPUT.value.name, i)
                for i in range(config.num_perturbations)
            ],
            num_perturbations=config.num_perturbations,
            add_prob=config.whitespace_add_prob,
            remove_prob=config.whitespace_remove_prob,
        )


def get_model_responses_from_perturbed_inputs(
    perturbation: SemanticPerturbation,
    prompt_template: str,
    model: ModelRunner,
) -> Tuple[SemanticPerturbation, GeneratePrompt, GetModelOutputs]:
    # Generate prompts from perturbed inputs
    gen_perturbed_prompts = GeneratePrompt(
        input_keys=perturbation.output_keys,
        output_keys=[
            create_output_key(GeneratePrompt.__name__, perturbed_input_key)
            for perturbed_input_key in perturbation.output_keys
        ],
        prompt_template=prompt_template,
    )

    # Invoke model with prompts generated above
    get_perturbed_outputs = GetModelOutputs(
        input_to_output_keys={
            perturbed_prompt_key: [create_output_key(GetModelOutputs.__name__, perturbed_prompt_key)]
            for perturbed_prompt_key in gen_perturbed_prompts.output_keys
        },
        model_runner=model,
    )

    return perturbation, gen_perturbed_prompts, get_perturbed_outputs
