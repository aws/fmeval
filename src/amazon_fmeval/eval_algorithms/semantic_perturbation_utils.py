"""
This module contains several semantic perturbations from the NL-Augmenter package. The
goals of having this module are twofolds:

1. NL-Augmenter has old dependencies which makes it difficult to install it in our env. We anyways
do not need all of NL-Augmenter, so we copy over the perturbations we need over here.
2. We might add more perturbations from other packages like `nlaug`, or even have our own custom
ones, in the future, so we want to have a uniform API for these perturbations.
"""
from abc import ABC, abstractmethod
import random
import itertools
from typing import Dict, List, Union
from dataclasses import dataclass
import functools

import numpy as np


@dataclass(frozen=True)
class ButterFingerConfig:
    """
    Config for the Butter Finger perturbation.
    Defaults set to match those in NL-Augmenter.

    :param perturbation_prob: The probability that a given character will be perturbed.
    """

    perturbation_prob: float = 0.1


@dataclass(frozen=True)
class RandomUpperCaseConfig:
    """
    Config for the RandomUpperCase perturbation.
    Defaults set to match those in NL-Augmenter.

    :param corrupt_proportion: Fraction of characters to be changed to uppercase.
    """

    corrupt_proportion: float = 0.1


@dataclass(frozen=True)
class WhitespaceAddRemoveConfig:
    """
    Config for WhitespaceAddRemove perturbation.
    Defaults set to match those in NL-Augmenter.

    :param remove_prob: Given a whitespace, remove it with this much probability.
    :param add_prob: Given a non-whitespace, add a whitespace before it with this probability.
    """

    remove_prob: float = 0.1
    add_prob: float = 0.05


class SemanticPerturbationUtil(ABC):
    """
    The interface that each perturbation should implement.
    """

    def __init__(self, seed: int = 5):
        self.set_seed(seed)

    @abstractmethod
    def perturb(
        self,
        text: str,
        config: Union[ButterFingerConfig, RandomUpperCaseConfig, WhitespaceAddRemoveConfig],
        num_perturbations: int = 5,
    ) -> List[str]:
        """
        Given an input text, generates one or more perturbed versions of it. Some perturbations can
        only generate a single perturbed version, e.g., converting all numbers to numerics (eight -> 8).

        :param text: The input text that needs to be perturbed.
        :param config: The configuration containing parameters for the perturbation.
        :param num_perturbations: Number of perturbed versions to generate. Some perturbations can
        only generate a single perturbed versions and will ignore this parameter.
        :returns: A list of perturbed texts.
        """

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)


class ButterFinger(SemanticPerturbationUtil):
    """
    Given a text, add keyboard induced typos in randomly selected words.
    Keyboard induced typos are ones where a character is replaced by adjacent characters on the keyboard.

    Example:
        Original: A quick brown fox jumps over the lazy dog 10 times.
        Perturbed: W quick brmwn fox jumps over the lazy dig 10 times.

    Adopted from: https://github.com/GEM-benchmark/NL-Augmenter/blob/c591130760b453b3ad09516849dfc26e721eeb24/nlaugmenter/transformations/butter_fingers_perturbation/transformation.py
    """

    # Setting default values from NL-Augmenter
    QUERTY_KEY_APPROX: Dict[str, str] = dict()
    QUERTY_KEY_APPROX["q"] = "qwasedzx"
    QUERTY_KEY_APPROX["w"] = "wqesadrfcx"
    QUERTY_KEY_APPROX["e"] = "ewrsfdqazxcvgt"
    QUERTY_KEY_APPROX["r"] = "retdgfwsxcvgt"
    QUERTY_KEY_APPROX["t"] = "tryfhgedcvbnju"
    QUERTY_KEY_APPROX["y"] = "ytugjhrfvbnji"
    QUERTY_KEY_APPROX["u"] = "uyihkjtgbnmlo"
    QUERTY_KEY_APPROX["i"] = "iuojlkyhnmlp"
    QUERTY_KEY_APPROX["o"] = "oipklujm"
    QUERTY_KEY_APPROX["p"] = "plo['ik"

    QUERTY_KEY_APPROX["a"] = "aqszwxwdce"
    QUERTY_KEY_APPROX["s"] = "swxadrfv"
    QUERTY_KEY_APPROX["d"] = "decsfaqgbv"
    QUERTY_KEY_APPROX["f"] = "fdgrvwsxyhn"
    QUERTY_KEY_APPROX["g"] = "gtbfhedcyjn"
    QUERTY_KEY_APPROX["h"] = "hyngjfrvkim"
    QUERTY_KEY_APPROX["j"] = "jhknugtblom"
    QUERTY_KEY_APPROX["k"] = "kjlinyhn"
    QUERTY_KEY_APPROX["l"] = "lokmpujn"

    QUERTY_KEY_APPROX["z"] = "zaxsvde"
    QUERTY_KEY_APPROX["x"] = "xzcsdbvfrewq"
    QUERTY_KEY_APPROX["c"] = "cxvdfzswergb"
    QUERTY_KEY_APPROX["v"] = "vcfbgxdertyn"
    QUERTY_KEY_APPROX["b"] = "bvnghcftyun"
    QUERTY_KEY_APPROX["n"] = "nbmhjvgtuik"
    QUERTY_KEY_APPROX["m"] = "mnkjloik"
    QUERTY_KEY_APPROX[" "] = " "

    def perturb(
        self, text: str, config: ButterFingerConfig, num_perturbations: int = 5  # type: ignore[override]
    ) -> List[str]:
        prob_of_typo = int(config.perturbation_prob * 100)
        perturbed_texts = []
        for _ in itertools.repeat(None, num_perturbations):
            butter_text = []
            for letter in text:
                lcletter = letter.lower()
                if lcletter not in self.QUERTY_KEY_APPROX.keys():
                    new_letter = lcletter
                else:
                    if random.choice(range(0, 100)) <= prob_of_typo:
                        new_letter = random.choice(self.QUERTY_KEY_APPROX[lcletter])
                    else:
                        new_letter = lcletter
                # go back to original case
                if not lcletter == letter:
                    new_letter = new_letter.upper()
                butter_text.append(new_letter)
            perturbed_texts.append("".join(butter_text))
        return perturbed_texts


class RandomUpperCase(SemanticPerturbationUtil):
    """
    Convert random characters in the text to uppercase.
    Example:
        Original: A quick brown fox jumps over the lazy dog 10 times.
        Perturbed: A qUick brOwn fox jumps over the lazY dog 10 timEs.

    Adopted from: https://github.com/GEM-benchmark/NL-Augmenter/blob/c591130760b453b3ad09516849dfc26e721eeb24/nlaugmenter/transformations/random_upper_transformation/transformation.py#L1
    """

    def perturb(
        self, text: str, config: RandomUpperCaseConfig, num_perturbations: int = 5  # type: ignore[override]
    ) -> List[str]:
        return list(map(functools.partial(self.random_upper, config=config), itertools.repeat(text, num_perturbations)))

    @staticmethod
    def random_upper(text: str, config: RandomUpperCaseConfig):
        positions = np.random.choice(
            range(len(text)),
            int(len(text) * config.corrupt_proportion),
            False,
        )

        new_sentence = [letter if index not in positions else letter.upper() for index, letter in enumerate(text)]
        return "".join(new_sentence)


class WhitespaceAddRemove(SemanticPerturbationUtil):
    """
    Add and remove whitespaces at random.
    Example:
        Original: A quick brown fox jumps over the lazy dog 10 times.
        Perturbed: A q uick bro wn fox ju mps overthe lazy dog 10 times.

    Adopted from: https://github.com/GEM-benchmark/NL-Augmenter/blob/c591130760b453b3ad09516849dfc26e721eeb24/nlaugmenter/transformations/whitespace_perturbation/transformation.py
    """

    def perturb(
        self, text: str, config: WhitespaceAddRemoveConfig, num_perturbations: int = 5  # type: ignore[override]
    ) -> List[str]:
        perturbed_texts = []
        for _ in range(num_perturbations):
            perturbed_text = []
            for char in text:
                random_num = random.random()
                perturbed_text += WhitespaceAddRemove.whitespace(char, random_num, config.remove_prob, config.add_prob)
            perturbed_texts.append("".join(perturbed_text))
        return perturbed_texts

    @staticmethod
    def whitespace(char, random_num, remove_prob, add_prob):
        if char.isspace() and random_num < remove_prob:
            return []
        perturbed_char = [char]
        if (not char.isspace()) and random_num < add_prob:
            perturbed_char.append(" ")

        return perturbed_char
