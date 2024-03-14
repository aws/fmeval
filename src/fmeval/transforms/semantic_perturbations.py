import random
import numpy as np
from typing import Dict, List, Any
from abc import abstractmethod

from fmeval.transforms.transform import Transform
from fmeval.transforms.util import validate_call
from fmeval.util import require


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class SemanticPerturbation(Transform):
    """The abstract base class for semantic perturbation transforms.

    Concrete subclasses of SemanticPerturbation should simply implement the
    `perturb` method and their own __init__ method. Subclasses need not implement
    the __call__ method, as it is already implemented in this class, but are
    free to do so if additional customization is required.
    """

    def __init__(
        self, input_keys: List[str], output_keys: List[str], num_perturbations: int, seed: int, *args, **kwargs
    ):
        """SemanticPerturbation initializer.

        :param input_keys: A single-element list containing the key corresponding to the text input to be perturbed.
        :param output_keys: The keys corresponding to perturbed text outputs generated by the `perturb` method.
        :param num_perturbations: The number of perturbed outputs to generate via the `perturb` method.
            Note that the number of output keys must match this parameter.
        :param seed: A random seed, used by pseudorandom number generators.
        :param *args: Variable length argument list.
        :param **kwargs: Arbitrary keyword arguments.
        """
        require(
            len(output_keys) == num_perturbations,
            f"len(output_keys) is {len(output_keys)} while num_perturbations is {num_perturbations}. They should match.",
        )
        require(len(input_keys) == 1, f"{self.__class__.__name__} takes a single input key.")
        super().__init__(input_keys, output_keys, num_perturbations, seed, args, kwargs)
        self.num_perturbations = num_perturbations
        set_seed(seed)

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with perturbed text outputs.

        :param record: The input record.
        :returns: The input record with perturbed text outputs added in.
        """
        input_key = self.input_keys[0]
        perturbed_texts = self.perturb(record[input_key])
        for key, text in zip(self.output_keys, perturbed_texts):
            record[key] = text
        return record

    @abstractmethod
    def perturb(
        self,
        text: str,
    ) -> List[str]:
        """Given an input text, generates one or more perturbed versions of it.

        Some perturbations can only generate a single perturbed version, e.g.
        converting all numbers to numerics (eight -> 8).

        :param text: The input text to be perturbed.
        :returns: A list of perturbed texts.
        """


class ButterFinger(SemanticPerturbation):
    """Given some text, add keyboard-induced typos in randomly selected words.

    Keyboard-induced typos are ones where a character is replaced by adjacent characters on the keyboard.

    Example:
        Original: A quick brown fox jumps over the lazy dog 10 times.
        Perturbed: W quick brmwn fox jumps over the lazy dig 10 times.

    Adapted from: https://github.com/GEM-benchmark/NL-Augmenter/blob/c591130760b453b3ad09516849dfc26e721eeb24/nlaugmenter/transformations/butter_fingers_perturbation/transformation.py
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

    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        num_perturbations: int = 5,
        seed: int = 5,
        perturbation_prob: float = 0.1,
    ):
        """ButterFinger initializer.

        :param input_keys: A single-element list containing the key corresponding to the text input to be perturbed.
        :param output_keys: The keys corresponding to perturbed text outputs generated by the `perturb` method.
        :param num_perturbations: The number of perturbed outputs to generate via the `perturb` method.
            Note that the number of output keys must match this parameter.
        :param seed: A random seed, used by pseudorandom number generators.
        :param perturbation_prob: The probability that a given character in the input text will be perturbed.
        """
        super().__init__(
            input_keys,
            output_keys,
            num_perturbations,
            seed,
            perturbation_prob=perturbation_prob,
        )
        self.perturbation_prob = perturbation_prob

    def perturb(self, text: str) -> List[str]:
        """Return a list where each element is a copy of the original text after undergoing perturbations.

        The perturbations mimic keyboard-induced typos and are applied (with a certain probability)
        to each character in `text`.

        :param text: The input text to be perturbed.
        :returns: A list of perturbed text outputs.
        """
        prob_of_typo = int(self.perturbation_prob * 100)
        perturbed_texts = []
        for _ in range(self.num_perturbations):
            butter_finger_text = []
            for letter in text:
                lowercase_letter = letter.lower()
                if lowercase_letter not in ButterFinger.QUERTY_KEY_APPROX.keys():
                    new_letter = lowercase_letter
                else:
                    if random.choice(range(0, 100)) <= prob_of_typo:
                        new_letter = random.choice(ButterFinger.QUERTY_KEY_APPROX[lowercase_letter])
                    else:
                        new_letter = lowercase_letter
                # go back to original case
                if lowercase_letter != letter:
                    new_letter = new_letter.upper()
                butter_finger_text.append(new_letter)
            perturbed_texts.append("".join(butter_finger_text))
        return perturbed_texts


class RandomUppercase(SemanticPerturbation):
    """Convert random characters in input text to uppercase characters.

    Example:
        Original: A quick brown fox jumps over the lazy dog 10 times.
        Perturbed: A qUick brOwn fox jumps over the lazY dog 10 timEs.

    Adapted from: https://github.com/GEM-benchmark/NL-Augmenter/blob/c591130760b453b3ad09516849dfc26e721eeb24/nlaugmenter/transformations/random_upper_transformation/transformation.py#L1
    """

    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        num_perturbations: int = 5,
        seed: int = 5,
        uppercase_fraction: float = 0.1,
    ):
        """RandomUpperCase initializer.

        :param input_keys: A single-element list containing the key corresponding to the text input to be perturbed.
        :param output_keys: The keys corresponding to perturbed text outputs generated by the `perturb` method.
        :param num_perturbations: The number of perturbed outputs to generate via the `perturb` method.
            Note that the number of output keys must match this parameter.
        :param seed: A random seed, used by pseudorandom number generators.
        :param uppercase_fraction: The fraction of characters to be changed to uppercase.
        """
        super().__init__(
            input_keys,
            output_keys,
            num_perturbations,
            seed,
            uppercase_fraction=uppercase_fraction,
        )
        self.uppercase_fraction = uppercase_fraction

    def perturb(self, text: str) -> List[str]:
        """Return a list where each element is a copy of the original text with a fraction of characters capitalized.

        :param text: The input text to be perturbed.
        :returns: A list of perturbed text outputs.
        """

        def random_uppercase_text():
            """Return a copy of `text` where a fraction of the characters are converted to uppercase.

            :returns: A copy of `text` where a fraction of the characters are converted to uppercase.
            """
            positions = np.random.choice(
                range(len(text)),
                int(len(text) * self.uppercase_fraction),
                False,
            )
            new_text = [letter if index not in positions else letter.upper() for index, letter in enumerate(text)]
            return "".join(new_text)

        return [random_uppercase_text() for _ in range(self.num_perturbations)]


class AddRemoveWhitespace(SemanticPerturbation):
    """Add and remove whitespaces within a piece of text at random.
    Example:
        Original: A quick brown fox jumps over the lazy dog 10 times.
        Perturbed: A q uick bro wn fox ju mps overthe lazy dog 10 times.

    Adapted from: https://github.com/GEM-benchmark/NL-Augmenter/blob/c591130760b453b3ad09516849dfc26e721eeb24/nlaugmenter/transformations/whitespace_perturbation/transformation.py
    """

    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        num_perturbations: int = 5,
        seed: int = 5,
        add_prob: float = 0.05,
        remove_prob: float = 0.1,
    ):
        """AddRemoveWhitespace initializer.

        :param input_keys: A single-element list containing the key corresponding to the text input to be perturbed.
        :param output_keys: The keys corresponding to perturbed text outputs generated by the `perturb` method.
        :param num_perturbations: The number of perturbed outputs to generate via the `perturb` method.
            Note that the number of output keys must match this parameter.
        :param seed: A random seed, used by pseudorandom number generators.
        :param add_prob: The probability of adding a whitespace character after a non-whitespace character.
        :param remove_prob: The probability of removing a whitespace character.
        """
        super().__init__(
            input_keys,
            output_keys,
            num_perturbations,
            seed,
            add_prob=add_prob,
            remove_prob=remove_prob,
        )
        self.add_prob = add_prob
        self.remove_prob = remove_prob

    def perturb(self, text: str) -> List[str]:
        """Return a list where each element is the original text with whitespaces potentially added or removed.

        :param text: The input text to be perturbed.
        :returns: A list of perturbed text outputs.
        """

        def update_char(char: str, p: float):
            """Return an updated character, with whitespace potentially added or removed.

            :param char: The input character.
            :param p: A number in the interval [0, 1) used to determine whether
                whitespace will be added/removed.
            :returns: An updated character, with whitespace potentially added or removed from the input character.
            """
            if char.isspace() and p < self.remove_prob:
                return ""
            if (not char.isspace()) and p < self.add_prob:
                return char + " "
            return char

        perturbed_texts = []
        for _ in range(self.num_perturbations):
            perturbed_text = []
            for ch in text:
                p = random.random()
                perturbed_text += [update_char(ch, p)]
            perturbed_texts.append("".join(perturbed_text))
        return perturbed_texts
