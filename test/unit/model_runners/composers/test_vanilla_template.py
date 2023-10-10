from typing import NamedTuple, List

import pytest

from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.composers.template import VanillaTemplate


class TestVanillaTemplate:
    class TestCaseGetIdentifiers(NamedTuple):
        template: str
        identifiers: List[str]

    @pytest.mark.parametrize(
        "test_case",
        [
            # Empty (no identifier)
            TestCaseGetIdentifiers("", []),
            # No identifier
            TestCaseGetIdentifiers("No identifier", []),
            # Two identifiers
            TestCaseGetIdentifiers("There are $two $identifiers", ["two", "identifiers"]),
            # Escape dollar sign
            TestCaseGetIdentifiers("A $$$identifier", ["identifier"]),
            # Multiline
            TestCaseGetIdentifiers("$line1\n$line2\n$line3", ["line1", "line2", "line3"]),
        ],
    )
    def test_valid_identifiers(self, test_case):
        vanilla_template = VanillaTemplate(test_case.template)
        assert test_case.identifiers == vanilla_template.get_unique_identifiers()

    def test_invalid_characters(self):
        unsupported_characters = [
            # digit, space, punctuation, non-ASCII
            "1 . ë",
        ]

        for unsupported_character in unsupported_characters:
            template = "$" + unsupported_character + ""
            vanilla_template = VanillaTemplate(template)
            identifiers = vanilla_template.get_unique_identifiers()
            assert [] == identifiers

    def test_reappeared_placeholder(self):
        template = "$good $good study"
        error = EvalAlgorithmClientError
        message = "Identifier 'good' reappears in template '\$good \$good study'."

        with pytest.raises(error, match=message):
            vanilla_template = VanillaTemplate(template)
            identifiers = vanilla_template.get_unique_identifiers()
            assert message == identifiers

    def test_valid_template(self):
        template = "$identifier"
        expected = "1"

        vanilla_template = VanillaTemplate(template)
        result = vanilla_template.substitute(identifier=1)
        assert expected == result
        assert str(vanilla_template) == "VanillaTemplate(template=$identifier)"
