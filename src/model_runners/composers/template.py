import logging
from string import Template
from typing import List
import re

from orchestrator.utils import util

logger = logging.getLogger(__name__)


class VanillaTemplate(Template):
    """Extend the standard string.Template class with an utility method."""

    def get_unique_identifiers(self) -> List[str]:
        """Returns a list of the unique identifiers in the template.

        The identifiers are in the order they appear, ignoring any invalid identifiers.
        The method originates from Python 3.11 Template.get_identifiers (see [1] and [2]),
        but with additional checks to disallow reappearing identifiers in the template.

        [1] https://docs.python.org/3/library/string.html#string.Template.get_identifiers
        [2] https://github.com/python/cpython/blob/3.11/Lib/string.py#L157

        :return: The list of unique identifiers.
        """
        ids = []
        for mo in self.pattern.finditer(self.template):
            named = mo.group("named")
            if named is not None:
                util.require(
                    named not in ids,
                    f"Identifier '{named}' reappears in template '{self.template}'.",
                )
                ids.append(named)
        return ids

    def __str__(self):
        # Return a meaningful string representation of the object
        return f"VanillaTemplate(template={self.template})"


class PromptTemplate(Template):
    """Custom template that allow using feature name as identifier.

    For a feature named "A" in the headers configuration, put a placeholder "${A}" in the template string
    (the double-quotes are part of the placeholder) and then it will be replaced by the feature value.
    Each placeholder in the template must appears exactly once.
    """

    # Allow for additional valid characters to be in braced identifier
    delimiter = re.escape("${")
    braceidpattern = r"[\w ._]+"
    # Override the pattern to recognize an identifier A only when it looks like ${A}.
    # The other groups captures nothing but have to keep them to satisfy string.Template code.
    # Note that the pattern is a f-string which uses double-brace to represent a literal brace.
    pattern = rf"""
    {delimiter}(?:
      (?P<braced>{braceidpattern})}} |
      (?P<invalid>) |
      (?P<escaped>) |
      (?P<named>)
    )
    """  # type: ignore # The string.Template class will compile the pattern

    def get_unique_identifiers(self) -> List[str]:
        """Returns a list of the unique identifiers in the template using the braceidpattern.

        The identifiers are in the order they appear, ignoring any invalid identifiers.
        The method originates from Python 3.11 Template.get_identifiers (see [1] and [2]),
        but with additional checks to disallow reappearing identifiers in the template.

        [1] https://docs.python.org/3/library/string.html#string.Template.get_identifiers
        [2] https://github.com/python/cpython/blob/3.11/Lib/string.py#L157

        :return: The list of unique identifiers.
        """
        ids = []
        for match_object in self.pattern.finditer(self.template):  # type: ignore # The string.Template class compiles the pattern
            id = match_object.group("braced")
            if id is not None:
                util.require(
                    id not in ids,
                    f"Identifier '{id}' reappears in prompt template '{self.template}'.",
                )
                ids.append(id)
        return ids
