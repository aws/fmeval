import logging
from string import Template
from typing import List

import fmeval.util as util

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
