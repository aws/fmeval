from enum import Enum


class JmespathQueryType(Enum):
    """Used for error logging in JsonParser methods.

    While not strictly necessary, they give error messages
    additional context regarding the purpose/intention behind
    a given JMESPath query.
    """

    FEATURES = "features"
    TARGET = "target"
    INFERENCE = "inference"
    CATEGORY = "category"
