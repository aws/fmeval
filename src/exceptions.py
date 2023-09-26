"""
Error classes for exceptions
"""


# User facing exceptions
class UserError(Exception):
    """
    processing container failed due to an error that can be prevented or avoided by the customer
    """


class AlgorithmError(Exception):
    """
    processing container failed due to an unexpected or unknown failure that cannot be avoided by the customer
    and is caused by a bug in the processing container
    """

