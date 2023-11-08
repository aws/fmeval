import contextlib
from timeit import default_timer as timer
import logging


@contextlib.contextmanager
def timed_block(block_name: str, logger: logging.Logger):
    """
    Measure and log execution time for the code block in the context
    :param block_name: a string describing the code block
    :param logger: used to log the execution time
    """
    start = timer()
    try:
        yield
    finally:
        end = timer()
        logger.info(f"{block_name} took {(end - start):.2f} seconds.")
        logger.info("===================================================")
