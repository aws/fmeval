import os
import sys
import logging

import ray
from pytest import fixture
from fmeval.util import project_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@fixture(scope="session")
def integration_tests_dir():
    return os.path.join(project_root(__file__), "test", "integration")


@fixture(scope="session", autouse=True)
def append_integration_dir(integration_tests_dir):
    sys.path.append(integration_tests_dir)


@fixture(scope="class", autouse=True)
def shutdown_ray():
    """
    Forcefully shut down Ray to ensure that resources
    used by the tests in a particular class get freed.
    """
    yield
    ray.shutdown()
