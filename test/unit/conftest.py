import os

from _pytest.fixtures import fixture

from fmeval.util import project_root


@fixture(scope="session")
def unit_tests_dir():
    return os.path.join(project_root(__file__), "test", "unit")
