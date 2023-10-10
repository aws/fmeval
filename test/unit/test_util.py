import os

import pytest

from exceptions import EvalAlgorithmClientError
from util import require, project_root


def test_require():
    message = "Required resource is missing"
    require(True, message)


def test_require_fail():
    message = "Required resource is missing"
    with pytest.raises(EvalAlgorithmClientError, match=message):
        require(False, message)


def test_project_root():
    """
    GIVEN __name__
    WHEN util.project_root is called
    THEN the project root directory is returned
    """
    assert os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_util.py") == os.path.abspath(
        os.path.join(project_root(__name__), "test", "unit", "test_util.py")
    )
