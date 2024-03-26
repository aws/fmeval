import tempfile

import pytest
import os
from unittest.mock import patch, Mock

from fmeval.constants import DEFAULT_EVAL_RESULTS_PATH
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.util import (
    require,
    project_root,
    singleton,
    create_shared_resource,
    get_eval_results_path,
    cleanup_shared_resource,
)


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


def test_eval_get_results_path():
    """
    GIVEN the EVAL_RESULTS_PATH env variable is set (or not set).
    WHEN get_eval_results_path is called.
    THEN the correct path is returned and the directory exists.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = os.path.join(tmpdir, "custom", "path")
        os.environ["EVAL_RESULTS_PATH"] = results_path
        assert get_eval_results_path() == results_path
        assert os.path.exists(os.path.abspath(results_path))
        os.environ.pop("EVAL_RESULTS_PATH")

    assert get_eval_results_path() == DEFAULT_EVAL_RESULTS_PATH
    assert os.path.exists(os.path.abspath(DEFAULT_EVAL_RESULTS_PATH))


@singleton
class TestSingletonClass:
    def __init__(self):
        pass


def test_singleton_instance():
    singleton1 = TestSingletonClass()
    singleton2 = TestSingletonClass()

    assert singleton1 is singleton2


def test_create_shared_resource():
    """
    GIVEN an object.
    WHEN create_shared_resource is called on this object.
    THEN the relevant Ray functions are called with the correct arguments.

    Note: this unit test is included primarily for 100% unit test
    coverage purposes. It is critical that this function be
    tested without mocking anything, to ensure that the function
    works with Ray as intended.
    """

    class Dummy:
        def __init__(self, name: str, age: int):
            self.name = name
            self.age = age

        def __reduce__(self):
            return Dummy, (self.name, self.age)

    with patch("fmeval.util.ray.remote") as mock_ray_remote:
        mock_actor_class = Mock()
        mock_actor_class.remote = Mock()

        mock_wrapped_resource_class = Mock()
        mock_wrapped_resource_class.remote = Mock()

        mock_actor_class.return_value = mock_wrapped_resource_class
        mock_ray_remote.return_value = mock_actor_class

        num_cpus = 3
        create_shared_resource(Dummy("C", 2), num_cpus=num_cpus)

        mock_ray_remote.assert_called_once_with(num_cpus=num_cpus)
        mock_actor_class.assert_called_once_with(Dummy)
        mock_wrapped_resource_class.remote.assert_called_once_with("C", 2)


@patch("fmeval.util.ray.kill")
def test_cleanup_shared_resource(mock_ray_kill):
    """
    GIVEN a shared resource.
    WHEN cleanup_shared_resource is called.
    THEN ray.kill is called on this resource.
    """
    resource = Mock()
    cleanup_shared_resource(resource)
    mock_ray_kill.assert_called_once_with(resource)
