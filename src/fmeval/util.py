import os
import re
import ray
import multiprocessing as mp
import importlib.metadata

from ray.actor import ActorHandle
from fmeval.constants import EVAL_RESULTS_PATH, DEFAULT_EVAL_RESULTS_PATH, PARALLELIZATION_FACTOR
from fmeval.exceptions import EvalAlgorithmInternalError, EvalAlgorithmClientError


def require(expression, msg: str):
    """
    Raise EvalAlgorithmClientError if expression is not True
    """
    if not expression:
        raise EvalAlgorithmClientError(msg)


def assert_condition(expression, msg: str):
    """
    Raise EvalAlgorithmInternalError if expression is not True
    """
    if not expression:
        raise EvalAlgorithmInternalError(msg)


def project_root(current_file: str) -> str:
    """
    :return: project root
    """
    curpath = os.path.abspath(os.path.dirname(current_file))

    def is_project_root(path: str) -> bool:
        return os.path.exists(os.path.join(path, ".root"))

    while not is_project_root(curpath):  # pragma: no cover
        parent = os.path.abspath(os.path.join(curpath, os.pardir))
        if parent == curpath:
            raise EvalAlgorithmInternalError("Got to the root and couldn't find a parent folder with .root")
        curpath = parent
    return curpath


def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def get_eval_results_path():
    """
    Util method to return results path for eval_algos. This method looks for EVAL_RESULTS_PATH environment variable,
    if present returns that else default path
    :returns: Local directory path of eval algo results
    """
    if os.environ.get(EVAL_RESULTS_PATH) is not None:
        os.makedirs(os.environ[EVAL_RESULTS_PATH], exist_ok=True)
        return os.environ[EVAL_RESULTS_PATH]
    else:
        os.makedirs(DEFAULT_EVAL_RESULTS_PATH, exist_ok=True)
        return DEFAULT_EVAL_RESULTS_PATH


def singleton(cls):
    """
    Decorator to make a class Singleton
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def get_num_actors():
    try:
        num_actors = (
            int(os.environ[PARALLELIZATION_FACTOR]) if PARALLELIZATION_FACTOR in os.environ else (mp.cpu_count() - 1)
        )
    except ValueError:
        num_actors = mp.cpu_count() - 1
    return num_actors


def create_shared_resource(resource: object, num_cpus: int = 1) -> ActorHandle:
    """Create a Ray actor out of `resource`.

    Typically, `resource` will be an object that consumes a significant amount of
    memory (ex: a BertscoreHelperModel instance) that you do not want to create
    on a per-transform (i.e. per-process) basis, but rather wish to have as a "global resource".

    Conceptually, the object that is returned from this function can be thought
    of as the input object, except it now exists in shared memory, as opposed
    to the address space of the process it was created in. Note that this
    function returns a Ray actor handle, which must be interacted with using the
    Ray remote API.

    :param resource: The object which we create a Ray actor from.
        This object's class must implement the `__reduce__` method
        with a return value of the form (ClassName, serialized_data),
        where serialized_data is a tuple containing arguments to __init__,
        in order to be compatible with this function.
    :param num_cpus: The num_cpus parameter to pass to ray.remote().
        This parameter represents the number of Ray logical CPUs
        (see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#physical-resources-and-logical-resources)
        that the created actor will require.
    :returns: The Ray actor handle corresponding to the created actor.
    """
    resource_cls, serialized_data = resource.__reduce__()  # type: ignore[misc]
    wrapped_resource_cls = ray.remote(num_cpus=num_cpus)(resource_cls)
    return wrapped_resource_cls.remote(*serialized_data)  # type: ignore


def cleanup_shared_resource(resource: ActorHandle) -> None:
    """Remove the resource from shared memory.

    Concretely, this function kills the Ray actor corresponding
    to `resource`, which in most cases will be an actor created
    via create_shared_resource.

    :param resource: A Ray actor handle to a shared resource
        (ex: a BertscoreHelperModel).
    :returns: None
    """
    ray.kill(resource)


def get_fmeval_package_version() -> str:
    """
    :returns: The current fmeval package version.
    """
    return importlib.metadata.version("fmeval")
