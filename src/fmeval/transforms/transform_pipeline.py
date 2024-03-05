import ray.data
from typing import List, Union

from fmeval.constants import TRANSFORM_PIPELINE_MAX_SIZE
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.transforms.transform import Transform
from fmeval.util import require

NestedTransform = Union[Transform, "TransformPipeline"]


class TransformPipeline:
    def __init__(self, nested_transforms: List[NestedTransform]):
        require(
            isinstance(nested_transforms, List),
            "TransformPipeline initializer accepts a list containing Transforms or TransformPipelines.",
        )
        seen_keys = set()
        duplicate_keys = []
        self.transforms: List[Transform] = []
        for nested_transform in nested_transforms:
            if isinstance(nested_transform, Transform):
                for key in nested_transform.output_keys:
                    if key in seen_keys:
                        duplicate_keys.append(key)
                    else:
                        seen_keys.add(key)
                self.transforms.append(nested_transform)
            elif isinstance(nested_transform, TransformPipeline):
                self.transforms += nested_transform.transforms
            else:
                raise EvalAlgorithmClientError(
                    f"nested_transform has type {type(nested_transform)}, "
                    "but either Transform or TransformPipeline is expected."
                )
        require(
            len(duplicate_keys) == 0,
            "TransformPipeline contains Transforms with the same output keys as other Transforms. "
            f"Duplicate keys: {duplicate_keys}.",
        )
        require(
            len(self.transforms) <= TRANSFORM_PIPELINE_MAX_SIZE,
            f"TransformPipeline initialized with {len(self.transforms)} Transforms. "
            f"Currently, the max pipeline size is {TRANSFORM_PIPELINE_MAX_SIZE}.",
        )

    def execute(self, dataset: ray.data.Dataset):
        for transform in self.transforms:
            dataset = dataset.map(
                transform.__class__,
                fn_constructor_args=transform.args,
                fn_constructor_kwargs=transform.kwargs,
                # num_cpus configures how many logical CPUs
                # (see https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
                # each map worker requires.
                # To prevent the case where an actor cannot be created due to
                # insufficient logical CPUs, we set num_cpus to a small fractional value.
                # The default value for num_cpus is 1 and the default number of logical CPUs
                # is the number of cores on the machine, meaning that if we don't
                # override num_cpus, any pipeline with more Transforms than cores will
                # fail to execute.
                num_cpus=(1 / TRANSFORM_PIPELINE_MAX_SIZE),
                # Set concurrency to 1 for now. Even with a concurrency parameter of 1,
                # the code runs significantly faster (by roughly 4x) compared to fmeval v1.
                concurrency=1,
            )
        return dataset
