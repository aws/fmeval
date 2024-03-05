import ray.data
from typing import List, Union
from collections import defaultdict

from fmeval.constants import TRANSFORM_PIPELINE_MAX_SIZE
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.transforms.transform import Transform
from fmeval.util import require

NestedTransform = Union[Transform, "TransformPipeline"]


class TransformPipeline:
    """A TransformPipeline represents a sequence of Transforms to be applied to a dataset.

    TransformPipelines can be created from a combination of Transforms or other TransformPipelines,
    thus enabling the creation of a pipeline with a nested, tree-like structure.

    Note: mutating the `transforms` list of a child pipeline (either by adding or removing
    elements from the list) is not recommended, as the changes will not propagate to the parent
    pipeline. The parent pipeline's list of transforms will continue to be whatever it was when
    the parent pipeline was initialized. If you find the need to mutate a child pipeline,
    consider creating a separate, new pipeline instead.

    Note: mutating the Transform objects that comprise a child pipeline's `transforms` list *will*
    affect the parent pipeline. However, Transform objects should essentially never be mutated
    after initialization. Doing so can lead to unexpected behavior, and is strongly advised against.
    """

    def __init__(self, nested_transforms: List[NestedTransform]):
        """TransformPipeline initializer.

        :param nested_transforms: A list of Transforms and/or TransformPipelines.
        """
        require(
            isinstance(nested_transforms, List),
            "TransformPipeline initializer accepts a list containing Transforms or TransformPipelines, "
            f"but received an object with type {type(nested_transforms)}.",
        )
        seen_keys = set()
        transform_to_duplicate_keys = defaultdict(list)
        self.transforms: List[Transform] = []
        for nested_transform in nested_transforms:
            if isinstance(nested_transform, Transform):
                for key in nested_transform.output_keys:
                    if key in seen_keys:
                        transform_to_duplicate_keys[nested_transform].append(key)
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
            len(transform_to_duplicate_keys.keys()) == 0,
            "TransformPipeline contains Transforms with the same output keys as other Transforms. "
            "Here are the problematic Transforms, paired with their offending keys: "
            f"{str(dict(transform_to_duplicate_keys))}",
        )
        require(
            len(self.transforms) <= TRANSFORM_PIPELINE_MAX_SIZE,
            f"TransformPipeline initialized with {len(self.transforms)} Transforms. "
            f"Currently, the max pipeline size is {TRANSFORM_PIPELINE_MAX_SIZE}. "
            "An overly-large pipeline is typically an indication that your Transforms "
            "are performing tasks that are too fine-grained. See how this negatively "
            "affects performance here: "
            "https://docs.ray.io/en/latest/ray-core/patterns/too-fine-grained-tasks.html",
        )

    def execute(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        """Apply the Transforms in self.transforms to the input dataset.

        :param dataset: A Ray Dataset.
        :returns: The resulting Ray Dataset after all Transforms have been applied.
        """
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
