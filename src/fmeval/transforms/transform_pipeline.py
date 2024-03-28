import ray.data
from typing import List, Union, Dict, Any
from collections import defaultdict

from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.transforms.batched_transform import BatchedTransform
from fmeval.transforms.transform import Transform
from fmeval.util import require, get_num_actors

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

    def execute(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        """Apply the Transforms in self.transforms to the input dataset.

        :param dataset: A Ray Dataset.
        :returns: The resulting Ray Dataset after all Transforms have been applied.
        """
        for transform in self.transforms:
            if isinstance(transform, BatchedTransform):
                dataset = dataset.map_batches(
                    transform.__class__,
                    batch_size=transform.batch_size if transform.batch_size != -1 else "default",
                    fn_constructor_args=transform.args,
                    fn_constructor_kwargs=transform.kwargs,
                    concurrency=(1, get_num_actors()),
                ).materialize()
            else:
                dataset = dataset.map(
                    transform.__class__,
                    fn_constructor_args=transform.args,
                    fn_constructor_kwargs=transform.kwargs,
                    concurrency=(1, get_num_actors()),
                ).materialize()
        return dataset

    def execute_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the Transforms in self.transforms to a single record.

        :param record: An input record.
        :returns: The record with augmentations from all the applied Transforms.
        """
        for transform in self.transforms:
            record = transform(record)
        return record
