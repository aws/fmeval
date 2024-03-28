import numpy as np

from typing import Dict
from abc import abstractmethod
from fmeval.transforms.transform import Transform


class BatchedTransform(Transform):
    """A BatchedTransform is a Transform that takes in a batch of records instead of a single record.

    Certain transforms will have a significant performance boost when processing records in batches
    (the performance boost depends on the logic internal to the transform's __call__ method).

    This abstract base class should be inherited by such transforms.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Return a batch of records containing data that gets computed in this method.

        :param batch: The batch to be transformed.
        :returns: A batch of records containing data that gets computed in this method.
            This batch can be the same object as the input batch. In this case,
            the logic in this method should mutate the input batch directly.
        """

    @property
    def batch_size(self) -> int:
        """The size of the batches that this transform should process.

        Defaults to -1, in which case default batch size options will
        be used when executing the transform.
        """
        return -1  # pragma: no cover
