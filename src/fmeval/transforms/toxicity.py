import itertools
from typing import Any, Dict, List, Union, Optional

from ray import ObjectRef
import ray

from fmeval.helper_models import ToxicityDetector
from fmeval.transforms.transform import Transform
from fmeval.transforms.util import validate_call
from fmeval.util import assert_condition


class ToxicityScores(Transform):
    def __init__(
        self,
        text_keys: List[str],
        score_names: List[str],
        toxicity_detector: Union[ToxicityDetector, ObjectRef],
        output_keys: Optional[List[str]] = None,
    ):
        """
        Transform for computing toxicity scores via a model.

        :param text_keys: the keys of the strings to be evaluated
        :param score_names: the names of the scores that will be computed by the detector
        :param toxicity_detector: A ToxicityDetector instance or a Ray actor handle for a ToxicityDetector.
        :param output_keys: optional keys for the outputs. If not given, the output keys will
            be formed as "{text_key}_{score_name}" for all possible combination, unless
            text_keys is a singleton, in which case only the score names will be used as output keys.
        """
        super().__init__(text_keys, score_names, toxicity_detector, output_keys)
        if output_keys is not None:
            assert_condition(
                len(text_keys) == len(output_keys),
                "When output keys are given, they should have the same length as text_keys."
                f"Found len(text_keys)={len(text_keys)} and len(output_keys)={len(output_keys)}",
            )
        self.text_keys = text_keys
        self.score_names = score_names
        self.toxicity_detector = toxicity_detector
        if len(self.text_keys) == 1 and output_keys is None:
            output_keys = ['']  # default behaviour when there's only one input: use only the score names
        self.out_keys = output_keys if output_keys is not None else self.text_keys  # use the text_keys as default,
        # note this is not the self.output_keys that becomes populated by the method below
        self.register_input_output_keys(
            input_keys=text_keys,
            output_keys=[
                ToxicityScores._make_key(*pair) for pair in itertools.product(self.out_keys, self.score_names)
            ],
        )

    @staticmethod
    def _make_key(name: str, score_name: str):
        """Internal name constructor for toxicity output keys."""
        return f"{name}_{score_name}" if name else score_name

    @property
    def batch_size(self) -> Union[bool, int]:
        # In the future we should determine this number depending on the instance we're on. 64 should work fine for
        # both models and across machines, hopefully. We can change this if needed
        return 64

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        for inpt, out in zip(self.text_keys, self.out_keys):
            text_inputs = list(record[inpt])  # this converts np array that come from ray to list, accepted by
            # the HuggingFace tokenizers
            if isinstance(self.toxicity_detector, ToxicityDetector):
                scores = self.toxicity_detector.invoke_model(text_inputs)
            else:
                scores = ray.get(  # type: ignore[return-value]
                    self.toxicity_detector.invoke_model.remote(text_inputs)  # type: ignore[union-attr]
                )
            # scores is a {score: List[float]} dict
            for score, value in scores.items():
                record[ToxicityScores._make_key(out, score)] = value
        return record
