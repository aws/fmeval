import ray

from abc import abstractmethod
from typing import Any, Dict, Union, List, Optional
from ray.actor import ActorHandle

from fmeval.transforms.transform import Transform
from fmeval.transforms.util import validate_call
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModel
from fmeval.util import assert_condition

BERT_SCORE = "bertscore"


class QAAccuracyMetric(Transform):
    """The abstract base class for QA Accuracy metric transforms.

    Concrete subclasses of QAAccuracyMetric should simply implement the
    `compute_metric` method and their own __init__ method. Subclasses need not implement
    the __call__ method, as it is already implemented in this class, but are
    free to do so if additional customization is required.
    """

    def __init__(
        self,
        target_output_keys: List[str],
        model_output_keys: List[str],
        output_keys: List[str],
        allow_duplicate_input_keys: bool,
        *args,
        **kwargs,
    ):
        """QAAccuracyMetric initializer.

        Note that the ordering of the elements in `target_output_keys`, `model_output_keys`,
        and `output_keys` must match, i.e. the kth element of `target_output_keys` and the
        kth element of `model_output_keys` are used to compute the kth metric, which has
        an output key of `output_keys[k]`.

        :param target_output_keys: The keys corresponding to target outputs.
        :param model_output_keys: The keys corresponding to model outputs.
        :param output_keys: The output keys for this Transform, which correspond
            to the metrics/scores that get computed.
        :param allow_duplicate_input_keys: Whether to allow duplicate keys in
            `target_output_keys` and `model_output_keys`. This parameter is usually
            False, but will be True when a SummarizationAccuracyMetric is created
            to compute metrics on perturbed model outputs. In this case,
            `target_output_keys` will be a list of a single repeated key, while
            `model_output_keys` will contain the keys for perturbed model outputs.
        :param *args: Variable length argument list.
        :param **kwargs: Arbitrary keyword arguments.
        """
        assert_condition(
            len(target_output_keys) == len(model_output_keys) and len(target_output_keys) == len(output_keys),
            "len(target_output_keys), len(model_output_keys) and len(output_keys) should all match. "
            f"len(target_output_keys) is {len(target_output_keys)}, len(model_output_keys) is "
            f"{len(model_output_keys)}, and len(output_keys) is {len(output_keys)}.",
        )
        super().__init__(
            target_output_keys,
            model_output_keys,
            output_keys,
            allow_duplicate_input_keys,
            *args,
            **kwargs,
        )
        self.register_input_output_keys(
            target_output_keys + model_output_keys,
            output_keys,
            allow_duplicates=allow_duplicate_input_keys,
        )
        self.target_output_keys = target_output_keys
        self.model_output_keys = model_output_keys

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with metrics computed via self.compute_metric.

        :param record: The input record.
        :returns: The input record with metrics added in.
        """
        for target_output_key, model_output_key, output_key in zip(
            self.target_output_keys, self.model_output_keys, self.output_keys
        ):
            score = self.compute_metric(record[target_output_key], record[model_output_key])
            record[output_key] = score
        return record

    @abstractmethod
    def compute_metric(self, target_output: str, model_output: str) -> float:
        """Compute the metric that is specific to this Transform.

        :param target_output: The target/reference output.
        :param model_output: The actual output produced by the model.
        :returns: A float representing the computed metric value.
        """


class BertScoreWithDelimiter(QAAccuracyMetric):
    """This transform computes the maximum value of a bertscore given multiple target answers
    separated by a delimiter"""

    def __init__(
        self,
        target_output_keys: List[str],
        model_output_keys: List[str],
        output_keys: List[str],
        allow_duplicate_input_keys: bool,
        bertscore_model: Union[BertscoreHelperModel, ActorHandle],
        target_output_delimiter: Optional[str] = "<OR>",
    ):
        """BertScoreWithDelimiter initializer.

        :param target_output_keys: The keys corresponding to target outputs.
        :param model_output_keys: The keys corresponding to model outputs.
        :param output_keys: The output keys for this Transform, which correspond
            to the BERT scores that get computed.
        :param allow_duplicate_input_keys: See docstring for SummarizationAccuracyMetric.
        :param bertscore_model: A BertscoreHelperModel instance or a Ray actor handle for a BertscoreHelperModel.
        """
        super().__init__(
            target_output_keys,
            model_output_keys,
            output_keys,
            allow_duplicate_input_keys,
            bertscore_model,
            target_output_delimiter,
        )
        self.bertscore_model = bertscore_model
        self.target_output_delimiter = target_output_delimiter

    def compute_metric(self, target_output: str, model_output: str) -> float:
        """Compute the maximum BERTScore metric over multiple target answers separated by a target output delimiter.

        :param target_output: The target/reference output.
        :param model_output: The actual output produced by the model.
        :returns: The maximum BERT metric value.
        """
        possible_targets = target_output.split(self.target_output_delimiter)
        if isinstance(self.bertscore_model, BertscoreHelperModel):
            return max([self.bertscore_model.get_helper_scores(target, model_output) for target in possible_targets])
        else:
            possible_scores = list(
                map(
                    lambda x: self.bertscore_model.get_helper_scores.remote(x, model_output),  # type: ignore[return-value]
                    possible_targets,
                )
            )
            all_scores = ray.get(possible_scores)

            return max(all_scores)
