import ray
import nltk
import evaluate as hf_evaluate

from abc import abstractmethod
from typing import Any, Dict, Union, List, Optional
from ray.actor import ActorHandle
from nltk import word_tokenize
from nltk.translate import meteor_score

from fmeval.constants import BERTSCORE_DEFAULT_MODEL
from fmeval.transforms.transform import Transform
from fmeval.transforms.util import validate_call
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModel
from fmeval.util import assert_condition

METEOR_SCORE = "meteor"
ROUGE_SCORE = "rouge"
BERT_SCORE = "bertscore"

# rouge constants
ROUGE_1 = "rouge1"
ROUGE_2 = "rouge2"
ROUGE_L = "rougeL"

ROUGE_TYPES = [ROUGE_1, ROUGE_2, ROUGE_L]


class SummarizationAccuracyMetric(Transform):
    """The abstract base class for summarization accuracy metric transforms.

    Concrete subclasses of SummarizationAccuracyMetric should simply implement the
    `compute_metric` method and their own __init__ method. Subclasses need not implement
    the __call__ method, as it is already implemented in this class, but are
    free to do so if additional customization is required.
    """

    def __init__(
        self,
        output_keys: List[str],
        model_output_keys: List[str],
        allow_duplicate_input_keys: bool,
        target_output_keys: Optional[List[str]] = None,
        target_output_keys_provider: str = "",
        *args,
        **kwargs,
    ):
        """SummarizationAccuracyMetric initializer.

        Note that the ordering of the elements in `model_output_keys`, and `output_keys`
        must match, i.e. the kth element of kth element of `model_output_keys` is used
        to compute the kth metric, which has an output key of `output_keys[k]`.

        :param output_keys: The output keys for this Transform, which correspond
            to the metrics/scores that get computed.
        :param model_output_keys: The keys corresponding to model outputs.
        :param allow_duplicate_input_keys: Whether to allow duplicate keys in
            `target_output_keys` and `model_output_keys`. This parameter is usually
            False.
        :param target_output_keys: The keys corresponding to target outputs. If none are
            provided, this will default to 'None' and will fall back on the
            `target_output_keys_provider`.
        :param target_output_keys_provider: The key corresponding to a list of target
            outputs. Will only be used if `target_output_keys` is not provided. This key
            must not be "" otherwise it will be considered invalid.
        :param *args: Variable length argument list.
        :param **kwargs: Arbitrary keyword arguments.
        """
        assert_condition(
            len(model_output_keys) == len(output_keys),
            "len(model_output_keys) and len(output_keys) should match. "
            f"len(model_output_keys) is "
            f"{len(model_output_keys)}, and len(output_keys) is {len(output_keys)}.",
        )
        assert_condition(
            target_output_keys is not None or target_output_keys_provider != "",
            f"target_output_keys is {target_output_keys}, and target_output_keys_provider"
            f" is {target_output_keys_provider}."
            "At least one must be valid.",
        )
        super().__init__(
            output_keys,
            model_output_keys,
            allow_duplicate_input_keys,
            target_output_keys,
            target_output_keys_provider,
            *args,
            **kwargs,
        )
        input_keys = [target_output_keys_provider] if target_output_keys is None else target_output_keys
        self.register_input_output_keys(
            input_keys + model_output_keys,
            output_keys,
            allow_duplicates=allow_duplicate_input_keys,
        )
        self.target_output_keys = target_output_keys
        self.model_output_keys = model_output_keys
        self.target_output_keys_provider = target_output_keys_provider

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with metrics computed via self.compute_metric.
        The max score is computed over all possible targets represented by
        self.target_output_keys and stored in the input record.

        :param record: The input record.
        :returns: The input record with metrics added in.
        """
        if self.target_output_keys is not None:
            for model_output_key, output_key in zip(self.model_output_keys, self.output_keys):
                scores = [
                    self.compute_metric(record[target], record[model_output_key]) for target in self.target_output_keys
                ]
                record[output_key] = max(scores)
        else:
            target_output_list = record[self.target_output_keys_provider]
            for model_output_key, output_key in zip(self.model_output_keys, self.output_keys):
                scores = [self.compute_metric(target, record[model_output_key]) for target in target_output_list]
                record[output_key] = max(scores)
        return record

    @abstractmethod
    def compute_metric(self, target_output: str, model_output: str) -> float:
        """Compute the metric that is specific to this Transform.

        :param target_output: The target/reference output.
        :param model_output: The actual output produced by the model.
        :returns: A float representing the computed metric value.
        """


class MeteorScore(SummarizationAccuracyMetric):
    """This Transform augments an input record with the METEOR metric, computed from target and model outputs.

    METEOR is a metric for text similarity between the machine-produced summary
    and human-produced reference summaries.
    Unigrams can be matched based on their surface forms, stemmed forms,
    and meanings; furthermore, METEOR can be easily extended to include more
    advanced matching strategies. Once all generalized unigram matches
    between the two strings have been found, METEOR computes a score for
    this matching using a combination of unigram-precision, unigram-recall, and
    a measure of fragmentation that is designed to directly capture how
    well-ordered the matched words in the machine translation are in relation
    to the reference.
    """

    def __init__(
        self,
        output_keys: List[str],
        model_output_keys: List[str],
        allow_duplicate_input_keys: bool,
        target_output_keys: Optional[List[str]] = None,
        target_output_keys_provider: str = "",
        load_modules: bool = True,
    ):
        """MeteorScore initializer.

        :param output_keys: The output keys for this Transform, which correspond
            to the Meteor scores that get computed.
        :param model_output_keys: The keys corresponding to model outputs.
        :param allow_duplicate_input_keys: Whether to allow duplicate keys in
            `target_output_keys` and `model_output_keys`.
        :param target_output_keys: The keys corresponding to target outputs.
        :param target_output_keys_provider: The key corresponding to a list of target
            outputs. Will only be used if `target_output_keys` is not provided.
        :param load_modules: Whether to load the meteor helper modules.
        """
        super().__init__(
            output_keys,
            model_output_keys,
            allow_duplicate_input_keys,
            target_output_keys,
            target_output_keys_provider,
            # The first instance of this class that gets created will
            # load the helper modules, so copies of this instance
            # need not load them again.
            load_modules=False,
        )
        if load_modules:
            MeteorScore._load_modules()

    @staticmethod
    def _load_modules() -> None:  # pragma: no cover
        """Load helper modules required by meteor metric.

        :returns: None
        """
        nltk.download("wordnet")
        nltk.download("punkt")
        nltk.download("omw-1.4")

    def compute_metric(self, target_output: str, model_output: str) -> float:
        """Compute the Meteor metric.

        :param target_output: The target/reference output.
        :param model_output: The actual output produced by the model.
        :returns: The meteor metric value.
        """
        return meteor_score.single_meteor_score(
            reference=word_tokenize(target_output),
            hypothesis=word_tokenize(model_output),
        )


class RougeScore(SummarizationAccuracyMetric):
    """This transform augments an input record with the ROUGE score, computed from target and model outputs.

    The ROUGE-N, where N=[1,2,L], score is a standard metric for summarization quality.
    It computes the word overlap between the reference and model summary. Given that this metric is based on simple
    word overlap statistics, it works best for extractive summaries.
    Note that if we rephrase the summary without changing its meaning the ROUGE-N score will drop.

    Reference: https://huggingface.co/spaces/evaluate-metric/rouge
    """

    def __init__(
        self,
        output_keys: List[str],
        model_output_keys: List[str],
        allow_duplicate_input_keys: bool,
        target_output_keys: Optional[List[str]] = None,
        target_output_keys_provider: str = "",
        rouge_type: str = ROUGE_2,
        use_stemmer: bool = True,
    ):
        """RougeScore initializer.

        :param output_keys: The output keys for this Transform, which correspond
            to the Rouge scores that get computed.
        :param model_output_keys: The keys corresponding to model outputs.
        :param allow_duplicate_input_keys: Whether to allow duplicate keys in
            `target_output_keys` and `model_output_keys`.
        :param target_output_keys: The keys corresponding to target outputs.
        :param target_output_keys_provider: The key corresponding to a list of target
            outputs. Will only be used if `target_output_keys` is not provided.
        :param rouge_type: Which ROUGE type to use (1, 2, L).
        :param use_stemmer: Whether to use a stemmer for ROUGE.
        """
        super().__init__(
            output_keys,
            model_output_keys,
            allow_duplicate_input_keys,
            target_output_keys,
            target_output_keys_provider,
            rouge_type=rouge_type,
            use_stemmer=use_stemmer,
        )
        self.rouge_type = rouge_type
        self.use_stemmer = use_stemmer
        self.rouge_metric = hf_evaluate.load("rouge")

    def compute_metric(self, target_output: str, model_output: str) -> float:
        """Compute the ROUGE metric.

        :param target_output: The target/reference output.
        :param model_output: The actual output produced by the model.
        :returns: The ROUGE metric value.
        """
        return self.rouge_metric.compute(
            predictions=[model_output],
            references=[target_output],
            use_stemmer=self.use_stemmer,
            rouge_types=[self.rouge_type],
        )[self.rouge_type]


class BertScore(SummarizationAccuracyMetric):
    """This transform augments an input record with the BERT score, computed from target and model outputs.

    BERTscore is a similarity-based metric that compares the embedding of the prediction and target sentences
    under a learned model, typically, from the BERT family.
    This score may lead to increased flexibility compared to ROUGE and METEOR in terms of rephrasing since
    semantically similar sentences are (typically) embedded similarly.

    See https://huggingface.co/spaces/evaluate-metric/bertscore
    """

    def __init__(
        self,
        output_keys: List[str],
        model_output_keys: List[str],
        target_output_keys: Optional[List[str]] = None,
        target_output_keys_provider: str = "",
        bertscore_model: Union[BertscoreHelperModel, ActorHandle] = BertscoreHelperModel(BERTSCORE_DEFAULT_MODEL),
    ):
        """BertScore initializer.

        :param output_keys: The output keys for this Transform, which correspond
            to the BERT scores that get computed.
        :param model_output_keys: The keys corresponding to model outputs.
        :param allow_duplicate_input_keys: Whether to allow duplicate keys in
            `target_output_keys` and `model_output_keys`.
        :param target_output_keys: The keys corresponding to target outputs.
        :param bertscore_model: A BertscoreHelperModel instance or a Ray actor handle for a BertscoreHelperModel.
            If no model is provided, the parameter will be set to the default BertscoreHelperModel
        """
        super().__init__(
            output_keys,
            model_output_keys,
            allow_duplicate_input_keys,
            target_output_keys,
            target_output_keys_provider,
            bertscore_model,
        )
        self.bertscore_model = bertscore_model

    def compute_metric(self, target_output: str, model_output: str) -> float:
        """Compute the BERTScore metric.

        :param target_output: The target/reference output.
        :param model_output: The actual output produced by the model.
        :returns: The BERT metric value.
        """
        if isinstance(self.bertscore_model, BertscoreHelperModel):
            return self.bertscore_model.get_helper_scores(target_output, model_output)
        else:
            return ray.get(  # type: ignore[return-value]
                self.bertscore_model.get_helper_scores.remote(target_output, model_output)  # type: ignore[union-attr]
            )
