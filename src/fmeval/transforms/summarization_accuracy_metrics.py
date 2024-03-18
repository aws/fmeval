import ray
import nltk
import evaluate as hf_evaluate

from typing import Any, Dict, Union
from ray import ObjectRef
from nltk import word_tokenize
from nltk.translate import meteor_score

from fmeval.transforms.transform import Transform
from fmeval.transforms.util import validate_call
from fmeval.helper_models import BertscoreModel


METEOR_SCORE = "meteor"
ROUGE_SCORE = "rouge"
BERT_SCORE = "bertscore"

# rouge constants
ROUGE_1 = "rouge1"
ROUGE_2 = "rouge2"
ROUGE_L = "rougeL"

ROUGE_TYPES = [ROUGE_1, ROUGE_2, ROUGE_L]


def _load_meteor_modules() -> None:  # pragma: no cover
    """Load helper modules required by meteor metric.

    :returns: None
    """
    nltk.download("wordnet")
    nltk.download("punkt")
    nltk.download("omw-1.4")


class MeteorScore(Transform):
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
        output_key: str,
        target_output_key: str,
        model_output_key: str,
        load_meteor_modules: bool = True,
    ):
        """MeteorScore initializer.

        :param output_key: The output key for this Transform, which corresponds to
            the METEOR metric that gets computed.
        :param target_output_key: The key corresponding to the target output.
        :param model_output_key: The key corresponding to the model (LLM) output.
        :param load_meteor_modules: Whether to load the meteor helper modules.
        """
        super().__init__(
            output_key,
            target_output_key,
            model_output_key,
            # The first instance of this class that gets created will
            # load the helper modules, so copies of this instance
            # need not load them again.
            load_meteor_modules=False,
        )
        self.register_input_output_keys([target_output_key, model_output_key], [output_key])
        self.output_key = output_key
        self.target_output_key = target_output_key
        self.model_output_key = model_output_key
        if load_meteor_modules:
            _load_meteor_modules()

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with the computed METEOR metric.

        :param record: The input record.
        :returns: The input record with the METEOR metric added in.
        """
        record[self.output_key] = meteor_score.single_meteor_score(
            reference=word_tokenize(record[self.target_output_key]),
            hypothesis=word_tokenize(record[self.model_output_key]),
        )
        return record


class RougeScore(Transform):
    """This transform augments an input record with the ROUGE score, computed from target and model outputs.

    The ROUGE-N, where N=[1,2,L], score is a standard metric for summarization quality.
    It computes the word overlap between the reference and model summary. Given that this metric is based on simple
    word overlap statistics, it works best for extractive summaries.
    Note that if we rephrase the summary without changing its meaning the ROUGE-N score will drop.

    Reference: https://huggingface.co/spaces/evaluate-metric/rouge
    """

    def __init__(
        self,
        output_key: str,
        target_output_key: str,
        model_output_key: str,
        rouge_type: str = ROUGE_2,
        use_stemmer: bool = True,
    ):
        """RougeScore initializer.

        :param output_key: The output key for this Transform, which corresponds
            to the ROUGE score that gets computed.
        :param target_output_key: The key corresponding to the target output.
        :param model_output_key: The key corresponding to the model (LLM) output.
        :param rouge_type: Which ROUGE type to use (1, 2, L).
        :param use_stemmer: Whether to use a stemmer for ROUGE.
        """
        super().__init__(
            output_key, target_output_key, model_output_key, rouge_type=rouge_type, use_stemmer=use_stemmer
        )
        self.register_input_output_keys([target_output_key, model_output_key], [output_key])
        self.output_key = output_key
        self.target_output_key = target_output_key
        self.model_output_key = model_output_key
        self.rouge_type = rouge_type
        self.use_stemmer = use_stemmer
        self.rouge_metric = hf_evaluate.load("rouge")

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with the computed ROUGE score.

        :param record: The input record.
        :returns: The input record with the ROUGE score added in.
        """
        record[self.output_key] = self.rouge_metric.compute(
            predictions=[record[self.model_output_key]],
            references=[record[self.target_output_key]],
            use_stemmer=self.use_stemmer,
            rouge_types=[self.rouge_type],
        )[self.rouge_type]
        return record


class BertScore(Transform):
    """This transform augments an input record with the BERT score, computed from target and model outputs.

    BERTscore is a similarity-based metric that compares the embedding of the prediction and target sentences
    under a learned model, typically, from the BERT family.
    This score may lead to increased flexibility compared to ROUGE and METEOR in terms of rephrasing since
    semantically similar sentences are (typically) embedded similarly.

    See https://huggingface.co/spaces/evaluate-metric/bertscore
    """

    def __init__(
        self,
        output_key: str,
        target_output_key: str,
        model_output_key: str,
        bertscore_model: Union[BertscoreModel, ObjectRef],
    ):
        """BertScore initializer.

        :param output_key: The output key for this Transform, which corresponds
            to the BERT score that gets computed.
        :param target_output_key: The key corresponding to the target output.
        :param model_output_key: The key corresponding to the model (LLM) output.
        :param bertscore_model: A BertscoreModel instance or a Ray actor handle for a BertscoreModel.
        """
        super().__init__(output_key, target_output_key, model_output_key, bertscore_model)
        self.register_input_output_keys([target_output_key, model_output_key], [output_key])
        self.output_key = output_key
        self.target_output_key = target_output_key
        self.model_output_key = model_output_key
        self.bertscore_model = bertscore_model

    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Augment the input record with the computed BERT score.

        :param record: The input record.
        :returns: The input record with the BERT score added in.
        """
        score = (
            self.bertscore_model.invoke_model(record[self.target_output_key], record[self.model_output_key])
            if isinstance(self.bertscore_model, BertscoreModel)
            else ray.get(
                self.bertscore_model.invoke_model.remote(  # type: ignore[union-attr]
                    record[self.target_output_key], record[self.model_output_key]
                )
            )
        )
        record[self.output_key] = score
        return record
