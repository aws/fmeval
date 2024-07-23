import logging

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass
from nltk.metrics.scores import f_measure, precision, recall
from fmeval.constants import (
    BERTSCORE_DEFAULT_MODEL,
    DatasetColumns,
    MEAN,
)
from fmeval.data_loaders.util import get_dataset
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.common import evaluate_dataset
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModelTypes, BertscoreHelperModel
from fmeval.eval_algorithms.save_strategy import SaveStrategy
from fmeval.eval_algorithms.util import validate_dataset, get_dataset_configs, normalize_text_quac_protocol
from fmeval.eval_algorithms.eval_algorithm import EvalAlgorithmConfig, EvalAlgorithmInterface
from fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalOutput,
    EvalScore,
)
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.transform import Transform
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.transforms.util import validate_call
from fmeval.util import (
    get_eval_results_path,
    require,
    create_shared_resource,
    cleanup_shared_resource,
    assert_condition,
)
from fmeval.transforms.qa_accuracy_metrics import BertScoreWithDelimiter, BERT_SCORE

F1_SCORE = "f1_score"
EXACT_MATCH_SCORE = "exact_match_score"
QUASI_EXACT_MATCH_SCORE = "quasi_exact_match_score"
PRECISION_OVER_WORDS = "precision_over_words"
RECALL_OVER_WORDS = "recall_over_words"
BERT_SCORE = "bert_score"

# for metrics that are included in the QAAccuracyScores Transform
QA_ACCURACY_SCORE_NAMES = [
    F1_SCORE,
    EXACT_MATCH_SCORE,
    QUASI_EXACT_MATCH_SCORE,
    PRECISION_OVER_WORDS,
    RECALL_OVER_WORDS,
]

# for all metrics in qa_accuracy (metrics from both the QAAccuracyScores Transform and the BertScore Transform)
SCORE_NAMES = QA_ACCURACY_SCORE_NAMES + [BERT_SCORE]

logger = logging.getLogger(__name__)


def _normalize_and_strip_text(text: str, *, normalize_text: bool = False, strip_text: bool = False) -> str:
    """
    Combine two common operations -- normalization and stripping -- used by several metrics.
    :param normalize_text: Normalize the text. We use the QuAC protocol for normalization.
    :param strip_text: Strip the text, that is, remove whitespace characters from the beginning and end of the text.
    :returns: The normalized (if the normalize_text flag was set to True) and stripped (if the strip_text flag was set
              to True). If neither of the flags was set, the function returns the original text.
    """
    if strip_text:
        text = text.strip()
    if normalize_text:  # pragma: no branch
        text = normalize_text_quac_protocol(text)
    return text


def _split(text: str) -> Set[str]:
    """
    Splits the text to compute precision, recall scores and F1-score based on string.whitespace characters
     (namely ' \t\n\r\x0b\x0c') and converting the resulting list into a set.
    """
    return set(text.split())


def _f1_score(
    model_output: str, target_output: str, *, normalize_text: bool = False, strip_text: bool = False
) -> float:
    """
    Inspired by the implementation in HELM: https://github.com/stanford-crfm/helm/blob/62f817eb695a31e8389e3f7be30609d3f0871837/src/helm/benchmark/metrics/basic_metrics.py#L182

    Given the model output and the target output, compute the f1 score between the two.
    F1-score is the harmonic mean of precision and recall where precision is the number of
    words in the prediction that are also found in the target output and recall is the number
    of words in the target output that are also found in the answer.

    :param model_output: The output of a model that we want to evaluate.
    :param target_output: The reference or the "ground truth" output.
    :param normalize_text: Normalize the text before computing f1. We normalize the text following the QuAC protocol.
    :param strip_text: Strip the model_output and the target_output before computing the f1 score. Stripping amounts to removing whitespace characters from the beginning and end of the strings.
    :returns: The F1 score.
    """
    model_output = _normalize_and_strip_text(model_output, normalize_text=normalize_text, strip_text=strip_text)
    target_output = _normalize_and_strip_text(target_output, normalize_text=normalize_text, strip_text=strip_text)
    ret = f_measure(reference=_split(target_output), test=_split(model_output))
    if ret is None:  # pragma: no cover
        return 0.0
    else:
        return float(ret)


def _precision(
    model_output: str, target_output: str, *, normalize_text: bool = False, strip_text: bool = False
) -> float:
    """
    Given the model output and the target output, compute the precision.
    Precision is the fraction of words in the prediction that are also found in the target output.
    Before computing precision, we normalize the text following the QuAC protocol.

    :param model_output: The output of a model that we want to evaluate.
    :param target_output: The reference or the "ground truth" output.
    :param normalize_text: Normalize the text before computing precision.
    :param strip_text: Strip the model_output and the target_output before computing precision. Stripping amounts to removing whitespace characters from the beginning and end of the strings.
    :returns: Precision.
    """
    model_output = _normalize_and_strip_text(model_output, normalize_text=normalize_text, strip_text=strip_text)
    target_output = _normalize_and_strip_text(target_output, normalize_text=normalize_text, strip_text=strip_text)
    ret = precision(reference=_split(target_output), test=_split(model_output))
    if ret is None:  # pragma: no cover
        return 0.0
    else:
        return float(ret)


def _recall(model_output: str, target_output: str, *, normalize_text: bool = False, strip_text: bool = False) -> float:
    """
    Given the model output and the target output, compute the recall.
    Recall is the fraction of words in the target output that are also found in the prediction.
    Before computing recall, we normalize the text following the QuAC protocol.

    :param model_output: The output of a model that we want to evaluate.
    :param target_output: The reference or the "ground truth" output.
    :param normalize_text: Normalize the text before computing recall.
    :param strip_text: Strip the model_output and the target_output before computing recall. Stripping amounts to removing whitespace characters from the beginning and end of the strings.
    :returns: Recall.
    """
    model_output = _normalize_and_strip_text(model_output, normalize_text=normalize_text, strip_text=strip_text)
    target_output = _normalize_and_strip_text(target_output, normalize_text=normalize_text, strip_text=strip_text)
    ret = recall(reference=_split(target_output), test=_split(model_output))
    if ret is None:  # pragma: no cover
        return 0.0
    else:
        return float(ret)


def _exact_match_score(model_output: str, target_output: str) -> float:
    """
    Inspired by HELM: https://github.com/stanford-crfm/helm/blob/62f817eb695a31e8389e3f7be30609d3f0871837/src/helm/benchmark/metrics/basic_metrics.py#L137
    Computes if the two strings exactly match.

    :param model_output: The output of a model that we want to evaluate.
    :param target_output: The reference or the "ground truth" output.
    :returns: 0 is the two inputs do not match, else 1.
    """
    return float(model_output.strip() == target_output.strip())


def _quasi_exact_match_score(model_output: str, target_output: str) -> float:
    """
    Inspired by HELM: https://github.com/stanford-crfm/helm/blob/62f817eb695a31e8389e3f7be30609d3f0871837/src/helm/benchmark/metrics/basic_metrics.py#L144
    Computes if the two strings exactly match after normalizing them.

    Normalization: Given a text, normalize it using the SQUAD/QUAC protocol (remove punctuations, excess spaces,
    and articles) and return the lowercased tokens.
    SQUAD (https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/) and
    QuAC benchmarks (https://s3.amazonaws.com/my89public/quac/scorer.py) use this protocol to normalize text before
    evaluating it. Can learn more at fmeval/src/fmeval/eval_algorithms/util.py

    :param model_output: The output of a model that we want to evaluate.
    :param target_output: The reference or the "ground truth" output.
    :returns: 1 if the two strings match after normalization else 0.
    """
    return float(
        normalize_text_quac_protocol(model_output.strip()) == normalize_text_quac_protocol(target_output.strip())
    )


QA_ACCURACY_SCORES_TO_FUNCS: Dict[str, Callable[..., float]] = {
    F1_SCORE: partial(_f1_score, normalize_text=True, strip_text=True),
    EXACT_MATCH_SCORE: _exact_match_score,
    QUASI_EXACT_MATCH_SCORE: _quasi_exact_match_score,
    PRECISION_OVER_WORDS: partial(_precision, normalize_text=True, strip_text=True),
    RECALL_OVER_WORDS: partial(_recall, normalize_text=True, strip_text=True),
}


class QAAccuracyScores(Transform):
    def __init__(
        self,
        target_output_key: str = DatasetColumns.TARGET_OUTPUT.value.name,
        model_output_key: str = DatasetColumns.MODEL_OUTPUT.value.name,
        output_keys: List[str] = QA_ACCURACY_SCORE_NAMES,
        target_output_delimiter: Optional[str] = "<OR>",
    ):
        super().__init__(target_output_key, model_output_key, output_keys, target_output_delimiter)
        self.register_input_output_keys(
            input_keys=[target_output_key, model_output_key],
            output_keys=output_keys,
        )
        self.target_output_key = target_output_key
        self.model_output_key = model_output_key
        self.output_keys = output_keys
        self.target_output_delimiter = target_output_delimiter

    def _get_score(
        self,
        target_output: str,
        model_output: str,
        score_fn: Callable[..., float],
        **fn_kwargs,
    ) -> float:
        """Compute an accuracy score from target_output and model_output.

        :param target_output: A single string potentially containing multiple
            target output values. If there are multiple target output values,
            they will be separated by `target_output_delimiter`.
            For example, if valid target outputs for a question are ["UK", "England"]
            and the delimiter is "<OR>", then `target_output` will be "UK<OR>England".
        :param model_output: The model output.
        :param target_output_delimiter: The delimiter used to separate the possible
            target outputs within the `target_output` string.
        :param score_fn: One of the functions in QA_ACCURACY_SCORES_TO_FUNCS.
        :returns: A computed QA accuracy score.
        """
        possible_targets = target_output.split(self.target_output_delimiter)
        return max([score_fn(model_output, target, **fn_kwargs) for target in possible_targets])

    @validate_call
    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        target_output = record[self.target_output_key]
        model_output = record[self.model_output_key]
        for output_key, score_name in zip(self.output_keys, QA_ACCURACY_SCORE_NAMES):
            record[output_key] = self._get_score(
                target_output=target_output,
                model_output=model_output,
                score_fn=QA_ACCURACY_SCORES_TO_FUNCS[score_name],
            )
        return record


@dataclass(frozen=True)
class QAAccuracyConfig(EvalAlgorithmConfig):
    """Configures the QA Accuracy evaluation algorithm.

    :param target_output_delimiter: There can be multiple valid target outputs for a given question.
        This delimiter is used to combine all possible target outputs into a single string.
        For example, if valid answers are ["UK", "England"] and the delimiter is "<OR>", then the
        target output text will be "UK<OR>England".
    :param model_type_for_bertscore: BERT model type to use for computing BERT score.
    """

    target_output_delimiter: Optional[str] = "<OR>"
    model_type_for_bertscore: str = BERTSCORE_DEFAULT_MODEL

    def __post_init__(self):
        require(
            self.target_output_delimiter != "",
            "Empty target_output_delimiter is provided. "
            "Please either provide a non-empty string, or set it to None.",
        )
        require(
            BertscoreHelperModelTypes.model_is_allowed(self.model_type_for_bertscore),
            f"Invalid model_type_for_bertscore: {self.model_type_for_bertscore} requested in "
            f"QAAccuracyConfig. Please choose from acceptable values: "
            f"{BertscoreHelperModelTypes.model_list()}.",
        )


class QAAccuracy(EvalAlgorithmInterface):
    """
    This evaluation measures how well the model performs in question answering (QA) tasks. The model is queried
    for a range of facts, and we evaluate the accuracy of its response by comparing model output to target answer under different metrics:

    1. Exact match (EM): Binary score, 1 if model output and target answer match exactly.
    2. Quasi-exact match: Binary score. Similar to exact match, but both model output and target answer are normalized first
    by removing any articles and punctuation.
    3. Precision over Words: The fraction of words in the prediction that are also found in the target answer. The text is normalized as before.
    4. Recall over Words: The fraction of words in the target answer that are also found in the prediction.
    5. F1 over Words: The harmonic mean of precision and recall, over words (normalized).
    6. [BERTScore](https://arxiv.org/pdf/1904.09675.pdf) uses a second ML model (from the BERT family) to compute sentence embeddings and compare their cosine similarity. This score may account for additional linguistic flexibility over ROUGE and METEOR since semantically similar sentences should be embedded closer to each other.


    Precision, Recall and F1 over Words are more flexible as they assign non-zero scores to
    model answers containing parts of the ground truth. Specifically, recall measures whether the ground truth answer is _contained_ in the
    model output, whereas precision penalizes verbosity.

    All metrics are reported on average over `num_records` datapoints and per category, resulting in a number between 0
    (worst) and 1 (best) for each metric.
    """

    eval_name = EvalAlgorithm.QA_ACCURACY.value

    def __init__(self, eval_algorithm_config: QAAccuracyConfig = QAAccuracyConfig()):
        """QAAccuracy initializer.

        :param eval_algorithm_config: QA Accuracy evaluation algorithm config.
        """
        super().__init__(eval_algorithm_config)

        self.bertscore_model = BertscoreHelperModel(eval_algorithm_config.model_type_for_bertscore)
        qa_accuracy_score = QAAccuracyScores(target_output_delimiter=eval_algorithm_config.target_output_delimiter)
        bert_score = BertScoreWithDelimiter(
            target_output_keys=[DatasetColumns.TARGET_OUTPUT.value.name],
            model_output_keys=[DatasetColumns.MODEL_OUTPUT.value.name],
            output_keys=[BERT_SCORE],
            allow_duplicate_input_keys=True,
            bertscore_model=self.bertscore_model,
            target_output_delimiter=eval_algorithm_config.target_output_delimiter,
        )
        self._eval_algorithm_config = eval_algorithm_config
        self.qa_accuracy_score = qa_accuracy_score  # saving the QAAccuracyScores transform
        self.bert_score = bert_score  # saving the BertScore transform
        self.pipeline = TransformPipeline([qa_accuracy_score, bert_score])

    def evaluate_sample(self, target_output: str, model_output: str) -> List[EvalScore]:
        """Compute QA accuracy metrics for a single sample.

        :param target_output: The expected/desired model output.
        :param model_output: The actual model output.
        :returns: A list of EvalScore objects, one for each of the QA accuracy metrics.
        """
        sample = {
            DatasetColumns.TARGET_OUTPUT.value.name: target_output,
            DatasetColumns.MODEL_OUTPUT.value.name: model_output,
        }
        result = self.pipeline.execute_record(sample)
        return [EvalScore(name=score_name, value=result[score_name]) for score_name in SCORE_NAMES]

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[Union[DataConfig, List[DataConfig]]] = None,
        prompt_template: Optional[str] = None,
        num_records: int = 100,
        save: bool = False,
        save_strategy: Optional[SaveStrategy] = None,
    ) -> List[EvalOutput]:
        """Compute QA accuracy metrics on one or more datasets.

        :param model: An instance of ModelRunner representing the model under evaluation.
            If this argument is None, the `dataset_config` argument must not be None,
            and must correspond to a dataset that already contains a column with model outputs.
        :param dataset_config: Configures a single dataset or list of datasets used for the
            evaluation. If not provided, this method will run evaluations using all of its
            supported built-in datasets.
        :param prompt_template: A template used to generate prompts that are fed to the model.
            If not provided, defaults will be used. If provided, `model` must not be None.
        :param num_records: The number of records to be sampled randomly from the input dataset(s)
            used to perform the evaluation(s).
        :param save: If set to true, prompt responses and scores will be saved to a file.
        :param save_strategy: Specifies the strategy to use the save the localized outputs of the evaluations. If not
            specified, it will save it to the path that can be configured by the EVAL_RESULTS_PATH environment variable.
            If that environment variable is also not configured, it will be saved to the default path `/tmp/eval_results/`.

        :return: A list of EvalOutput objects.
        """
        # Create a shared resource to be used during the evaluation.
        bertscore_shared_resource = create_shared_resource(self.bertscore_model)
        bert_score = BertScoreWithDelimiter(
            target_output_keys=[DatasetColumns.TARGET_OUTPUT.value.name],
            model_output_keys=[DatasetColumns.MODEL_OUTPUT.value.name],
            output_keys=[BERT_SCORE],
            allow_duplicate_input_keys=True,
            bertscore_model=bertscore_shared_resource,
            target_output_delimiter=self.qa_accuracy_score.target_output_delimiter,
        )

        # Create a new pipeline that uses the shared resource instead of self.bertscore_model.
        pipeline = TransformPipeline([self.qa_accuracy_score, bert_score])

        dataset_configs = get_dataset_configs(dataset_config, self.eval_name)
        eval_outputs = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [DatasetColumns.TARGET_OUTPUT.value.name])
            eval_output = evaluate_dataset(
                dataset=dataset,
                pipeline=pipeline,
                dataset_name=dataset_config.dataset_name,
                eval_name=self.eval_name,
                metric_names=SCORE_NAMES,
                eval_results_path=get_eval_results_path(),
                model=model,
                prompt_template=prompt_template,
                agg_method=MEAN,
                save=save,
                save_strategy=save_strategy,
            )
            eval_outputs.append(eval_output)
        cleanup_shared_resource(bertscore_shared_resource)
        return eval_outputs
