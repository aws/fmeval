import logging
import string
from functools import partial


from typing import Any, Callable, List, Optional, Dict

from dataclasses import dataclass

from nltk.metrics.scores import f_measure, precision, recall

import fmeval.util as util
from fmeval.constants import (
    MODEL_INPUT_COLUMN_NAME,
    MODEL_OUTPUT_COLUMN_NAME,
    TARGET_OUTPUT_COLUMN_NAME,
    MODEL_LOG_PROBABILITY_COLUMN_NAME,
    MEAN,
)
from fmeval.data_loaders.util import get_dataset
from fmeval.data_loaders.data_config import DataConfig
from fmeval.eval_algorithms.util import (
    generate_model_predict_response_for_dataset,
    generate_prompt_column_for_dataset,
    aggregate_evaluation_scores,
    validate_dataset,
    save_dataset,
    generate_output_dataset_path,
)
from fmeval.eval_algorithms.eval_algorithm import (
    EvalAlgorithmInterface,
    EvalAlgorithmConfig,
)
from fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalOutput,
    EvalScore,
    EVAL_DATASETS,
    DATASET_CONFIGS,
    get_default_prompt_template,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.perf_util import timed_block

ENGLISH_ARTICLES = ["a", "an", "the"]
ENGLISH_PUNCTUATIONS = string.punctuation

F1_SCORE = "f1_score"
EXACT_MATCH_SCORE = "exact_match_score"
QUASI_EXACT_MATCH_SCORE = "quasi_exact_match_score"
PRECISION = "precision"
RECALL = "recall"

PROMPT_COLUMN_NAME = "prompt"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QAAccuracyConfig(EvalAlgorithmConfig):
    """
    Configuration for the QA Accuracy Evaluation

    :param target_output_delimiter: Target Output can have multiple answers. We expect customer to combine all the
        possible answers into a single string and use the delimiter to separate them. For instance,
        if the answers are ["UK", "England"] and the delimiter="<OR>", then the target_output should be "UK<OR>England".
    """

    target_output_delimiter: Optional[str] = "<OR>"

    def __post_init__(self):
        if self.target_output_delimiter == "":
            raise EvalAlgorithmClientError(
                "Empty target_output_delimiter is provided. Please either provide a non-empty string, or set it to None."
            )


def _normalize_text_quac_protocol(text: str) -> str:
    """
    Inspired by HELM: https://github.com/stanford-crfm/helm/blob/62f817eb695a31e8389e3f7be30609d3f0871837/src/helm/benchmark/metrics/basic_metrics.py
    Given a text, normalize it using the SQUAD / QUAC protocol. That is remove punctuations, excess spaces and articles, and return the lowercased tokens.
    SQUAD (https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/) and
    QuAC benchmarks (https://s3.amazonaws.com/my89public/quac/scorer.py) use this protocol to normalize text before evaluating it.
    HELM (https://github.com/stanford-crfm/helm/blob/62f817eb695a31e8389e3f7be30609d3f0871837/src/helm/benchmark/metrics/basic_metrics.py#L116)
    and HuggingFace evaluate (https://github.com/huggingface/evaluate/blob/775555d80af30d83dc6e9f42051840d29a34f31b/metrics/squad/compute_score.py#L11)
    also use this to normalization procedure.

    :param text: The text that needs to be normalized.
    :returns: The normalized text.
    """

    text = text.lower()
    text = "".join(character for character in text if character not in ENGLISH_PUNCTUATIONS)
    return " ".join([word for word in text.split(" ") if (word != "" and word not in ENGLISH_ARTICLES)])


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
        text = _normalize_text_quac_protocol(text)
    return text


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
    ret = f_measure(reference=set(target_output.split(" ")), test=set(model_output.split(" ")))
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
    ret = precision(reference=set(target_output.split(" ")), test=set(model_output.split(" ")))
    if ret is None:  # pragma: no cover
        return 0.0
    else:
        return float(ret)


def _recall(model_output: str, target_output: str, *, normalize_text: bool = False, strip_text: bool = False) -> float:
    """
    Given the model output and the target output, compute the recall.
    Recall is the fraction of words in the target output that are also found in the answer.
    Before computing recall, we normalize the text following the QuAC protocol.

    :param model_output: The output of a model that we want to evaluate.
    :param target_output: The reference or the "ground truth" output.
    :param normalize_text: Normalize the text before computing recall.
    :param strip_text: Strip the model_output and the target_output before computing recall. Stripping amounts to removing whitespace characters from the beginning and end of the strings.
    :returns: Recall.
    """
    model_output = _normalize_and_strip_text(model_output, normalize_text=normalize_text, strip_text=strip_text)
    target_output = _normalize_and_strip_text(target_output, normalize_text=normalize_text, strip_text=strip_text)
    ret = recall(reference=set(target_output.split(" ")), test=set(model_output.split(" ")))
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

    :param model_output: The output of a model that we want to evaluate.
    :param target_output: The reference or the "ground truth" output.
    :returns: 1 if the two strings match after normalization else 0.
    """
    return float(
        _normalize_text_quac_protocol(model_output.strip()) == _normalize_text_quac_protocol(target_output.strip())
    )


QA_ACCURACY_SCORES_TO_FUNCS: Dict[str, Callable[..., float]] = {
    F1_SCORE: partial(_f1_score, normalize_text=True, strip_text=True),
    EXACT_MATCH_SCORE: _exact_match_score,
    QUASI_EXACT_MATCH_SCORE: _quasi_exact_match_score,
    PRECISION: partial(_precision, normalize_text=True, strip_text=True),
    RECALL: partial(_recall, normalize_text=True, strip_text=True),
}


class QAAccuracy(EvalAlgorithmInterface):
    """
    QA Accuracy Eval algorithm
    """

    eval_name = EvalAlgorithm.QA_ACCURACY.value

    def __init__(self, eval_algorithm_config: QAAccuracyConfig = QAAccuracyConfig()):
        """Default constructor

        :param eval_algorithm_config: QA Accuracy eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self._eval_algorithm_config = eval_algorithm_config

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        save: bool = False,
        num_records=100,
    ) -> List[EvalOutput]:
        """
        QA Accuracy evaluate.

        :param model: An instance of ModelRunner which is the model under evaluation
        :param dataset_config: The config to load the dataset to use for evaluation. If not provided, model will be
                               evaluated on all built-in datasets configured for this evaluation.
        :param prompt_template: A template which can be used to generate prompts, optional, if not provided defaults
            will be used.
        :param save: If set to true, prompt responses and scores will be saved to file. The output is written to
                     EvalAlgorithmInterface.EVAL_RESULTS_PATH
        :param num_records: The number of records to be sampled randomly from the input dataset to perform the
                            evaluation
        :returns: List of EvalOutput objects. Current implementation returns only one score.
        """
        if dataset_config:
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[self.eval_name]]

        eval_outputs: List[EvalOutput] = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [TARGET_OUTPUT_COLUMN_NAME, MODEL_INPUT_COLUMN_NAME])
            dataset_prompt_template = None
            if MODEL_OUTPUT_COLUMN_NAME not in dataset.columns():
                util.require(model, "No ModelRunner provided. ModelRunner is required for inference on model_inputs")
                dataset_prompt_template = (
                    get_default_prompt_template(dataset_config.dataset_name) if not prompt_template else prompt_template
                )
                dataset = generate_prompt_column_for_dataset(
                    prompt_template=dataset_prompt_template,
                    data=dataset,
                    model_input_column_name=MODEL_INPUT_COLUMN_NAME,
                    prompt_column_name=PROMPT_COLUMN_NAME,
                )
                assert model  # to satisfy mypy
                dataset = generate_model_predict_response_for_dataset(
                    model=model,
                    data=dataset,
                    model_input_column_name=PROMPT_COLUMN_NAME,
                    model_output_column_name=MODEL_OUTPUT_COLUMN_NAME,
                    model_log_probability_column_name=MODEL_LOG_PROBABILITY_COLUMN_NAME,
                )
            with timed_block(f"Computing score and aggregation on dataset {dataset_config.dataset_name}", logger):

                def _generate_eval_scores(row: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
                    """
                    Map function generating the scores for every input record in input dataset
                    """
                    for eval_score, eval_fn in QA_ACCURACY_SCORES_TO_FUNCS.items():
                        row[eval_score] = self._get_score(
                            target_output=row[TARGET_OUTPUT_COLUMN_NAME],
                            model_output=row[MODEL_OUTPUT_COLUMN_NAME],
                            eval_fn=eval_fn,
                        )
                    return row

                dataset = dataset.map(_generate_eval_scores).materialize()

                dataset_scores, category_scores = aggregate_evaluation_scores(
                    dataset, [F1_SCORE, EXACT_MATCH_SCORE, QUASI_EXACT_MATCH_SCORE, PRECISION, RECALL], agg_method=MEAN
                )

                eval_outputs.append(
                    EvalOutput(
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                        prompt_template=dataset_prompt_template,
                        dataset_scores=dataset_scores,
                        category_scores=category_scores,
                        output_path=generate_output_dataset_path(
                            path_to_parent_dir=self._eval_results_path,
                            eval_name=self.eval_name,
                            dataset_name=dataset_config.dataset_name,
                        ),
                    )
                )
            if save:
                save_dataset(
                    dataset=dataset,
                    score_names=list(QA_ACCURACY_SCORES_TO_FUNCS.keys()),
                    path=generate_output_dataset_path(
                        path_to_parent_dir=self._eval_results_path,
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                    ),
                )

        return eval_outputs

    def _get_score(
        self, target_output: str, model_output: str, eval_fn: Callable[..., float], **fn_kwargs: Any
    ) -> float:
        """
        Method to generate accuracy score for a target_output and model_output

        :param target_output: Target output
        :param model_output: Model output
        :returns: Computed score
        """
        possible_targets = target_output.split(self._eval_algorithm_config.target_output_delimiter)

        return max([eval_fn(model_output, target, **fn_kwargs) for target in possible_targets])

    def evaluate_sample(self, target_output: str, model_output: str) -> List[EvalScore]:  # type: ignore[override]
        """
        Evaluate a single QA record.

        :param target_output: The expected responses from the model.
        :param model_output: An instance of ModelOutput which contains the responses from the model needed for this
                             evaluation.
        :returns: A List of EvalScores computed for prompts and responses.
        """
        if target_output is None:
            raise EvalAlgorithmClientError("Missing required input: target_output, for QA Accuracy evaluate_sample")
        if model_output is None:
            raise EvalAlgorithmClientError("Missing required input: model_output, for QA Accuracy evaluate_sample")

        return [
            EvalScore(
                name=eval_score,
                value=self._get_score(target_output=target_output, model_output=model_output, eval_fn=eval_fn),
            )
            for eval_score, eval_fn in QA_ACCURACY_SCORES_TO_FUNCS.items()
        ]
