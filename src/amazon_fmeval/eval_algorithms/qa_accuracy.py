import logging
import string
from functools import partial

import pandas as pd

from typing import Any, Callable, List, Optional, Dict

from dataclasses import dataclass

from nltk.metrics.scores import f_measure

import amazon_fmeval.util as util
from amazon_fmeval.constants import (
    MODEL_INPUT_COLUMN_NAME,
    MODEL_OUTPUT_COLUMN_NAME,
    TARGET_OUTPUT_COLUMN_NAME,
    MODEL_LOG_PROBABILITY_COLUMN_NAME,
    MEAN,
)
from amazon_fmeval.data_loaders.util import get_dataset
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.eval_algorithms.util import (
    generate_model_predict_response_for_dataset,
    generate_prompt_column_for_dataset,
    aggregate_evaluation_scores,
    validate_dataset,
    save_dataset,
    generate_output_dataset_path,
)
from amazon_fmeval.eval_algorithms.eval_algorithm import (
    EvalAlgorithmInterface,
    EvalAlgorithmConfig,
)
from amazon_fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalOutput,
    EvalScore,
    EVAL_DATASETS,
    EVAL_PROMPT_TEMPLATES,
    DATASET_CONFIGS,
)
from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.model_runner import ModelRunner
from amazon_fmeval.perf_util import timed_block

ENGLISH_ARTICLES = ["a", "an", "the"]
ENGLISH_PUNCTUATIONS = string.punctuation

F1_SCORE = "f1_score"
EXACT_MATCH_SCORE = "exact_match_score"
QUASI_EXACT_MATCH_SCORE = "quasi_exact_match_score"

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
    Given a text, normalize it using the SQUAD / QUAC protocol. That is remove punctuations, excess whitespaces and articles, and return the lowercased tokens.
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


def _f1_score(model_output: str, target_output: str, *, normalize_text: bool = False) -> float:
    """
    Inspired by the implementation in HELM: https://github.com/stanford-crfm/helm/blob/62f817eb695a31e8389e3f7be30609d3f0871837/src/helm/benchmark/metrics/basic_metrics.py#L182

    Given the model output and the target output, compute the f1 score between the two.
    F1-score is the harmonic mean of precision and recall where precision is the number of
    words in the prediction that are also found in the target output and recall is the number
    of words in the target output that are also found in the answer. We normalize the text following
    the QuAC protocol above.

    :param model_output: The output of a model that we want to evaluate.
    :param target_output: The reference or the "ground truth" output.
    :param normalize_text: Normalize the text before computing f1.
    :returns: The F1 score.
    """
    if normalize_text:  # pragma: no branch
        model_output, target_output = (_normalize_text_quac_protocol(text) for text in (model_output, target_output))
    ret = f_measure(set(model_output.split(" ")), set(target_output.split(" ")))
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
    F1_SCORE: partial(_f1_score, normalize_text=True),
    EXACT_MATCH_SCORE: _exact_match_score,
    QUASI_EXACT_MATCH_SCORE: _quasi_exact_match_score,
}


class QAAccuracy(EvalAlgorithmInterface):
    """
    QA Accuracy Eval algorithm
    """

    eval_name = EvalAlgorithm.QA_ACCURACY.value

    def __init__(self, eval_algorithm_config: QAAccuracyConfig):
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
        is_custom_dataset_evaluation = False
        if dataset_config:
            is_custom_dataset_evaluation = True
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[self.eval_name]]

        eval_outputs: List[EvalOutput] = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config, num_records)
            validate_dataset(dataset, [TARGET_OUTPUT_COLUMN_NAME, MODEL_INPUT_COLUMN_NAME])
            if MODEL_OUTPUT_COLUMN_NAME not in dataset.columns():
                util.require(model, "No ModelRunner provided. ModelRunner is required for inference on model_inputs")
                if is_custom_dataset_evaluation:
                    # TODO when user provide built-in DataConfig, we should provide default prompt_template
                    util.require(
                        prompt_template is not None,
                        f"Missing required input: prompt_template for evaluating custom dataset : {dataset_config}",
                    )
                else:
                    prompt_template = EVAL_PROMPT_TEMPLATES[self.eval_name, dataset_config.dataset_name]
                    util.assert_condition(
                        prompt_template is not None,
                        f"No Prompt Template configured for ({self.eval_name}, {dataset_config.dataset_name})",
                    )
                assert prompt_template  # to satisfy mypy
                dataset = generate_prompt_column_for_dataset(
                    prompt_template=prompt_template,
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
                for eval_score, eval_fn in QA_ACCURACY_SCORES_TO_FUNCS.items():

                    def _generate_eval_scores(df: pd.DataFrame) -> pd.Series:  # pragma: no cover
                        """
                        Map function generating the scores for every input record in input dataset
                        """
                        return pd.Series(
                            data=[
                                self._get_score(
                                    target_output=row[TARGET_OUTPUT_COLUMN_NAME],
                                    model_output=row[MODEL_OUTPUT_COLUMN_NAME],
                                    eval_fn=eval_fn,
                                )
                                for index, row in df.iterrows()
                            ]
                        )

                    dataset = dataset.add_column(eval_score, _generate_eval_scores)
                    dataset = dataset.materialize()

                dataset_scores, category_scores = aggregate_evaluation_scores(
                    dataset, [F1_SCORE, EXACT_MATCH_SCORE, QUASI_EXACT_MATCH_SCORE], agg_method=MEAN
                )

                eval_outputs.append(
                    EvalOutput(
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                        prompt_template=prompt_template,
                        dataset_scores=dataset_scores,
                        category_scores=category_scores,
                        output_path=self._eval_results_path,
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
