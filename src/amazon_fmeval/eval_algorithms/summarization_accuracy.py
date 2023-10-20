import logging
from dataclasses import dataclass
from typing import Optional, List, Callable

import evaluate as hf_evaluate
import nltk
import pandas as pd
from datasets import Dataset
from nltk import word_tokenize
from nltk.translate import meteor_score

import amazon_fmeval.util as util
from amazon_fmeval.constants import (
    TARGET_OUTPUT_COLUMN_NAME,
    MODEL_INPUT_COLUMN_NAME,
    MODEL_OUTPUT_COLUMN_NAME,
    MEAN,
)
from amazon_fmeval.data_loaders.util import DataConfig, get_dataset
from amazon_fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalScore,
    EvalOutput,
    EVAL_DATASETS,
    DATASET_CONFIGS,
    get_default_prompt_template,
)
from amazon_fmeval.eval_algorithms.eval_algorithm import (
    EvalAlgorithmConfig,
    EvalAlgorithmInterface,
)
from amazon_fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModel
from amazon_fmeval.eval_algorithms.util import (
    generate_prompt_column_for_dataset,
    generate_model_predict_response_for_dataset,
    validate_dataset,
    aggregate_evaluation_scores,
    generate_output_dataset_path,
    save_dataset,
)
from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.model_runner import ModelRunner
from amazon_fmeval.perf_util import timed_block

PROMPT_COLUMN_NAME = "prompt"
METEOR_SCORE = "meteor"
ROUGE_SCORE = "rouge"
BERT_SCORE = "bertscore"

# rouge constants
ROUGE_1 = "rouge1"
ROUGE_2 = "rouge2"
ROUGE_L = "rougeL"

ROUGE_TYPES = [ROUGE_1, ROUGE_2, ROUGE_L]

# bertscore constants
MICROSOFT_DEBERTA_MODEL = "microsoft/deberta-xlarge-mnli"
ROBERTA_MODEL = "roberta-large-mnli"
DEFAULT_MODEL_TYPE = MICROSOFT_DEBERTA_MODEL
MODEL_TYPES_SUPPORTED = [MICROSOFT_DEBERTA_MODEL, ROBERTA_MODEL]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SummarizationAccuracyConfig(EvalAlgorithmConfig):
    """
    Configuration for the summarization accuracy eval algorithm

    :param rouge_type: Type of rouge metric in eval results
    :param use_stemmer_for_rouge: bool value to set using stemmer for rouge metric
    :param model_type_for_bertscore: model to use for bert score
    """

    rouge_type: str = ROUGE_2
    use_stemmer_for_rouge: bool = True
    model_type_for_bertscore: str = DEFAULT_MODEL_TYPE

    def __post_init__(self):
        if not self.rouge_type in ROUGE_TYPES:
            raise EvalAlgorithmClientError(
                f"Invalid rouge_type: {self.rouge_type} requested in SummarizationAccuracyConfig, "
                f"please choose from acceptable values: {ROUGE_TYPES}"
            )

        if not self.model_type_for_bertscore in MODEL_TYPES_SUPPORTED:
            raise EvalAlgorithmClientError(
                f"Invalid model_type_for_bertscore: {self.model_type_for_bertscore} requested in "
                f"SummarizationAccuracyConfig, please choose from acceptable values: {MODEL_TYPES_SUPPORTED}"
            )


class SummarizationAccuracy(EvalAlgorithmInterface):
    """
    Summarization Accuracy Eval algorithm

    The aim of this eval algo is to evaluate how well a model can summarise text.
    The algo uses a reference summary to compare the output generated by the model and a series
    of quality metrics based on overlapping between words (ROUGE and METEOR) and similarity scores (bert scores)
    """

    def __init__(self, eval_algorithm_config: SummarizationAccuracyConfig = SummarizationAccuracyConfig()):
        """Default constructor

        :param eval_algorithm_config: Summarization Accuracy eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.eval_name = EvalAlgorithm.SUMMARIZATION_ACCURACY.value
        self._eval_algorithm_config = eval_algorithm_config
        self._load_eval_helpers()
        self._score_eval_func_mapping = {
            METEOR_SCORE: get_meteor_score,
            ROUGE_SCORE: get_rouge_score,
            BERT_SCORE: get_bert_score,
        }

    def _load_eval_helpers(self):
        """
        Method to download required helpers for eval_algo in constructor call
        """
        # load helper modules for meteor
        nltk.download("wordnet")
        nltk.download("punkt")
        nltk.download("omw-1.4")

        # load HelperMode for bertscore
        BertscoreHelperModel(model_type=self._eval_algorithm_config.model_type_for_bertscore)

    def evaluate_sample(self, target_output: str, model_output: str) -> List[EvalScore]:  # type: ignore[override]
        """
        Summarization Accuracy evaluate sample.

        :param target_output: The expected responses from the model
        :param model_output: The output of a model that we want to evaluate.
        :return: list of EvalScore objects
        """
        if target_output is None:
            raise EvalAlgorithmClientError(
                "Missing required input: target_output, for Summarization Accuracy evaluate_sample"
            )
        if model_output is None:
            raise EvalAlgorithmClientError(
                "Missing required input: model_output, for Summarization Accuracy evaluate_sample"
            )

        return [
            EvalScore(
                name=eval_score,
                value=eval_fn(
                    target_output=target_output, model_output=model_output, config=self._eval_algorithm_config
                ),
            )
            for eval_score, eval_fn in self._score_eval_func_mapping.items()
        ]

    def evaluate(
        self,
        model: Optional[ModelRunner] = None,
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        save: bool = False,
        num_records=100,
    ) -> List[EvalOutput]:
        """
        Summarization Accuracy evaluate

        :param model: An instance of ModelRunner which is the model under evaluation
        :param dataset_config: Configures the single dataset used for evaluation. If not provided,
            evaluation will use all of it's supported built-in datasets
        :param prompt_template: A template which can be used to generate prompts, optional, if not provided defaults
            will be used.
        :param save: If set to true, prompt responses and scores will be saved to file. The output is written to
                     EvalAlgorithmInterface.EVAL_RESULTS_PATH
        :param num_records: The number of records to be sampled randomly from the input dataset to perform the
                            evaluation

        :return: List of EvalOutput objects.
        """
        if dataset_config:
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[self.eval_name]]

        eval_outputs = []
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
                    dataset_prompt_template, dataset, MODEL_INPUT_COLUMN_NAME, PROMPT_COLUMN_NAME
                )
                assert model  # to satisfy mypy
                dataset = generate_model_predict_response_for_dataset(
                    model, dataset, PROMPT_COLUMN_NAME, MODEL_OUTPUT_COLUMN_NAME
                )
            with timed_block(f"Computing score and aggregation on dataset {dataset_config.dataset_name}", logger):
                for eval_score, eval_func in self._score_eval_func_mapping.items():
                    dataset = add_score_to_dataset(
                        dataset=dataset,
                        eval_func=eval_func,
                        score_column_name=eval_score,
                        config=self._eval_algorithm_config,
                    )

                dataset_scores, category_scores = aggregate_evaluation_scores(
                    dataset, [METEOR_SCORE, ROUGE_SCORE, BERT_SCORE], agg_method=MEAN
                )
                eval_outputs.append(
                    EvalOutput(
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                        prompt_template=dataset_prompt_template,
                        dataset_scores=dataset_scores,
                        category_scores=category_scores,
                        output_path=self._eval_results_path,
                    )
                )
            if save:
                save_dataset(
                    dataset=dataset,
                    score_names=[METEOR_SCORE, ROUGE_SCORE, BERT_SCORE],
                    path=generate_output_dataset_path(
                        path_to_parent_dir=self._eval_results_path,
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                    ),
                )

        return eval_outputs


def get_meteor_score(target_output: str, model_output: str, config: SummarizationAccuracyConfig) -> float:
    """
    METEOR, an automatic metric for machine translation evaluation
    that is based on a generalized concept of unigram matching between the
    machine-produced translation and human-produced reference translations.
    Unigrams can be matched based on their surface forms, stemmed forms,
    and meanings; furthermore, METEOR can be easily extended to include more
    advanced matching strategies. Once all generalized unigram matches
    between the two strings have been found, METEOR computes a score for
    this matching using a combination of unigram-precision, unigram-recall, and
    a measure of fragmentation that is designed to directly capture how
    well-ordered the matched words in the machine translation are in relation
    to the reference.

    METEOR gets an R correlation value of 0.347 with human evaluation on the Arabic
    data and 0.331 on the Chinese data. This is shown to be an improvement on
    using simply unigram-precision, unigram-recall and their harmonic F1
    combination.

    :param target_output: The expected responses from the model
    :param model_output: The output of a model that we want to evaluate.
    :returns: meteor score
    """
    return meteor_score.single_meteor_score(
        reference=word_tokenize(target_output), hypothesis=word_tokenize(model_output)
    )


def get_rouge_score(target_output: str, model_output: str, config: SummarizationAccuracyConfig) -> float:
    """
    The ROUGE-N, where N=[1,2,L], score is a standard metric for summarization quality.
    It computes the word overlap between the reference and model summary. Given that this metric is based on simple
    word overlap statistics, it works best for extractive summaries.
    Note that if we rephrase the summary without changing its meaning the ROUGE-N score will drop.

    Reference: https://huggingface.co/spaces/evaluate-metric/rouge

    :param target_output: The expected responses from the model
    :param model_output: The output of a model that we want to evaluate.
    :param config: Eval algo config
    :returns: rouge score: boolean indicating using stemmer for rouge
    """
    rouge = hf_evaluate.load("rouge")
    return rouge.compute(
        predictions=[model_output],
        references=[target_output],
        use_stemmer=config.use_stemmer_for_rouge,
        rouge_types=[config.rouge_type],
    )[config.rouge_type]


def get_bert_score(target_output: str, model_output: str, config: SummarizationAccuracyConfig) -> float:
    """
    BERTscore is a similarity-based metric that compares the embedding of the prediction and target sentences
    under a (learned) model, typically, from the BERT family.
    This score may lead to increased flexibility compared to rouge and METEOR in terms of rephrasing since
    semantically similar sentences are (typically) embedded similarly.

    https://huggingface.co/spaces/evaluate-metric/bertscore

    :param target_output: The expected responses from the model
    :param model_output: The output of a model that we want to evaluate.
    :param config: Eval algo config
    :returns: rouge score: boolean indicating using stemmer for rouge
    """
    bertscore = BertscoreHelperModel(model_type=config.model_type_for_bertscore)
    return bertscore.get_helper_scores(target_output, model_output)


def add_score_to_dataset(
    dataset: Dataset, eval_func: Callable, score_column_name: str, config: SummarizationAccuracyConfig
):
    """
    Util method to add a score column to a ray dataset.

    :param dataset: ray Dataset to be used for eval score generation
    :param eval_func: eval function callable method
    :param score_column_name: column name for score to be added
    :param config: Eval algo config
    :returns: ray Dataset with score column
    """

    def _generate_eval_scores(df: pd.DataFrame) -> pd.Series:  # pragma: no cover
        """
        Map function generating the scores for every input record in input dataset
        """
        return pd.Series(
            data=[
                eval_func(row[TARGET_OUTPUT_COLUMN_NAME], row[MODEL_OUTPUT_COLUMN_NAME], config)
                for index, row in df.iterrows()
            ]
        )

    return dataset.add_column(score_column_name, _generate_eval_scores).materialize()
