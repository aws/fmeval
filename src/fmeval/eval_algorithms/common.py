import logging
from typing import List, Optional

from ray.data import Dataset

from fmeval.constants import EVAL_OUTPUT_RECORDS_BATCH_SIZE, MEAN, DatasetColumns
from fmeval.eval_algorithms import EvalOutput, get_default_prompt_template
from fmeval.eval_algorithms.save_strategy import SaveStrategy, FileSaveStrategy
from fmeval.eval_algorithms.util import (
    EvalOutputRecord,
    aggregate_evaluation_scores,
    validate_dataset,
    generate_output_dataset_path,
    create_model_invocation_pipeline,
)
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.perf_util import timed_block
from fmeval.transforms.transform_pipeline import TransformPipeline

logger = logging.getLogger(__name__)


def save_dataset(dataset: Dataset, score_names: List[str], save_strategy: SaveStrategy) -> None:  # pragma: no cover
    """
    Writes the dataset to a JSON Lines file, where each JSON Lines object
    is the JSON representation of an `EvalOutputRecord`.

    :param dataset: a Ray Dataset that is produced during the execution of
        an EvalAlgorithm's `evaluate` method. This dataset is expected
        to include columns for every score computed by the evaluation algorithm.
    :param score_names: the names of the score columns in `dataset`.
    :param save_strategy: the SaveStrategy to be used to save the outputs.


        Example Dataset:
         ________________________________________________
        | "model_input" | "aux" | "rouge" | "bert_score"|
        -------------------------------------------------
        |    "hello"    | 0.189 |   0.5   |     0.42    |
        -------------------------------------------------
        |    "world"    | 0.162 |  0.314  |    0.271    |
        -------------------------------------------------

        Note that the "aux" column name does not belong to constants.COLUMN_NAMES, meaning that this column
        won't get included in the saved outputs. See the docstring for EvalOutputRecord.from_row for more details.

        Corresponding Json Lines file contents:
        {"model_input" : "hello", "scores" : [{"name": "rouge", "value": 0.5}, {"name": "bert_score", "value": 0.42}]}
        {"model_input" : "world", "scores" : [{"name": "rouge", "value": 0.314}, {"name": "bert_score", "value": 0.271}]}


    """
    with timed_block(f"Saving dataset to file", logger):
        # We need the outer dict that wraps the EvalOutputRecord because map() requires
        # whatever is returned from the lambda function to be a dict
        dataset = dataset.map(lambda row: {"record": EvalOutputRecord.from_row(row, score_names)})
        # Without this line, dataset.iter_rows() below is not guaranteed to return the rows
        # in the same order that they appear in `dataset`.
        dataset.materialize()

        with save_strategy:
            for batch in dataset.iter_batches(batch_size=EVAL_OUTPUT_RECORDS_BATCH_SIZE):
                save_strategy.save(batch["record"])


def evaluate_dataset(
    dataset: Dataset,
    pipeline: TransformPipeline,
    dataset_name: str,
    eval_name: str,
    metric_names: List[str],
    eval_results_path: str,
    model: Optional[ModelRunner] = None,
    prompt_template: Optional[str] = None,
    agg_method: str = MEAN,
    save: bool = False,
    save_strategy: Optional[SaveStrategy] = None,
) -> EvalOutput:
    """Execute an evaluation algorithm's pipeline on a dataset.

    :param dataset: The dataset to be evaluated.
    :param pipeline: The evaluation algorithm's pipeline, to be executed on the dataset.
    :param dataset_name: The name of the dataset being evaluated. This is metadata that
        will be included in the returned EvalOutput object.
    :param eval_name: The name of the evaluation algorithm.
    :param metric_names: The names of the metrics that this evaluation algorithm computes.
        prior to performing any evaluation logic. This parameter is algorithm-specific.
    :param eval_results_path: A file containing evaluation results will be stored at this path.
    :param model: An instance of ModelRunner representing the model under evaluation.
        If this argument is None, model responses cannot be obtained. In such cases,
        the dataset configured by `dataset_config` should already contain a column for
        model outputs.
    :param prompt_template: A template used to generate prompts that are fed to the model.
        If set to None, a default value will be used. Note that if this argument is not None,
        `model` must also not be None.
    :param agg_method: The aggregation method to use when aggregating the computed metric values.
        Currently, only MEAN is supported.
    :param save: If set to true, prompt responses and scores will be saved to a file.
        The path that this file is stored at is configured by `eval_results_path`.

    :return: An EvalOutput object encapsulating the results of the evaluation.
    """
    if model:
        try:
            validate_dataset(dataset, [DatasetColumns.MODEL_INPUT.value.name])
        except EvalAlgorithmClientError:
            raise EvalAlgorithmClientError(
                "evaluate_dataset has been given a ModelRunner to obtain outputs from "
                "but the provided dataset does not contain a model input column."
            )
        prompt_template = get_default_prompt_template(dataset_name) if not prompt_template else prompt_template
        model_invocation_pipeline = create_model_invocation_pipeline(model, prompt_template)
        pipeline = TransformPipeline([model_invocation_pipeline, pipeline])
    else:
        if prompt_template:
            logger.warning(
                "A prompt template, but no corresponding model, was provided."
                "Model outputs from the dataset will be used, and this prompt template will be ignored."
            )
        try:
            validate_dataset(dataset, [DatasetColumns.MODEL_OUTPUT.value.name])
        except EvalAlgorithmClientError:
            raise EvalAlgorithmClientError(
                "evaluate_dataset has been given a dataset with no model output column "
                "and no ModelRunner to obtain outputs from. Please either provide a model "
                "or use a dataset that contains model outputs already."
            )

    with (timed_block(f"Computing score and aggregation on dataset {dataset_name}", logger)):
        dataset = pipeline.execute(dataset)
        dataset_scores, category_scores = aggregate_evaluation_scores(dataset, metric_names, agg_method=agg_method)

        output_path = generate_output_dataset_path(
            path_to_parent_dir=eval_results_path,
            eval_name=eval_name,
            dataset_name=dataset_name,
        )
        eval_output = EvalOutput(
            eval_name=eval_name,
            dataset_name=dataset_name,
            prompt_template=prompt_template,
            dataset_scores=dataset_scores,
            category_scores=category_scores,
            output_path=output_path,
        )

        if save:  # pragma: no branch
            save_dataset(
                dataset=dataset,
                score_names=metric_names,
                save_strategy=save_strategy if save_strategy else FileSaveStrategy(output_path),
            )

        return eval_output
