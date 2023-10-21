from typing import List, Optional, Any
import ray.data
from textwrap import shorten

from amazon_fmeval.eval_algorithms import EvalOutput, DATASET_CONFIGS, EvalAlgorithm, TREX
from amazon_fmeval.eval_algorithms.classification_accuracy import CLASSIFICATION_ACCURACY_SCORE
from amazon_fmeval.eval_algorithms.general_semantic_robustness import WER_SCORE
from amazon_fmeval.eval_algorithms.prompt_stereotyping import (
    SENT_LESS_LOG_PROB_COLUMN_NAME,
    SENT_MORE_LOG_PROB_COLUMN_NAME,
    LOG_PROBABILITY_DIFFERENCE,
)
from amazon_fmeval.constants import CATEGORY_COLUMN_NAME, COLUMN_NAMES
from amazon_fmeval.reporting.cells import MarkdownCell, BarPlotCell, TableCell, BoldCell, HeadingCell
from amazon_fmeval.reporting.constants import (
    LEFT,
    CATEGORY_BAR_COLOR,
    OVERALL_BAR_COLOR,
    NUM_SAMPLES_TO_DISPLAY_IN_TABLE,
    DATASET_SCORE_LABEL,
    SCORE_DESCRIPTIONS,
    DATASET_DETAILS,
    TABLE_DESCRIPTION,
    WER_TABLE_DESCRIPTION,
    STEREOTYPING_TABLE_DESCRIPTION,
    FACTUAL_KNOWLEDGE_TABLE_DESCRIPTION,
    TREX_DESCRIPTION_EXAMPLES,
    BUILT_IN_DATASET,
    CUSTOM_DATASET,
    AGGREGATE_ONLY_SCORES,
    MAX_CHAR,
    PROMPT_COLUMN_NAME,
    TOXICITY_EVAL_NAMES,
    TOXIGEN_NAME,
    DETOXIFY_NAME,
)
from amazon_fmeval.reporting.util import format_dataset_name, format_string


TABLE_COLUMNS = (
    list(set(COLUMN_NAMES))
    + [PROMPT_COLUMN_NAME]
    + list(set(SCORE_DESCRIPTIONS.keys()))
    + [SENT_LESS_LOG_PROB_COLUMN_NAME, SENT_MORE_LOG_PROB_COLUMN_NAME]
)


class CategoryBarPlotCell(BarPlotCell):
    """
    This class represents a bar plot that displays category-level and overall evaluation scores.
    """

    def __init__(
        self,
        categories: List[str],
        scores: List[float],
        score_name: str,
        dataset_score: float,
        height: Optional[str] = None,
        width: Optional[str] = None,
        center: bool = True,
    ):
        """
        :param categories: The names of the categories.
        :param scores: The values of the category scores.
        :param score_name: The name of the score that was computed in the evaluation.
        :param dataset_score: The overall score for the dataset.
        :param height: Height of the plot as a string
        :param width: Width the plot as a string
        :param center: Boolean indicating if the plot should be center aligned in the page
        """
        labels = categories + [DATASET_SCORE_LABEL]
        heights = scores + [dataset_score]
        super().__init__(
            labels=labels,
            heights=heights,
            color=CategoryBarPlotCell._create_bar_plot_colors(labels),
            title=CategoryBarPlotCell._create_bar_plot_title(score_name),
            plot_height=height,
            plot_width=width,
            center=center,
        )

    @staticmethod
    def _create_bar_plot_colors(category_names: List[str]) -> List[str]:
        """
        Returns a list of colors corresponding to the bars for each of the categories.

        :param category_names: Includes "Overall" as the last category name
        :returns: A list of colors, where the kth element is the color
            of the bar corresponding to category_names[k]
        """
        return [CATEGORY_BAR_COLOR for _ in range(len(category_names) - 1)] + [OVERALL_BAR_COLOR]

    @staticmethod
    def _create_bar_plot_title(evaluation_type: str) -> str:
        """
        Generates a bar plot title from the evaluation type.

        :param evaluation_type: Ex - "Stereotyping"
        :returns: A title to be used in the bar plot for category scores
        """
        return format_string(f"{evaluation_type}", as_title=True, as_score=True, as_plot_title=True)


class RayDatasetTableCell(TableCell):
    """
    This class represents a table that displays data from a Ray Dataset object.
    """

    def __init__(
        self,
        dataset: ray.data.Dataset,
        col_to_sort: Optional[str] = None,
        k: Optional[int] = None,
        descending: bool = False,
        abs_val: bool = False,
        caption: Optional[str] = None,
        cell_align: str = LEFT,
    ):
        """
        :param dataset: The Ray Dataset that we create a TableCell out of
        :param col_to_sort: The name of the column in the dataset to sort by
        :param k: The number of samples from the dataset to display in the table
        :param descending: Whether to sort in descending order.
        :param abs_val: Whether to sort by absolute value when sorting is enabled.
        :param caption: The caption text before the table.
        :param cell_align: The text alignment within cells.
        """
        if col_to_sort:
            assert (
                col_to_sort in dataset.columns()
            ), f"Column to be sorted `{col_to_sort}` is not present in dataset columns: {dataset.columns()}"
            if abs_val:
                pd_dataset = dataset.to_pandas()
                pd_dataset = pd_dataset.sort_values(by=col_to_sort, key=abs, ascending=not descending)
                dataset = ray.data.from_pandas(pd_dataset)
            else:
                dataset = dataset.sort(col_to_sort, descending=descending)
        samples = dataset.take(k) if k else dataset.take_all()  # take() uses min(k, num samples in dataset)
        table_data = [RayDatasetTableCell.truncate_samples(list(sample.values())) for sample in samples]
        headers = dataset.columns()
        if CATEGORY_COLUMN_NAME in headers:  # pragma: no branch
            category_idx = headers.index(CATEGORY_COLUMN_NAME)
            table_data = [[row[category_idx]] + row[:category_idx] + row[category_idx + 1 :] for row in table_data]
            headers = [headers[category_idx]] + headers[:category_idx] + headers[category_idx + 1 :]
        headers = [format_string(header, as_column_name=True, as_title=True) for header in headers]
        super().__init__(data=table_data, headers=headers, cell_align=cell_align, caption=caption)

    @staticmethod
    def truncate_samples(samples: List[Any]) -> List[Any]:
        """
        :param samples: List of items representing one row in the table.
        :return: Table row with strings longer than MAX_CHAR truncated.
        """
        truncated_samples = [
            shorten(sample, MAX_CHAR) if isinstance(sample, str) and len(sample) > MAX_CHAR else sample
            for sample in samples
        ]
        return truncated_samples


class CategoryScoreCell(MarkdownCell):
    """
    This class displays a bar plot for the different category scores from an evaluation, and outlines the lowest
        scoring category.
    """

    def __init__(self, categories: List[str], scores: List[float], score_name: str, dataset_score: float, n: int = 10):
        """
        :param categories: The names of the categories.
        :param scores: The values of the category scores.
        :param score_name: The name of the score that was computed in the evaluation.
        :param dataset_score: The overall score for the dataset.
        :param n: Max number of categories to display.
        """

        note = (
            f"The top {n} categories are displayed here. To view the remaining category scores, see the `output.json` file at your S3 output location."
            if len(categories) > n
            else ""
        )
        sorted_scores, sorted_categories = (list(l) for l in zip(*sorted(zip(scores, categories))))
        bar_plot = CategoryBarPlotCell(
            sorted_categories[:n], sorted_scores[:n], score_name, dataset_score, height="70%", width="70%"
        )
        lowest_category = CategoryScoreCell._get_kth_category_score(categories, scores, k=0)
        super().__init__(
            f"The plot shows the score breakdown into individual categories.",
            note,
            bar_plot,
            f"The model scores lowest in the category **{lowest_category}**. ",
        )

    @staticmethod
    def _get_kth_category_score(categories: List[str], scores: List[float], k: int = 0, reverse: bool = False) -> str:
        """
        Sorts `category_scores` by their `score` attribute and returns the kth element in the sorted list.

        :param categories: The names of the categories.
        :param scores: The values of the category scores.
        :param k: The index of the CategoryScore to return
        :param reverse: Whether to sort in descending order
        """
        assert 0 <= k < len(categories), "The provided `k` argument is outside of the valid range"
        sorted_categories = [cat for score, cat in sorted(zip(scores, categories), reverse=reverse)]
        return sorted_categories[k]


class ScoreTableCell(MarkdownCell):
    """
    This class generates two tables displaying the highest and lowest-scoring examples from a particular score.
    """

    def __init__(self, dataset: ray.data.Dataset, score_column_name: str, binary: Optional[bool] = False):
        """
        :param dataset: The Ray Dataset used in the evaluation task.
        :param score_column_name: The name of the score column in the dataset.
        :param binary: Boolean indicating if the score is binary.
        """
        description = (
            WER_TABLE_DESCRIPTION
            if score_column_name == WER_SCORE
            else STEREOTYPING_TABLE_DESCRIPTION
            if score_column_name == LOG_PROBABILITY_DIFFERENCE
            else FACTUAL_KNOWLEDGE_TABLE_DESCRIPTION
            if binary
            else TABLE_DESCRIPTION
        )

        n_samples = min(NUM_SAMPLES_TO_DISPLAY_IN_TABLE, dataset.count())
        top_description = (
            (f"Top {n_samples} most stereotypical examples:")
            if score_column_name == LOG_PROBABILITY_DIFFERENCE
            else f"{n_samples} correct examples:"
            if binary
            else f"Top {n_samples} highest-scoring examples:"
        )
        bottom_description = (
            (f"Top {n_samples} least stereotypical examples:")
            if score_column_name == LOG_PROBABILITY_DIFFERENCE
            else f"{n_samples} incorrect examples:"
            if binary
            else f"Bottom {n_samples} lowest-scoring examples:"
        )
        abs_val = True if score_column_name == LOG_PROBABILITY_DIFFERENCE else False

        cells = [
            MarkdownCell(description),
            RayDatasetTableCell(
                dataset,
                score_column_name,
                k=n_samples,
                descending=True,
                abs_val=abs_val,
                caption=top_description,
            ),
            RayDatasetTableCell(
                dataset,
                score_column_name,
                k=n_samples,
                descending=False,
                abs_val=abs_val,
                caption=bottom_description,
            ),
        ]
        super().__init__(*cells)


class ScoreCell(MarkdownCell):
    """
    This class generates visualizations for an evaluation score, including the overall dataset score, a bar plot
        displaying category-level scores if provided, and tables displaying highest and lowest scoring examples.
    """

    def __init__(
        self,
        dataset: Optional[ray.data.Dataset],
        score_name: str,
        score_column_name: str,
        dataset_score: float,
        categories: Optional[List[str]],
        category_scores: Optional[List[float]],
    ):
        """
        :param dataset: The Ray Dataset used in the evaluation task.
        :param score_name: The name of the score that was computed in the evaluation.
        :param score_column_name: The name of the score column in the dataset.
        :param dataset_score: The aggregated score computed across the whole dataset.
        :param categories: The names of the categories.
        :param category_scores: The values of the category scores.
        """
        cells = [
            HeadingCell(text=f"{format_string(score_name, as_title=True, as_score=True)}", level=5),
            MarkdownCell(SCORE_DESCRIPTIONS[score_name]),
            BoldCell(f"Overall Score: {dataset_score}"),
        ]
        if categories and category_scores:  # pragma: no branch
            cells.append(CategoryScoreCell(categories, category_scores, score_name, dataset_score))
        if dataset:  # pragma: no cover
            columns = [i for i in TABLE_COLUMNS if i != "target_output"] if score_name == WER_SCORE else TABLE_COLUMNS
            present_columns = [col for col in dataset.columns() if col in columns]
            dataset = dataset.select_columns(present_columns)
            is_binary_score = (
                True if score_name in [EvalAlgorithm.FACTUAL_KNOWLEDGE.value, CLASSIFICATION_ACCURACY_SCORE] else False
            )
            cells.append(ScoreTableCell(dataset, score_column_name, binary=is_binary_score))
        super().__init__(*cells)


class EvalOutputCell(MarkdownCell):
    def __init__(
        self,
        eval_output: EvalOutput,
        dataset: Optional[ray.data.Dataset] = None,
        score_column_names: Optional[dict] = None,
    ):
        """
        :param eval_output: A EvalOutput object from an evaluation.
        :param dataset: The Ray dataset containing the evaluation scores.
        :param score_column_names: A dict mapping the score names and score column names for the evaluation.
        """
        dataset_type = BUILT_IN_DATASET if eval_output.dataset_name in DATASET_CONFIGS else CUSTOM_DATASET
        dataset_description = EvalOutputCell.get_dataset_description(
            dataset_name=eval_output.dataset_name,
            dataset_type=dataset_type,
            dataset=dataset,
            eval_name=eval_output.eval_name,
        )
        toxicity_detector_name = (
            f"Toxicity detector model: {DETOXIFY_NAME}"
            if eval_output.eval_name in TOXICITY_EVAL_NAMES and len(eval_output.dataset_scores) > 1
            else f"Toxicity detector model: {TOXIGEN_NAME}"
            if eval_output.eval_name in TOXICITY_EVAL_NAMES and len(eval_output.dataset_scores) == 1
            else ""
        )

        eval_cells = [
            HeadingCell(f"{dataset_type}: {format_dataset_name(eval_output.dataset_name, hyperlink=True)}", level=4),
            MarkdownCell(dataset_description),
            MarkdownCell(toxicity_detector_name),
        ]
        if eval_output.error:
            error_cell = BoldCell(f"This evaluation failed with the error message: {eval_output.error}")
            eval_cells.append(error_cell)
        else:
            dataset_scores = {dataset_score.name: dataset_score.value for dataset_score in eval_output.dataset_scores}
            for score_name, dataset_score_value in dataset_scores.items():
                categories = (
                    {
                        category_score.name: score.value
                        for category_score in eval_output.category_scores
                        for score in category_score.scores
                        if score.name == score_name
                    }
                    if eval_output.category_scores
                    else None
                )
                score_column_name = (
                    LOG_PROBABILITY_DIFFERENCE if score_name == EvalAlgorithm.PROMPT_STEREOTYPING.value else score_name
                )
                if score_name not in AGGREGATE_ONLY_SCORES:  # pragma: no branch
                    score_cell = ScoreCell(
                        dataset=dataset,
                        score_name=score_name,
                        score_column_name=score_column_name,
                        dataset_score=dataset_score_value,
                        categories=list(categories.keys()) if categories else None,
                        category_scores=list(categories.values()) if categories else None,
                    )
                    eval_cells.append(score_cell)

        super().__init__(*eval_cells)

    @staticmethod
    def get_dataset_sampling_description(dataset_name: str, dataset: ray.data.Dataset) -> str:
        """
        :param dataset_name: The name of the Ray dataset.
        :param dataset: The Ray dataset containing the evaluation scores.
        :return: String describing the number of samples used in the evaluation.
        """
        num_records = dataset.count()
        total_records = DATASET_DETAILS[dataset_name].size if dataset_name in DATASET_DETAILS else num_records

        return f"We sampled {num_records} records out of {total_records} in the full dataset."

    @staticmethod
    def get_dataset_description(
        dataset_name: str, dataset_type: str, dataset: Optional[ray.data.Dataset], eval_name: Optional[str] = None
    ) -> str:
        """
        :param dataset_name: The name of the Ray dataset.
        :param dataset_type: Whether the dataset is a built-in or custom dataset.
        :param dataset: The Ray dataset containing the evaluation scores.
        :param eval_name: The name of the selected evaluation.
        :return: The description of the dataset, including the number of samples used in the evaluation.
        """

        dataset_sampling_description = (
            EvalOutputCell.get_dataset_sampling_description(dataset_name, dataset) if dataset else ""
        )
        if dataset_type == CUSTOM_DATASET:
            return dataset_sampling_description
        else:
            dataset_description = (
                DATASET_DETAILS[dataset_name].description + TREX_DESCRIPTION_EXAMPLES
                if dataset_name == TREX and eval_name == EvalAlgorithm.FACTUAL_KNOWLEDGE.value
                else DATASET_DETAILS[dataset_name].description
            )
            return dataset_description + " " + dataset_sampling_description
