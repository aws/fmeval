from amazon_fmeval.eval_algorithms.eval_algorithm import EvalOutput
from amazon_fmeval.reporting.cells import MarkdownCell, BarPlotCell, TableCell, BoldCell, HeadingCell
from amazon_fmeval.reporting.constants import (
    CATEGORY_BAR_COLOR,
    OVERALL_BAR_COLOR,
    NUM_SAMPLES_TO_DISPLAY_IN_TABLE,
    DATASET_SCORE_LABEL,
)
from typing import List, Optional
import ray.data


class CategoryBarPlotCell(BarPlotCell):
    """
    This class represents a bar plot that displays category-level and overall evaluation scores.
    """

    def __init__(self, categories: List[str], scores: List[float], score_name: str, dataset_score: float):
        """
        :param categories: The names of the categories.
        :param scores: The values of the category scores.
        :param score_name: The name of the score that was computed in the evaluation.
        :param dataset_score: The overall score for the dataset.
        """
        labels = categories + [DATASET_SCORE_LABEL]
        heights = scores + [dataset_score]
        super().__init__(
            labels=labels,
            heights=heights,
            color=CategoryBarPlotCell._create_bar_plot_colors(labels),
            title=CategoryBarPlotCell._create_bar_plot_title(score_name),
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
        return f"{evaluation_type} Scores"


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
    ):
        """
        :param dataset: The Ray Dataset that we create a TableCell out of
        :param col_to_sort: The name of the column in the dataset to sort by
        :param k: The number of samples from the dataset to display in the table
        :param descending: Whether to sort in descending order.
        """
        if col_to_sort:
            assert (
                col_to_sort in dataset.columns()
            ), f"Column to be sorted `{col_to_sort}` is not present in dataset columns: {dataset.columns()}"
            dataset = dataset.sort(col_to_sort, descending=descending)
        samples = dataset.take(k) if k else dataset.take_all()  # take() uses min(k, num samples in dataset)
        table_data = [list(sample.values()) for sample in samples]  # convert list of dicts to list of lists
        super().__init__(data=table_data, headers=dataset.columns())


class CategoryScoreCell(MarkdownCell):
    """
    This class displays a bar plot for the different category scores from an evaluation, and outlines the lowest
        scoring category.
    """

    def __init__(self, categories: List[str], scores: List[float], score_name: str, dataset_score: float):
        """
        :param categories: The names of the categories.
        :param scores: The values of the category scores.
        :param score_name: The name of the score that was computed in the evaluation.
        :param dataset_score: The overall score for the dataset.
        """
        bar_plot = CategoryBarPlotCell(categories, scores, score_name, dataset_score)
        lowest_category = CategoryScoreCell._get_kth_category_score(categories, scores, k=0)
        super().__init__(
            f"Score breakdown per {score_name} evaluation category:",
            bar_plot,
            f"The model scores lowest in this category: **{lowest_category}**. ",
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

    def __init__(self, dataset: ray.data.Dataset, score_column_name: str):
        """
        :param dataset: The Ray Dataset used in the evaluation task.
        :param score_column_name: The name of the score column in the dataset.
        """
        super().__init__(
            MarkdownCell("Below are a few examples of the highest and lowest-scoring examples across all categories: "),
            BoldCell(f"Top {NUM_SAMPLES_TO_DISPLAY_IN_TABLE} highest-scoring examples:"),
            RayDatasetTableCell(dataset, score_column_name, k=NUM_SAMPLES_TO_DISPLAY_IN_TABLE, descending=True),
            BoldCell(f"Bottom {NUM_SAMPLES_TO_DISPLAY_IN_TABLE} lowest-scoring examples:"),
            RayDatasetTableCell(dataset, score_column_name, k=NUM_SAMPLES_TO_DISPLAY_IN_TABLE, descending=False),
        )


class ScoreCell(MarkdownCell):
    """
    This class generates visualizations for an evaluation score, including the overall dataset score, a bar plot
        displaying category-level scores if provided, and tables displaying highest and lowest scoring examples.
    """

    def __init__(
        self,
        dataset: ray.data.Dataset,
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
            HeadingCell(text=f"{score_name.capitalize()}", level=3),
            BoldCell(f"Overall Score: {dataset_score}"),
        ]
        if categories and category_scores:  # pragma: no branch
            cells.append(CategoryScoreCell(categories, category_scores, score_name, dataset_score))
        cells.append(ScoreTableCell(dataset, score_column_name))
        super().__init__(*cells)


class EvalOutputCell(MarkdownCell):
    def __init__(self, eval_output: EvalOutput, dataset: ray.data.Dataset, score_column_names: dict):
        """
        :param eval_output: A EvalOutput object from an evaluation.
        :param dataset: The Ray dataset containing the evaluation scores.
        :param score_column_names: A dict mapping the score names and score column names for the evaluation.
        """
        eval_heading = HeadingCell(text=f"{eval_output.eval_name.capitalize()}", level=2)
        dataset_name_cell = BoldCell(f"Dataset: {eval_output.dataset_name.capitalize()}")
        dataset_scores = {dataset_score.name: dataset_score.value for dataset_score in eval_output.dataset_scores}
        score_cells = []
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
            score_cell = ScoreCell(
                dataset=dataset,
                score_name=score_name,
                score_column_name=score_column_names[score_name],
                dataset_score=dataset_score_value,
                categories=list(categories.keys()) if categories else None,
                category_scores=list(categories.values()) if categories else None,
            )
            score_cells.append(score_cell)
        super().__init__(eval_heading, dataset_name_cell, *score_cells)
