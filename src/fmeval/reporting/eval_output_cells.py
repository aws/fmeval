from typing import List, Optional, Any
import ray.data
from textwrap import shorten
import numpy as np
from fmeval.eval_algorithms import (
    EvalOutput,
    DATASET_CONFIGS,
    EvalAlgorithm,
    TREX,
    CROWS_PAIRS,
    get_default_prompt_template,
)
from fmeval.eval_algorithms.classification_accuracy import CLASSIFICATION_ACCURACY_SCORE
from fmeval.eval_algorithms.factual_knowledge import FACTUAL_KNOWLEDGE, FACTUAL_KNOWLEDGE_QUASI_EXACT
from fmeval.eval_algorithms.general_semantic_robustness import WER_SCORE
from fmeval.eval_algorithms.prompt_stereotyping import PROMPT_STEREOTYPING
from fmeval.constants import DatasetColumns, DATASET_COLUMNS
from fmeval.reporting.cells import MarkdownCell, BarPlotCell, TableCell, BoldCell, HeadingCell
from fmeval.reporting.constants import (
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
    TOXICITY_EVAL_NAMES,
    TOXIGEN_NAME,
    DETOXIFY_NAME,
    CROWS_PAIRS_DISCLAIMER,
    PROBABILITY_RATIO,
    IS_BIASED,
    ACCURACY_SEMANTIC_ROBUSTNESS_SCORES,
    ACCURACY_SEMANTIC_ROBUSTNESS_ALGOS,
    DETOXIFY_URI,
    TOXIGEN_URI,
)
from fmeval.reporting.util import format_dataset_name, format_string, add_hyperlink
from html import escape

TABLE_COLUMNS = list(set(DATASET_COLUMNS)) + list(set(SCORE_DESCRIPTIONS.keys())) + [PROBABILITY_RATIO, IS_BIASED]


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
        origin: float = 0,
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
            origin=origin,
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
        if DatasetColumns.CATEGORY.value.name in headers:  # pragma: no branch
            category_idx = headers.index(DatasetColumns.CATEGORY.value.name)
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
            shorten(sample, MAX_CHAR)
            if isinstance(sample, str) and len(sample) > MAX_CHAR
            else np.round(sample, decimals=6)
            if isinstance(sample, float)
            else sample
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
        sorted_scores, sorted_categories = (list(l) for l in zip(*sorted(zip(scores, categories), reverse=True)))
        bar_plot_origin = 0.5 if score_name == PROMPT_STEREOTYPING else 0
        bar_plot = CategoryBarPlotCell(
            sorted_categories[:n],
            sorted_scores[:n],
            score_name,
            dataset_score,
            height="70%",
            width="70%",
            origin=bar_plot_origin,
        )

        lowest_category = (
            CategoryScoreCell._get_kth_category_score(categories, scores, reverse=True, origin=0.5, k=0)
            if score_name == PROMPT_STEREOTYPING
            else CategoryScoreCell._get_kth_category_score(categories, scores, k=0)
        )
        lowest_category = escape(lowest_category)
        lowest_score_description = (
            "The model stereotypes the most in the category"
            if score_name == PROMPT_STEREOTYPING
            else "The model scores lowest in the category"
        )
        super().__init__(
            f"The plot shows the score breakdown into individual categories.",
            note,
            bar_plot,
            f"{lowest_score_description} **{lowest_category}**. ",
        )

    @staticmethod
    def _get_kth_category_score(
        categories: List[str], scores: List[float], k: int = 0, reverse: bool = False, origin: float = 0
    ) -> str:
        """
        Sorts `category_scores` by their `score` attribute and returns the kth element in the sorted list.

        :param categories: The names of the categories.
        :param scores: The values of the category scores.
        :param k: The index of the CategoryScore to return
        :param reverse: Whether to sort in descending order
        :param origin: The origin of the score values.
        """
        assert 0 <= k < len(categories), "The provided `k` argument is outside of the valid range"
        scores = [abs(score - origin) for score in scores] if origin != 0 else scores
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
            if score_column_name == PROBABILITY_RATIO
            else FACTUAL_KNOWLEDGE_TABLE_DESCRIPTION
            if binary
            else TABLE_DESCRIPTION
        )

        n_samples = min(NUM_SAMPLES_TO_DISPLAY_IN_TABLE, dataset.count())
        top_description = (
            (f"Top {n_samples} most stereotypical examples:")
            if score_column_name == PROBABILITY_RATIO
            else f"{n_samples} correct examples:"
            if binary
            else f"Top {n_samples} examples with highest scores:"
        )
        bottom_description = (
            (f"Top {n_samples} least stereotypical examples:")
            if score_column_name == PROBABILITY_RATIO
            else f"{n_samples} incorrect examples:"
            if binary
            else f"Bottom {n_samples} examples with lowest scores:"
        )
        abs_val = True if score_column_name == PROBABILITY_RATIO else False

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
        score_name_display = (
            format_string(score_name, as_title=True)
            if score_name == WER_SCORE
            else format_string(score_name, as_title=True, as_score=True)
        )
        cells = [
            HeadingCell(text=score_name_display, level=5),
            MarkdownCell(SCORE_DESCRIPTIONS[score_name]),
            BoldCell(f"Average Score: {dataset_score}"),
        ]
        if categories and category_scores:  # pragma: no branch
            cells.append(CategoryScoreCell(categories, category_scores, score_name, dataset_score))
        if dataset:  # pragma: no cover
            columns = [i for i in TABLE_COLUMNS if i != "target_output"] if score_name == WER_SCORE else TABLE_COLUMNS
            present_columns = [col for col in dataset.columns() if col in columns]
            dataset = dataset.select_columns(present_columns)
            is_binary_score = (
                True
                if score_name in [FACTUAL_KNOWLEDGE, FACTUAL_KNOWLEDGE_QUASI_EXACT, CLASSIFICATION_ACCURACY_SCORE]
                else False
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
        prompt_template = EvalOutputCell.format_prompt_template(
            dataset_type, eval_output.dataset_name, eval_output.prompt_template
        )
        toxicity_detector_name = (
            f"**Toxicity detector model**: {add_hyperlink(DETOXIFY_NAME, DETOXIFY_URI)}"
            if eval_output.eval_name in TOXICITY_EVAL_NAMES and len(eval_output.dataset_scores) > 1
            else f"**Toxicity detector model**: {add_hyperlink(TOXIGEN_NAME, TOXIGEN_URI)}"
            if eval_output.eval_name in TOXICITY_EVAL_NAMES and len(eval_output.dataset_scores) == 1
            else ""
        )

        eval_cells = [
            HeadingCell(f"{dataset_type}: {format_dataset_name(eval_output.dataset_name, hyperlink=True)}", level=4),
            MarkdownCell(dataset_description),
            MarkdownCell(prompt_template),
            MarkdownCell(toxicity_detector_name),
        ]
        if eval_output.error:
            error_cell = BoldCell(f"This evaluation failed with the error message: {eval_output.error}")
            eval_cells.append(error_cell)
        else:
            dataset_scores = {dataset_score.name: dataset_score.value for dataset_score in eval_output.dataset_scores}
            for score_name, dataset_score_value in dataset_scores.items():  # pragma: no cover
                if (
                    eval_output.eval_name in ACCURACY_SEMANTIC_ROBUSTNESS_ALGOS
                    and score_name in ACCURACY_SEMANTIC_ROBUSTNESS_SCORES
                ):
                    continue
                else:
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
                        PROBABILITY_RATIO if score_name == EvalAlgorithm.PROMPT_STEREOTYPING.value else score_name
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
                DATASET_DETAILS[dataset_name].description + TREX_DESCRIPTION_EXAMPLES + dataset_sampling_description
                if dataset_name == TREX and eval_name == EvalAlgorithm.FACTUAL_KNOWLEDGE.value
                else DATASET_DETAILS[dataset_name].description
                + dataset_sampling_description
                + "\n\n"
                + CROWS_PAIRS_DISCLAIMER
                if dataset_name == CROWS_PAIRS
                else DATASET_DETAILS[dataset_name].description + " " + dataset_sampling_description
            )
            return dataset_description

    @staticmethod
    def format_prompt_template(dataset_type: str, dataset_name: str, prompt_template: Optional[str] = None) -> str:
        """
        :param dataset_type: string indicating if dataset is a built-in or custom dataset.
        :param dataset_name: the name of the dataset.
        :param prompt_template: optional prompt template used in the evaluation.
        :return: prompt template string formatted for the report.
        """
        prompt_template_str = "**Prompt Template:** "
        if prompt_template:
            return prompt_template_str + escape(prompt_template)
        elif dataset_type == BUILT_IN_DATASET:
            return prompt_template_str + get_default_prompt_template(dataset_name)
        else:
            return prompt_template_str + "No prompt template was provided for this dataset."
