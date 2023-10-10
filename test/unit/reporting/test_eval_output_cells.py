from unittest.mock import patch, Mock, call

from amazon_fmeval.eval_algorithms import EvalOutput, EvalScore, CategoryScore
from amazon_fmeval.reporting.cells import BarPlotCell, TableCell
from amazon_fmeval.reporting.eval_output_cells import (
    CategoryBarPlotCell,
    RayDatasetTableCell,
    CategoryScoreCell,
    ScoreTableCell,
    ScoreCell,
    EvalOutputCell,
)
from amazon_fmeval.reporting.constants import (
    CATEGORY_BAR_COLOR,
    OVERALL_BAR_COLOR,
    NUM_SAMPLES_TO_DISPLAY_IN_TABLE,
    DATASET_SCORE_LABEL,
)
from typing import Any, Dict, List, NamedTuple
import ray.data
import pytest


class TestEvalOutputCells:
    def test_category_bar_plot_cell_init_success(self):
        """
        GIVEN valid categories, scores and score_name
        WHEN an CategoryBarPlotCell is created
        THEN the created CategoryBarPlotCell is a BarPlotCell with attributes that are correctly extracted from
            the input.
        """
        categories = ["nationality", "religion", "age"]
        scores = [0.314, 0.271, 0.888]
        score_name = "Stereotyping"
        dataset_score = 0.6

        cat_bar_plot = CategoryBarPlotCell(categories, scores, score_name, dataset_score)
        manually_created_bar_plot = BarPlotCell(
            labels=categories + [DATASET_SCORE_LABEL],
            heights=scores + [dataset_score],
            color=[CATEGORY_BAR_COLOR, CATEGORY_BAR_COLOR, CATEGORY_BAR_COLOR, OVERALL_BAR_COLOR],
            title="Stereotyping Scores",
        )
        assert str(cat_bar_plot) == str(manually_created_bar_plot)

    class TestCaseRayDatasetTableCell(NamedTuple):
        items: List[Dict[str, Any]]
        kwargs: Dict[str, Any]
        expected_table: TableCell

    @pytest.mark.parametrize(
        "items, kwargs, expected_table",
        [
            TestCaseRayDatasetTableCell(
                items=[
                    {"Name": "b", "Age": 2},
                    {"Name": "a", "Age": 1},
                    {"Name": "c", "Age": 3},
                ],
                kwargs={},
                expected_table=TableCell(data=[["b", 2], ["a", 1], ["c", 3]], headers=["Name", "Age"]),
            ),
            TestCaseRayDatasetTableCell(
                items=[
                    {"Name": "b", "Age": 2},
                    {"Name": "a", "Age": 1},
                    {"Name": "c", "Age": 3},
                ],
                kwargs={"col_to_sort": "Age"},
                expected_table=TableCell(data=[["a", 1], ["b", 2], ["c", 3]], headers=["Name", "Age"]),
            ),
            TestCaseRayDatasetTableCell(
                items=[
                    {"Name": "b", "Age": 2},
                    {"Name": "a", "Age": 1},
                    {"Name": "c", "Age": 3},
                    {"Name": "d", "Age": 4},
                ],
                kwargs={"k": 3},
                expected_table=TableCell(data=[["b", 2], ["a", 1], ["c", 3]], headers=["Name", "Age"]),
            ),
            TestCaseRayDatasetTableCell(
                items=[
                    {"Name": "a", "Age": 1},
                    {"Name": "b", "Age": 2},
                    {"Name": "c", "Age": 3},
                    {"Name": "d", "Age": 4},
                    {"Name": "e", "Age": 5},
                ],
                kwargs={"col_to_sort": "Age", "k": 3, "descending": True},
                expected_table=TableCell(data=[["e", 5], ["d", 4], ["c", 3]], headers=["Name", "Age"]),
            ),
        ],
    )
    def test_ray_dataset_table_cell_init_success(self, items, kwargs, expected_table):
        """
        GIVEN valid arguments to the RayDatasetTable initializer
        WHEN a RayDatasetTable is created
        THEN the RayDatasetTable is a TableCell containing data
            that is properly extracted from the Ray Dataset
        """
        dataset = ray.data.from_items(items)
        table = RayDatasetTableCell(dataset=dataset, **kwargs)
        assert str(table) == str(expected_table)

    def test_ray_dataset_table_cell_init_failure(self):
        """
        GIVEN a valid Ray Dataset and an invalid `col_to_sort` argument
        WHEN a RayDatasetTable is created
        THEN an AssertionError is raised
        """
        dataset = ray.data.from_items([{"Name": "a", "Age": 1}])
        with pytest.raises(AssertionError, match=r"Column to be sorted `Invalid column` is not present"):
            RayDatasetTableCell(dataset=dataset, col_to_sort="Invalid column")

    class TestCaseKthCategoryScore(NamedTuple):
        k: int
        reverse: bool
        expected: str

    @pytest.mark.parametrize(
        "k, reverse, expected",
        [
            TestCaseKthCategoryScore(k=0, reverse=True, expected="age"),
            TestCaseKthCategoryScore(k=2, reverse=True, expected="religion"),
            TestCaseKthCategoryScore(k=0, reverse=False, expected="religion"),
            TestCaseKthCategoryScore(k=2, reverse=False, expected="age"),
        ],
    )
    def test_get_kth_category_score(self, k, reverse, expected):
        """
        GIVEN valid arguments
        WHEN _get_kth_category_score is called
        THEN the correct value is returned
        """
        categories = ["nationality", "religion", "age"]
        scores = [0.314, 0.271, 0.888]

        actual = CategoryScoreCell._get_kth_category_score(categories, scores, k, reverse)
        assert actual == expected

    @pytest.mark.parametrize("k", [-1, 3])
    def test_get_kth_category_score_invalid_k(self, k):
        """
        GIVEN an invalid `k` argument
        WHEN _get_kth_category_score is called
        THEN an exception is raised
        """
        categories = ["nationality", "religion", "age"]
        scores = [0.314, 0.271, 0.888]
        with pytest.raises(AssertionError, match=r"The provided `k` argument is outside of the valid range"):
            CategoryScoreCell._get_kth_category_score(categories, scores, k=k)

    def test_category_score_cell(self):
        """
        GIVEN valid categories and scores
        WHEN a CategoryScoreCell is created
        THEN the lowest-scoring category is correctly computed and the string representation of the CategoryScoreCell
            matches what is expected
        """
        categories = ["nationality", "religion", "age"]
        scores = [0.314, 0.271, 0.888]
        dataset_score = 0.5
        with patch("amazon_fmeval.reporting.eval_output_cells.CategoryBarPlotCell", return_value="AggBPCell"):

            cell = CategoryScoreCell(categories, scores, "prompt stereotyping", dataset_score)
            assert (
                str(cell) == f"Score breakdown per prompt stereotyping evaluation category:  \n\n"
                "AggBPCell  \n\n"
                "The model scores lowest in this category: **religion**. "
            )

    @patch("amazon_fmeval.reporting.eval_output_cells.RayDatasetTableCell", return_value="RayTable")
    def test_score_table_cell(self, mock_ray_table):
        """
        GIVEN a valid dataset and score column name
        WHEN a ScoreTableCell is created
        THEN the string representation of the ScoreTableCell matches what is expected,
            and the included RayDatasetTableCells that make up the ScoreTableCell
            are initialized correctly
        """
        dataset = Mock()
        score_column_name = ""
        cell = ScoreTableCell(dataset, score_column_name)

        # Assert structure of returned MarkdownCell is correct
        assert (
            str(cell)
            == f"Below are a few examples of the highest and lowest-scoring examples across all categories:   "
            f"\n\n**Top {NUM_SAMPLES_TO_DISPLAY_IN_TABLE} highest-scoring examples:**  \n\nRayTable  "
            f"\n\n**Bottom {NUM_SAMPLES_TO_DISPLAY_IN_TABLE} lowest-scoring examples:**  \n\nRayTable"
        )

        # Assert RayDatasetTableCells are created with the right arguments
        mock_ray_table.assert_has_calls(
            [
                call(
                    dataset,
                    score_column_name,
                    k=NUM_SAMPLES_TO_DISPLAY_IN_TABLE,
                    descending=True,
                ),
                call(
                    dataset,
                    score_column_name,
                    k=NUM_SAMPLES_TO_DISPLAY_IN_TABLE,
                    descending=False,
                ),
            ]
        )

    def test_score_cell(self):
        """
        GIVEN a valid
        WHEN a ScoreCell is created
        THEN the string representation of the ScoreCell matches what is expected
        """
        with patch("amazon_fmeval.reporting.eval_output_cells.CategoryScoreCell", return_value="category_score"), patch(
            "amazon_fmeval.reporting.eval_output_cells.ScoreTableCell", return_value="table"
        ):
            # GIVEN

            dataset = Mock()
            score_name = "Stereotyping"
            score_column_name = "stereotyping"
            dataset_score = 0.88
            categories = ["nationality", "religion", "age"]
            category_scores = [0.314, 0.271, 0.888]

            # WHEN
            cell = ScoreCell(
                dataset=dataset,
                score_name=score_name,
                score_column_name=score_column_name,
                dataset_score=dataset_score,
                categories=categories,
                category_scores=category_scores,
            )

            # THEN
            assert str(cell) == (
                "### Stereotyping  \n\n" "**Overall Score: 0.88**  \n\n" "category_score  \n\n" "table"
            )

    def test_eval_output_cell(self):
        """
        GIVEN a EvalOutput object and dataset containing evaluation scores.
        WHEN a EvalOutputCell is created
        THEN the string representation of the EvalOutputCell matches what is expected
        """
        dataset_scores = [
            EvalScore(name="rouge", value=0.33),
            EvalScore(name="bert score", value=0.5),
            EvalScore(name="meteor", value=0.9),
        ]
        category_scores = [
            CategoryScore(
                name="Gender",
                scores=[
                    EvalScore(name="rouge", value=0.6),
                    EvalScore(name="bert score", value=0.7),
                    EvalScore(name="meteor", value=0.8),
                ],
            ),
            CategoryScore(
                name="Race",
                scores=[
                    EvalScore(name="rouge", value=0.4),
                    EvalScore(name="bert score", value=0.3),
                    EvalScore(name="meteor", value=0.2),
                ],
            ),
        ]
        eval_output = EvalOutput(
            eval_name="summarization accuracy",
            dataset_name="Dataset 1",
            prompt_template="prompt",
            dataset_scores=dataset_scores,
            category_scores=category_scores,
        )

        items = [
            {
                "prompt": "prompt 1",
                "category": "Gender",
                "rouge_score_column": 0.53,
                "bert_score_column": 0.30,
                "meteor_score_column": 0.66,
            },
            {
                "prompt": "prompt 2",
                "category": "Gender",
                "rouge_score_column": 0.2,
                "bert_score_column": 0.5,
                "meteor_score_column": 0.1,
            },
            {
                "prompt": "prompt 3",
                "category": "Race",
                "rouge_score_column": 0.9,
                "bert_score_column": 0.84,
                "meteor_score_column": 0.70,
            },
            {
                "prompt": "prompt 4",
                "category": "Race",
                "rouge_score_column": 0.43,
                "bert_score_column": 0.3,
                "meteor_score_column": 0.6,
            },
            {
                "prompt": "prompt 5",
                "category": "Gender",
                "rouge_score_column": 0.3,
                "bert_score_column": 0.7,
                "meteor_score_column": 0.01,
            },
            {
                "prompt": "prompt 6",
                "category": "Race",
                "rouge_score_column": 0.8,
                "bert_score_column": 0.51,
                "meteor_score_column": 0.2,
            },
        ]
        dataset = ray.data.from_items(items)
        score_column_names = {
            "rouge": "rouge_score_column",
            "bert score": "bert_score_column",
            "meteor": "meteor_score_column",
        }
        # Patching `CategoryBarPlotCell` due to validation bug with `BarPlotCell`, see
        # `TestCell.test_bar_plot_cell_init_success` for details.
        with patch("amazon_fmeval.reporting.eval_output_cells.CategoryBarPlotCell", return_value="CategoryBarPlot"):
            cell = EvalOutputCell(eval_output=eval_output, dataset=dataset, score_column_names=score_column_names)
            expected_cell = "## Summarization accuracy  \n\n**Dataset: Dataset 1**  \n\n### Rouge  \n\n**Overall Score: 0.33**  \n\nScore breakdown per rouge evaluation category:  \n\nCategoryBarPlot  \n\nThe model scores lowest in this category: **Race**.   \n\nBelow are a few examples of the highest and lowest-scoring examples across all categories:   \n\n**Top 5 highest-scoring examples:**  \n\n<table align=center>  \n<tr> <th align=center>prompt</th> <th align=center>category</th> <th align=center>rouge_score_column</th> <th align=center>bert_score_column</th> <th align=center>meteor_score_column</th> </tr>  \n<tr> <td align=center>prompt 3</td> <td align=center>Race</td> <td align=center>0.9</td> <td align=center>0.84</td> <td align=center>0.7</td> </tr>  \n<tr> <td align=center>prompt 6</td> <td align=center>Race</td> <td align=center>0.8</td> <td align=center>0.51</td> <td align=center>0.2</td> </tr>  \n<tr> <td align=center>prompt 1</td> <td align=center>Gender</td> <td align=center>0.53</td> <td align=center>0.3</td> <td align=center>0.66</td> </tr>  \n<tr> <td align=center>prompt 4</td> <td align=center>Race</td> <td align=center>0.43</td> <td align=center>0.3</td> <td align=center>0.6</td> </tr>  \n<tr> <td align=center>prompt 5</td> <td align=center>Gender</td> <td align=center>0.3</td> <td align=center>0.7</td> <td align=center>0.01</td> </tr>  \n</table>  \n\n**Bottom 5 lowest-scoring examples:**  \n\n<table align=center>  \n<tr> <th align=center>prompt</th> <th align=center>category</th> <th align=center>rouge_score_column</th> <th align=center>bert_score_column</th> <th align=center>meteor_score_column</th> </tr>  \n<tr> <td align=center>prompt 2</td> <td align=center>Gender</td> <td align=center>0.2</td> <td align=center>0.5</td> <td align=center>0.1</td> </tr>  \n<tr> <td align=center>prompt 5</td> <td align=center>Gender</td> <td align=center>0.3</td> <td align=center>0.7</td> <td align=center>0.01</td> </tr>  \n<tr> <td align=center>prompt 4</td> <td align=center>Race</td> <td align=center>0.43</td> <td align=center>0.3</td> <td align=center>0.6</td> </tr>  \n<tr> <td align=center>prompt 1</td> <td align=center>Gender</td> <td align=center>0.53</td> <td align=center>0.3</td> <td align=center>0.66</td> </tr>  \n<tr> <td align=center>prompt 6</td> <td align=center>Race</td> <td align=center>0.8</td> <td align=center>0.51</td> <td align=center>0.2</td> </tr>  \n</table>  \n\n### Bert score  \n\n**Overall Score: 0.5**  \n\nScore breakdown per bert score evaluation category:  \n\nCategoryBarPlot  \n\nThe model scores lowest in this category: **Race**.   \n\nBelow are a few examples of the highest and lowest-scoring examples across all categories:   \n\n**Top 5 highest-scoring examples:**  \n\n<table align=center>  \n<tr> <th align=center>prompt</th> <th align=center>category</th> <th align=center>rouge_score_column</th> <th align=center>bert_score_column</th> <th align=center>meteor_score_column</th> </tr>  \n<tr> <td align=center>prompt 3</td> <td align=center>Race</td> <td align=center>0.9</td> <td align=center>0.84</td> <td align=center>0.7</td> </tr>  \n<tr> <td align=center>prompt 5</td> <td align=center>Gender</td> <td align=center>0.3</td> <td align=center>0.7</td> <td align=center>0.01</td> </tr>  \n<tr> <td align=center>prompt 6</td> <td align=center>Race</td> <td align=center>0.8</td> <td align=center>0.51</td> <td align=center>0.2</td> </tr>  \n<tr> <td align=center>prompt 2</td> <td align=center>Gender</td> <td align=center>0.2</td> <td align=center>0.5</td> <td align=center>0.1</td> </tr>  \n<tr> <td align=center>prompt 1</td> <td align=center>Gender</td> <td align=center>0.53</td> <td align=center>0.3</td> <td align=center>0.66</td> </tr>  \n</table>  \n\n**Bottom 5 lowest-scoring examples:**  \n\n<table align=center>  \n<tr> <th align=center>prompt</th> <th align=center>category</th> <th align=center>rouge_score_column</th> <th align=center>bert_score_column</th> <th align=center>meteor_score_column</th> </tr>  \n<tr> <td align=center>prompt 1</td> <td align=center>Gender</td> <td align=center>0.53</td> <td align=center>0.3</td> <td align=center>0.66</td> </tr>  \n<tr> <td align=center>prompt 4</td> <td align=center>Race</td> <td align=center>0.43</td> <td align=center>0.3</td> <td align=center>0.6</td> </tr>  \n<tr> <td align=center>prompt 2</td> <td align=center>Gender</td> <td align=center>0.2</td> <td align=center>0.5</td> <td align=center>0.1</td> </tr>  \n<tr> <td align=center>prompt 6</td> <td align=center>Race</td> <td align=center>0.8</td> <td align=center>0.51</td> <td align=center>0.2</td> </tr>  \n<tr> <td align=center>prompt 5</td> <td align=center>Gender</td> <td align=center>0.3</td> <td align=center>0.7</td> <td align=center>0.01</td> </tr>  \n</table>  \n\n### Meteor  \n\n**Overall Score: 0.9**  \n\nScore breakdown per meteor evaluation category:  \n\nCategoryBarPlot  \n\nThe model scores lowest in this category: **Race**.   \n\nBelow are a few examples of the highest and lowest-scoring examples across all categories:   \n\n**Top 5 highest-scoring examples:**  \n\n<table align=center>  \n<tr> <th align=center>prompt</th> <th align=center>category</th> <th align=center>rouge_score_column</th> <th align=center>bert_score_column</th> <th align=center>meteor_score_column</th> </tr>  \n<tr> <td align=center>prompt 3</td> <td align=center>Race</td> <td align=center>0.9</td> <td align=center>0.84</td> <td align=center>0.7</td> </tr>  \n<tr> <td align=center>prompt 1</td> <td align=center>Gender</td> <td align=center>0.53</td> <td align=center>0.3</td> <td align=center>0.66</td> </tr>  \n<tr> <td align=center>prompt 4</td> <td align=center>Race</td> <td align=center>0.43</td> <td align=center>0.3</td> <td align=center>0.6</td> </tr>  \n<tr> <td align=center>prompt 6</td> <td align=center>Race</td> <td align=center>0.8</td> <td align=center>0.51</td> <td align=center>0.2</td> </tr>  \n<tr> <td align=center>prompt 2</td> <td align=center>Gender</td> <td align=center>0.2</td> <td align=center>0.5</td> <td align=center>0.1</td> </tr>  \n</table>  \n\n**Bottom 5 lowest-scoring examples:**  \n\n<table align=center>  \n<tr> <th align=center>prompt</th> <th align=center>category</th> <th align=center>rouge_score_column</th> <th align=center>bert_score_column</th> <th align=center>meteor_score_column</th> </tr>  \n<tr> <td align=center>prompt 5</td> <td align=center>Gender</td> <td align=center>0.3</td> <td align=center>0.7</td> <td align=center>0.01</td> </tr>  \n<tr> <td align=center>prompt 2</td> <td align=center>Gender</td> <td align=center>0.2</td> <td align=center>0.5</td> <td align=center>0.1</td> </tr>  \n<tr> <td align=center>prompt 6</td> <td align=center>Race</td> <td align=center>0.8</td> <td align=center>0.51</td> <td align=center>0.2</td> </tr>  \n<tr> <td align=center>prompt 4</td> <td align=center>Race</td> <td align=center>0.43</td> <td align=center>0.3</td> <td align=center>0.6</td> </tr>  \n<tr> <td align=center>prompt 1</td> <td align=center>Gender</td> <td align=center>0.53</td> <td align=center>0.3</td> <td align=center>0.66</td> </tr>  \n</table>"
            assert str(cell) == expected_cell
