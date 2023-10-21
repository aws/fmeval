from unittest.mock import patch, Mock, call, MagicMock

from amazon_fmeval.eval_algorithms import EvalOutput, EvalScore, CategoryScore
from amazon_fmeval.eval_algorithms.prompt_stereotyping import PROMPT_STEREOTYPING
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
        score_name = PROMPT_STEREOTYPING
        dataset_score = 0.6

        cat_bar_plot = CategoryBarPlotCell(categories, scores, score_name, dataset_score)
        manually_created_bar_plot = BarPlotCell(
            labels=categories + [DATASET_SCORE_LABEL],
            heights=scores + [dataset_score],
            color=[CATEGORY_BAR_COLOR, CATEGORY_BAR_COLOR, CATEGORY_BAR_COLOR, OVERALL_BAR_COLOR],
            title="Is_biased Score",
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
                expected_table=TableCell(
                    data=[["b", 2], ["a", 1], ["c", 3]], headers=["Name", "Age"], cell_align="left"
                ),
            ),
            TestCaseRayDatasetTableCell(
                items=[
                    {"Name": "b", "Age": 2},
                    {"Name": "a", "Age": 1},
                    {"Name": "c", "Age": 3},
                ],
                kwargs={"col_to_sort": "Age"},
                expected_table=TableCell(
                    data=[["a", 1], ["b", 2], ["c", 3]], headers=["Name", "Age"], cell_align="left"
                ),
            ),
            TestCaseRayDatasetTableCell(
                items=[
                    {"Name": "b", "Age": 2},
                    {"Name": "a", "Age": 1},
                    {"Name": "c", "Age": 3},
                    {"Name": "d", "Age": 4},
                ],
                kwargs={"k": 3},
                expected_table=TableCell(
                    data=[["b", 2], ["a", 1], ["c", 3]], headers=["Name", "Age"], cell_align="left"
                ),
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
                expected_table=TableCell(
                    data=[["e", 5], ["d", 4], ["c", 3]], headers=["Name", "Age"], cell_align="left"
                ),
            ),
            TestCaseRayDatasetTableCell(
                items=[
                    {"Name": "b", "Age": -2},
                    {"Name": "a", "Age": 1},
                    {"Name": "c", "Age": 3},
                ],
                kwargs={"col_to_sort": "Age", "k": 3, "abs_val": True},
                expected_table=TableCell(
                    data=[["a", 1], ["b", -2], ["c", 3]], headers=["Name", "Age"], cell_align="left"
                ),
            ),
            TestCaseRayDatasetTableCell(
                items=[
                    {"Name": "b", "Age": 2, "category": "Sex"},
                    {"Name": "a", "Age": 1, "category": "Race"},
                    {"Name": "c", "Age": 3, "category": "Sex"},
                ],
                kwargs={},
                expected_table=TableCell(
                    data=[["Sex", "b", 2], ["Race", "a", 1], ["Sex", "c", 3]],
                    headers=["Category", "Name", "Age"],
                    cell_align="left",
                ),
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

        expected_cell = "The plot shows the score breakdown into individual categories.  \n\n  \n\nAggBPCell  \n\nThe model scores lowest in the category **religion**. "
        with patch("amazon_fmeval.reporting.eval_output_cells.CategoryBarPlotCell", return_value="AggBPCell"):
            cell = CategoryScoreCell(categories, scores, "prompt stereotyping", dataset_score)
            assert str(cell) == expected_cell

    @pytest.mark.parametrize(
        "score_column_name, binary, expected_cell",
        [
            (
                "word_error_rate",
                False,
                "Below are a few examples of the highest and lowest-scoring examples across all categories. The lower the word error rate, the better the model performs. Some text may be truncated due to length constraints. To view the full prompts, please go to the S3 job output location that you specified when configuring the job.  \n\nRayTable  \n\nRayTable",
            ),
            (
                "log_probability_difference",
                False,
                "For each sentence pair, we report the log probability difference, a value ranging -&#8734; to &#8734;, indicating how much the model stereotypes. Below are a few example of the most and least stereotypical prompts. Some text may be truncated due to length constraints. To view the full prompts, please go to the S3 job output location that you specified when configuring the job.  \n\nRayTable  \n\nRayTable",
            ),
            (
                "",
                False,
                "Below are a few examples of the highest and lowest-scoring examples across all categories. Some text may be truncated due to length constraints. To view the full prompts, please go to the S3 job output location that you specified when configuring the job.   \n\nRayTable  \n\nRayTable",
            ),
            (
                "",
                True,
                "Below are a few examples of correct and incorrect model responses. Some text may be truncated due to length constraints. To view the full prompts, please go to the S3 job output location that you specified when configuring the job.   \n\nRayTable  \n\nRayTable",
            ),
        ],
    )
    @patch("amazon_fmeval.reporting.eval_output_cells.RayDatasetTableCell", return_value="RayTable")
    def test_score_table_cell(self, mock_ray_table, score_column_name, binary, expected_cell):
        """
        GIVEN a valid dataset and score column name
        WHEN a ScoreTableCell is created
        THEN the string representation of the ScoreTableCell matches what is expected,
            and the included RayDatasetTableCells that make up the ScoreTableCell
            are initialized correctly
        """
        dataset = MagicMock()
        dataset.count = Mock(return_value=10)
        cell = ScoreTableCell(dataset, score_column_name, binary)

        # Assert structure of returned MarkdownCell is correct
        assert str(cell) == expected_cell

        if score_column_name == "log_probability_difference":
            abs_val = True
            captions = [
                f"Top {min(NUM_SAMPLES_TO_DISPLAY_IN_TABLE, dataset.count())} most stereotypical examples:",
                f"Top {min(NUM_SAMPLES_TO_DISPLAY_IN_TABLE, dataset.count())} least stereotypical examples:",
            ]
        elif binary:
            captions = [
                f"{min(NUM_SAMPLES_TO_DISPLAY_IN_TABLE, dataset.count())} correct examples:",
                f"{min(NUM_SAMPLES_TO_DISPLAY_IN_TABLE, dataset.count())} incorrect examples:",
            ]
            abs_val = False
        else:
            abs_val = False
            captions = [
                f"Top {min(NUM_SAMPLES_TO_DISPLAY_IN_TABLE, dataset.count())} highest-scoring examples:",
                f"Bottom {min(NUM_SAMPLES_TO_DISPLAY_IN_TABLE, dataset.count())} lowest-scoring examples:",
            ]
        # Assert RayDatasetTableCells are created with the right arguments
        mock_ray_table.assert_has_calls(
            [
                call(
                    dataset,
                    score_column_name,
                    k=NUM_SAMPLES_TO_DISPLAY_IN_TABLE,
                    descending=True,
                    abs_val=abs_val,
                    caption=captions[0],
                ),
                call(
                    dataset,
                    score_column_name,
                    k=NUM_SAMPLES_TO_DISPLAY_IN_TABLE,
                    descending=False,
                    abs_val=abs_val,
                    caption=captions[1],
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
            dataset = MagicMock()
            dataset.columns = Mock(return_value=["col1", "col2"])
            dataset.select_columns = Mock()
            score_name = "prompt_stereotyping"
            score_column_name = "prompt_stereotyping"
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
            expected_cell = "##### Is_biased Score  \n\nThe is_biased prompt stereotyping score is a value between 0 and 1. Non-stereotyping models score around 0.5. Both 0 and 1 indicate stereotyping model behaviour.  \n\n**Overall Score: 0.88**  \n\ncategory_score  \n\ntable"
            # THEN
            assert str(cell) == expected_cell

    def test_eval_output_cell_custom_dataset(self):
        """
        GIVEN a EvalOutput object for an evaluation on a custom dataset and dataset containing evaluation scores.
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
            eval_name="summarization_accuracy",
            dataset_name="Dataset 1",
            prompt_template="prompt",
            dataset_scores=dataset_scores,
            category_scores=category_scores,
        )

        dataset = MagicMock()
        dataset.count = Mock(return_value=10)
        dataset.columns = Mock(return_value=["col1", "col2", "col3"])
        dataset.select_columns = Mock()
        with patch("amazon_fmeval.reporting.eval_output_cells.ScoreCell", return_value="score_cell"):
            cell = EvalOutputCell(eval_output=eval_output, dataset=dataset)
            expected_cell = "#### Custom Dataset: Dataset 1  \n\nWe sampled 10 records out of 10 in the full dataset.  \n\n  \n\nscore_cell  \n\nscore_cell  \n\nscore_cell"
            assert str(cell) == expected_cell

    def test_eval_output_cell_built_in_dataset(self):
        """
        GIVEN a EvalOutput object for an evaluation on a built-in dataset and dataset containing evaluation scores.
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
            eval_name="summarization_accuracy",
            dataset_name="xsum",
            prompt_template="prompt",
            dataset_scores=dataset_scores,
            category_scores=category_scores,
        )
        dataset = MagicMock()
        dataset.count = Mock(return_value=10)
        dataset.columns = Mock(return_value=["col1", "col2", "col3"])
        dataset.select_columns = Mock()
        with patch("amazon_fmeval.reporting.eval_output_cells.ScoreCell", return_value="score_cell"):
            cell = EvalOutputCell(eval_output=eval_output, dataset=dataset)
            expected_cell = '#### Built-in Dataset: <a style="color:#006DAA;" href="https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset">XSUM</a>  \n\nA dataset consisting of newspaper articles from the BBC and their reference summaries. The reference summaries consist of a single sentence: the boldfaced sentence at the begininning of each BBC article, provided by article’s authors. We sampled 10 records out of 204045 in the full dataset.  \n\n  \n\nscore_cell  \n\nscore_cell  \n\nscore_cell'
            assert str(cell) == expected_cell

    def test_eval_output_cell_eval_error(self):
        """
        GIVEN an EvalOutput where the evaluation failed
        WHEN an EvalOutputCell is generated
        THEN the dataset name and error message are returned.
        """
        eval_output = EvalOutput(
            eval_name="summarization accuracy",
            dataset_name="Dataset 1",
            prompt_template="prompt",
            error="The summarization accuracy evaluation failed.",
        )
        with patch("amazon_fmeval.reporting.eval_output_cells.ScoreCell", return_value="score_cell"):
            cell = EvalOutputCell(eval_output=eval_output)
            expected_cell = "#### Custom Dataset: Dataset 1  \n\n  \n\n  \n\n**This evaluation failed with the error message: The summarization accuracy evaluation failed.**"
            assert str(cell) == expected_cell
