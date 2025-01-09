from typing import Any, List, NamedTuple, Optional, Union
from fmeval.reporting.cells import (
    MarkdownCell,
    FigureCell,
    TableCell,
    BarPlotCell,
    HeadingCell,
    BoldCell,
    ListCell,
    ListType,
    ColumnsLayoutCell,
)
from fmeval.reporting.constants import CENTER, LEFT, RIGHT

import matplotlib.pyplot as plt
import pytest


def get_num_table_rows(html: str) -> int:
    """
    Returns the number of table rows present in `html`.

    Helper method for test function below.

    :param html: The HTML code for a table
    :returns: The number of rows (including the header) in the table
    """
    return html.count("<tr>")


class TestCell:
    # Hard-coded base64 strings corresponding to different test cases
    FIGURE_CELL_NO_ARGS = "<br><img style=\"display: block;margin-left:auto; margin-right: auto;\" src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAApQAAAHzCAYAAACe1o1DAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAACSRJREFUeJzt1sENwCAQwLDS/Xc+diAPhGRPkGfWzMwHAACH/tsBAAC8zVACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkGyJLgfivwmfiAAAAABJRU5ErkJggg=='><br>"
    FIGURE_CELL_WITH_ARGS = "<br><img style=\"display: block;width:200; height:100;\" src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAApQAAAHzCAYAAACe1o1DAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAACSRJREFUeJzt1sENwCAQwLDS/Xc+diAPhGRPkGfWzMwHAACH/tsBAAC8zVACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkBhKAAASQwkAQGIoAQBIDCUAAImhBAAgMZQAACSGEgCAxFACAJAYSgAAEkMJAEBiKAEASAwlAACJoQQAIDGUAAAkhhIAgMRQAgCQGEoAABJDCQBAYigBAEgMJQAAiaEEACAxlAAAJIYSAIDEUAIAkGyJLgfivwmfiAAAAABJRU5ErkJggg=='><br>"

    class TestCaseMarkdownInit(NamedTuple):
        args: List[Union[str, MarkdownCell]]
        expected: Optional[str] = None

    @pytest.mark.parametrize(
        "args, expected",
        [
            TestCaseMarkdownInit(
                args=["# Title", "## Header", "Some text"], expected="# Title  \n\n## Header  \n\nSome text"
            ),
            TestCaseMarkdownInit(
                args=["# Title", MarkdownCell("## Header", "Some text")],
                expected="# Title  \n\n## Header  \n\nSome text",
            ),
        ],
    )
    def test_markdown_cell_init_success(self, args, expected):
        """
        GIVEN valid arguments to the MarkdownCell initializer
        WHEN a MarkdownCell object is created
        THEN the str representation of the initialized MarkdownCell
            matches what is expected
        """
        md = MarkdownCell(*args)
        assert str(md) == expected

    @pytest.mark.parametrize(
        "args",
        [
            TestCaseMarkdownInit(args=["# Title", "## Header", 123]),
            TestCaseMarkdownInit(args=[["## Header", "Some text"]]),
            TestCaseMarkdownInit(args=["# Title", ["## Header", "Some text"]]),
        ],
    )
    def test_markdown_cell_init_failure(self, args):
        """
        GIVEN an input that is not a string or MarkdownCell
        WHEN a MarkdownCell object is created
        THEN an exception is raised
        """
        with pytest.raises(AssertionError):
            MarkdownCell(*args)

    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_heading_cell_init(self, level):
        """
        GIVEN valid arguments
        WHEN a HeadingCell is created
        THEN the str representation of the HeadingCell matches what is expected
        """
        heading_cell = HeadingCell("Some text.", level=level)
        assert str(heading_cell) == f"{'#' * level} Some text."

    def test_bold_cell_init(self):
        """
        GIVEN a valid argument
        WHEN a BoldCell is created
        THEN the str representation of the BoldCell matches what is expected
        """
        bold_cell = BoldCell("Some text.")
        assert str(bold_cell) == "**Some text.**"

    @pytest.mark.parametrize(
        "list_type, expected_output",
        [
            (ListType.NUMBERED, "1. Item 1  \n2. Item 2  \n3. Item 3"),
            (ListType.BULLETED, "* Item 1  \n* Item 2  \n* Item 3"),
        ],
    )
    def test_list_cell_init(self, list_type, expected_output):
        """
        GIVEN valid parameters to ListCell
        WHEN a ListCell is created
        THEN the str representation of the ListCell matches what is expected
        """
        list_items = ["Item 1", "Item 2", "Item 3"]
        list_cell = ListCell(items=list_items, list_type=list_type)
        assert str(list_cell) == expected_output

    def test_column_layout_cell_init(self):
        """
        GIVEN valid parameters to ColumnsLayoutCell
        WHEN a ColumnsLayoutCell is created
        THEN the str representation of the ColumnsLayoutCell matches what is expected
        """
        cols = [["Item 1", "Item 2"], ["Item 3", "Item 4"]]
        cols_cell = ColumnsLayoutCell(cols)
        expected_output = '<div class="row" markdown="1">  \n\n<div class="column" markdown="1" style="float: left;width: 50%;">\nItem 1  \nItem 2  \n</div><div class="column" markdown="1" style="float: left;width: 50%;">\nItem 3  \nItem 4  \n</div>  \n\n</div>  \n\n <br style="clear:both" />'
        assert str(cols_cell) == expected_output

    class TestCaseFigureCellInit(NamedTuple):
        args: List[Union[int, str]]
        expected: str

    @pytest.mark.parametrize(
        "args, expected",
        [
            TestCaseFigureCellInit(
                args=[],
                expected=FIGURE_CELL_NO_ARGS,
            ),
            TestCaseFigureCellInit(
                args=["200", "100", False],
                expected=FIGURE_CELL_WITH_ARGS,
            ),
        ],
    )
    def test_figure_cell_init(self, args, expected):
        """
        GIVEN a Pyplot Figure object is passed to FigureCell initializer
        WHEN a FigureCell is created
        THEN the FigureCell's string representation matches what is expected
        """
        plt_fig = plt.figure()
        figure = FigureCell(plt_fig, *args)
        assert str(figure) == expected

    class TestCaseTableCellInit(NamedTuple):
        data: List[List[Any]]
        headers: List[str]

    @pytest.mark.parametrize("style", ["table-layout: fixed;", None])
    @pytest.mark.parametrize("caption", ["Table caption", None])
    @pytest.mark.parametrize("cell_align", [CENTER, LEFT, RIGHT])
    @pytest.mark.parametrize("table_align", [CENTER, LEFT, RIGHT])
    @pytest.mark.parametrize(
        "data, headers",
        [
            TestCaseTableCellInit(
                data=[[1, 2, 3], [4, 5, 6]],
                headers=["col_1", "col_2", "col_3"],
            ),
            TestCaseTableCellInit(
                data=[["a", "b", "c"], ["d", "e", "f"]],
                headers=["col_1", "col_2", "col_3"],
            ),
        ],
    )
    def test_table_cell_init_success(self, data, headers, table_align, cell_align, caption, style):
        """
        GIVEN valid `data` and `headers` arguments
        WHEN a TableCell is created
        THEN TableCell's __str__ method returns the expected HTML
        """
        table = TableCell(data, headers, table_align=table_align, cell_align=cell_align, style=style, caption=caption)
        html = str(table)
        assert html.startswith(f"<table align={table_align}")
        assert html.endswith("</table>")
        if style:
            assert f'style="{style}"' in html
        if caption:
            assert f'<caption style="text-align: left; padding-bottom: 15px;">{caption}</caption>' in html
        assert get_num_table_rows(str(table)) == len(data) + 1  # + 1 for header row

        # assert all headers are present
        for i, header in enumerate(headers):
            header_cell = (
                f'<th style="text-align: {cell_align};">{header}</th>'
                if cell_align and i != 0
                else f"<th >{header}</th>"
            )
            assert header_cell in html

        # assert every row in `data` is present in the HTML
        for row in data:
            for i, elem in enumerate(row):
                table_cell = (
                    f'<td style="text-align: {cell_align};">{elem}</td>'
                    if cell_align and i != 0
                    else f"<td >{elem}</td>"
                )
                assert table_cell in html

    def test_table_cell_init_mismatch(self):
        """
        GIVEN `data` and `column_names` arguments that have mismatching lengths
        WHEN a TableCell is created
        THEN an exception is raised
        """
        with pytest.raises(AssertionError, match=r"Number of headers in"):
            TableCell(data=[[1, 2, 3], [4, 5, 6]], headers=["col_1", "col_2"], table_align=LEFT, cell_align=LEFT)

    @pytest.mark.parametrize(
        "kwargs",
        [
            ({}),
            ({"color": ["blue", "blue", "red"], "title": "Blue and Red"}),
        ],
    )
    def test_bar_plot_cell_init_success(self, kwargs):
        """
        GIVEN valid `labels` and `heights` initializer arguments
        WHEN a BarPlotCell is created
        THEN the str representation of the BarPlotCell object matches what is expected

        NOTE: we do not perform validation of str(bar_plot) like we do in the
        other test functions due to an extremely strange bug.
        TODO: Investigate bug and add validation.
        """
        BarPlotCell(["a", "b", "c"], [1, 2.0, 3], **kwargs)

    def test_bar_plot_cell_init_failure(self):
        """
        GIVEN `labels` and `heights` arguments that have mismatching lengths
        WHEN a BarPlotCell is created
        THEN an exception is raised
        """
        with pytest.raises(AssertionError, match=r"Number of labels"):
            BarPlotCell(["a", "b", "c"], [1, 2], "Title")
