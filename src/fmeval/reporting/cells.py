from abc import ABC
from typing import Any, List, Optional, Union
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import markdown
from IPython import display

from fmeval.reporting.constants import CENTER, ListType, MARKDOWN_EXTENSIONS, RIGHT
from fmeval.reporting.constants import SINGLE_NEWLINE, DOUBLE_NEWLINE


class Cell(ABC):
    """
    Base class for a report cell.
    """


class MarkdownCell(Cell):
    """
    Base class representing a markdown cell.
    """

    content: List[Union[str, "MarkdownCell"]]

    def __init__(self, *args):
        """
        Input may be strings or MarkdownCells.
        Examples:
            assert str(MarkdownCell("# Hello1")) == "# Hello1"
            assert str(MarkdownCell("# Hello1", "# Hello2")) == "# Hello1  \n\n# Hello2"
            assert str(MarkdownCell(MarkdownCell("# Hello1"), "# Hello2") == "# Hello1  \n\n# Hello2"
        """
        assert all(isinstance(arg, str) or isinstance(arg, MarkdownCell) for arg in args)
        self.content = list(args)

    def __str__(self):
        return DOUBLE_NEWLINE.join([str(s) for s in self.content])

    def show(self):  # pragma: no cover
        """
        Displays the cell content in an IPython notebook cell.
        """
        content = markdown.markdown(self.__str__(), extensions=MARKDOWN_EXTENSIONS)
        return display.HTML(content)


class HeadingCell(MarkdownCell):
    """
    This class represents a Markdown heading.
    """

    def __init__(self, text: str, level: int):
        """
        :param text: The text for this header
        :param level: The heading level
        """
        super().__init__(f"{'#' * level} {text}")


class BoldCell(MarkdownCell):
    """
    This class represents a bold piece of text.
    """

    def __init__(self, text):
        super().__init__(f"**{text}**")


class ListCell(MarkdownCell):
    """
    Creates a bulleted or numbered list.
    """

    def __init__(self, items: List[str], list_type: ListType):
        """
        :param items:  A list of strings where each string represents one item in the list.
        :param list_type: Whether the list is bulleted or numbered.
        """
        if list_type == ListType.NUMBERED:
            list_content = SINGLE_NEWLINE.join(f"{idx}. {value}" for idx, value in enumerate(items, start=1))
        else:
            list_content = SINGLE_NEWLINE.join(f"* {value}" for value in items)
        super().__init__(list_content)


class ColumnsLayoutCell(MarkdownCell):
    """
    This class creates a multi-column layout cell
    """

    def __init__(self, columns: List[List[Any]]):
        """
        :param columns: A list of Lists of strings or MarkdownCells, where each inner list is one column
        """
        div_begin = '<div class="row" markdown="1">'
        div_end = "</div>"
        col_div = f'<div class="column" markdown="1" style="float: left;width: {str(int(100//len(columns)))}%;">\n'
        result = "".join(
            [col_div + SINGLE_NEWLINE.join([str(item) for item in col]) + SINGLE_NEWLINE + div_end for col in columns]
        )
        super().__init__(div_begin, result, div_end, ' <br style="clear:both" />')


class FigureCell(MarkdownCell):
    """
    This class represents a MarkdownCell containing HTML for a Pyplot Figure.
    """

    def __init__(
        self,
        fig: plt.Figure,
        width: Optional[str] = None,
        height: Optional[str] = None,
        center: Optional[bool] = True,
    ):
        """
        Initializes a FigureCell.

        :param fig: The Pyplot figure that this cell represents.
        :param width: See _html_wrapper docstring
        :param height: See _html_wrapper docstring
        :param center: if the figure is center aligned
        """
        encoded = FigureCell._encode(fig)
        html = FigureCell._html_wrapper(encoded=encoded, height=height, width=width, center=center)
        super().__init__(html)

    @staticmethod
    def _encode(fig: plt.Figure) -> bytes:
        """
        Returns the base64 encoding of `fig`.

        :param fig: The Pyplot Figure to be encoded.
        :returns: The base64 encoding of `fig`
        """
        buffer = BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)  # save memory by closing the figure
        buffer.seek(0)
        encoded = base64.b64encode(buffer.getvalue())
        return encoded

    @staticmethod
    def _html_wrapper(
        encoded: bytes, height: Optional[str], width: Optional[str], center: Optional[bool] = True
    ) -> str:
        """
        Decodes the provided base64-encoded bytes (which will generally correspond to
        an encoded Pyplot Figure) as a string, then wraps it in HTML.

        :param encoded: The base64 encoded bytes to be wrapped
        :param height: The `height` HTML attribute
        :param width: The `width` HTML attribute
        :param center: Horizontal centering of the figure
        :returns: An HTML string representing the wrapped encoded string
        """
        style = 'style="display: block;'
        if center:  # pragma: no branch
            style += "margin-left:auto; margin-right: auto;"
        if width:
            style += f"width:{width}; "
        if height:
            style += f"height:{height};"
        encoded_str = encoded.decode("utf-8")
        html = f"<br><img {style}\" src='data:image/png;base64,{encoded_str}'><br>"
        return html


class BarPlotCell(FigureCell):
    """
    This class represents a Pyplot bar plot figure.
    """

    def __init__(
        self,
        labels: List[str],
        heights: List[Union[int, float]],
        color: Optional[Union[List[str], str]] = None,
        title: str = "Title",
        plot_height: Optional[str] = None,
        plot_width: Optional[str] = None,
        center: Optional[bool] = True,
        origin: float = 0,
    ):
        """
        Initializes a BarPlotCell.

        :param labels: The labels corresponding to each of the bars in the plot
        :param heights: The heights of the bars in the plot
        :param title: The title of the bar plot
        :param plot_height: Height of the plot as a string
        :param plot_width: Width the plot as a string
        :param center: Boolean indicating if the plot should be center aligned in the page
        """
        assert len(labels) == len(
            heights
        ), f"Number of labels in {labels} does not match number of bar heights in {heights}"
        fig = BarPlotCell._create_bar_plot_fig(labels, heights, color=color, title=title, origin=origin)
        super().__init__(fig, height=plot_height, width=plot_width, center=center)

    @staticmethod
    def _create_bar_plot_fig(
        labels: List[str],
        heights: List[Union[int, float]],
        color: Optional[Union[List[str], str]] = None,
        title: str = "Title",
        set_spines_visible: bool = False,
        set_ticks_visible: bool = False,
        set_horizontal_grid_lines: bool = True,
        max_bar_width: float = 0.3,
        origin: float = 0,
    ) -> plt.Figure:
        fig, ax = plt.subplots()
        heights = [height - origin for height in heights]
        ax.bar(labels, heights, width=max_bar_width, color=color)
        locs = ax.get_yticks()
        ax.set_yticks(locs, [round(loc + origin, ndigits=3) for loc in locs])
        if origin != 0:  # pragma: no cover
            ax.axhline(0, color="gray")
            ax.text(
                x=1.02,
                y=0,
                s="unbiased model",
                va="center",
                ha="left",
                bbox=dict(facecolor="w", alpha=0.5),
                transform=ax.get_yaxis_transform(),
            )
        ax.set_title(title)

        # auto-format bar labels to not overlap
        fig.autofmt_xdate()
        if set_horizontal_grid_lines:  # pragma: no branch
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        if not set_spines_visible:  # pragma: no branch
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
        if not set_ticks_visible:  # pragma: no branch
            plt.tick_params(axis="both", which="both", length=0)

        return fig


class TableCell(MarkdownCell):
    """
    This class represents an HTML table.

    Note that despite having "Cell" in its name, this class does *not*
    represent a single cell within an HTML table, but rather the entire table.
    The "Cell" suffix is included to match the naming convention for subclasses
    of MarkdownCell.
    """

    def __init__(
        self,
        data: List[List[Any]],
        headers: List[str],
        table_align: str = CENTER,
        cell_align: str = RIGHT,
        style: Optional[str] = None,
        caption: Optional[str] = None,
    ):
        """
        Initializes a TableCell.

        :param data: A 2D array representing tabular data
        :param headers: The table's headers, i.e. column names
        :param table_align: The alignment of the table within the overarching markdown
        :param cell_align: The alignment of text within each cell of the table
        """
        assert len(headers) == len(data[0]), (
            f"Number of headers in {headers} does not match " f"the number of columns in the data: {len(data[0])}"
        )
        html = TableCell._create_table_html(data, headers, table_align, cell_align, style, caption)
        super().__init__(html)

    @staticmethod
    def _create_table_html(
        data: List[List[Any]],
        headers: List[str],
        table_align: str,
        cell_align: str,
        style: Optional[str],
        caption: Optional[str],
    ) -> str:
        """
        Creates the HTML for a table.

        :param data: A 2D array representing tabular data
        :param headers: The table's headers, i.e. column names
        :param table_align: The alignment of the table within the overarching markdown
        :param cell_align: The alignment of text within each cell of the table
        :returns: A string encoding the HTML for the table
        """
        table_style = f'style="{style}"' if style else ""
        html = [
            f"<table align={table_align} {table_style}>",
            f'<caption style="text-align: left; padding-bottom: 15px;">{caption}</caption>' if caption else "",
            TableCell._create_table_row(headers, cell_align, is_header=True),
        ]
        for row in data:
            html.append(TableCell._create_table_row(row, cell_align))
        html.append("</table>")
        return SINGLE_NEWLINE.join(html)

    @staticmethod
    def _create_table_row(row: List[Any], cell_align: str, is_header: bool = False) -> str:
        """
        Creates the HTML for a single table row.

        :param row: A list representing the elements in the row
        :param cell_align: The alignment of text within each cell of the table
        :is_header: Whether `row` corresponds to the table header
        :returns: A string encoding the HTML for the table row
        """
        tag = "th" if is_header else "td"
        html = ["<tr>"]
        for i, elem in enumerate(row):
            style = f'style="text-align: {cell_align};"' if i != 0 else ""
            html.append(f"<{tag} {style}>{elem}</{tag}>")
        html.append("</tr>")
        return " ".join(html)
