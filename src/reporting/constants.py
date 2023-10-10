from enum import Enum


# For general HTML alignment
CENTER = "center"
LEFT = "left"
RIGHT = "right"


class ListType(Enum):
    BULLETED = "bulleted"
    NUMBERED = "numbered"


# For general use in Markdown-related code
SINGLE_NEWLINE = "  \n"
DOUBLE_NEWLINE = "  \n\n"

# For tables and bar plots
NUM_SAMPLES_TO_DISPLAY_IN_TABLE = 5
CATEGORY_BAR_COLOR = "#074af2"  # A shade of blue
OVERALL_BAR_COLOR = "Red"

# Extensions used by the markdown library to convert markdown to HTML
MARKDOWN_EXTENSIONS = ["tables", "md_in_html"]

# Dataset score label used in category bar plot
DATASET_SCORE_LABEL = "Overall"
