from enum import Enum

# Output results path

EVAL_RESULTS_PATH = "EVAL_RESULTS_PATH"
DEFAULT_EVAL_RESULTS_PATH = "/tmp/eval_results/"

# Number of actors to use
PARALLELIZATION_FACTOR = "PARALLELIZATION_FACTOR"
PARTITION_MULTIPLIER = 5

# Environment variables for SageMaker endpoint urls
SAGEMAKER_SERVICE_ENDPOINT_URL = "SAGEMAKER_SERVICE_ENDPOINT_URL"
SAGEMAKER_RUNTIME_ENDPOINT_URL = "SAGEMAKER_RUNTIME_ENDPOINT_URL"


class ColumnNames(Enum):
    """
    This enum represents the names of columns that appear
    in the finalized Ray Dataset produced by an EvalAlgorithm.
    These are the only columns whose data gets written to output
    records by util.save_dataset. Other algorithm-specific columns
    that get produced as intermediate results (for example,
    CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME in ClassificationAccuracy)
    are not included here, and thus won't get saved by save_dataset.
    """

    MODEL_INPUT_COLUMN_NAME = "model_input"
    PROMPT_COLUMN_NAME = "prompt"
    MODEL_OUTPUT_COLUMN_NAME = "model_output"
    MODEL_LOG_PROBABILITY_COLUMN_NAME = "model_log_probability"
    TARGET_OUTPUT_COLUMN_NAME = "target_output"
    CATEGORY_COLUMN_NAME = "category"
    SENT_MORE_INPUT_COLUMN_NAME = "sent_more_input"
    SENT_LESS_INPUT_COLUMN_NAME = "sent_less_input"
    SENT_MORE_PROMPT_COLUMN_NAME = "sent_more_prompt"
    SENT_LESS_PROMPT_COLUMN_NAME = "sent_less_prompt"
    SENT_MORE_LOG_PROB_COLUMN_NAME = "sent_more_log_prob"
    SENT_LESS_LOG_PROB_COLUMN_NAME = "sent_less_log_prob"


COLUMN_NAMES = [e.value for e in ColumnNames]

# This suffix must be included at the end of all
# DataConfig attribute names where the attribute
# represents a mechanism for locating the data for
# a column. An example mechanism is a JMESPath query
# (when the dataset format is JSON/JSON Lines).
DATA_CONFIG_LOCATION_SUFFIX = "_location"

# Supported MIME types
MIME_TYPE_JSON = "application/json"
MIME_TYPE_JSONLINES = "application/jsonlines"

SUPPORTED_MIME_TYPES = [MIME_TYPE_JSON, MIME_TYPE_JSONLINES]

# Aggregation methods
MEAN = "mean"

# Configures `save_dataset` behavior regarding how many
# EvalOutputRecords to accumulate before writing them
# to the output JSON Lines file.
EVAL_OUTPUT_RECORDS_BATCH_SIZE = 1024

# Dataloader seed
SEED = 1234

# Semantic robustness perturbation types
BUTTER_FINGER = "butter_finger"
RANDOM_UPPER_CASE = "random_upper_case"
WHITESPACE_ADD_REMOVE = "whitespace_add_remove"

PREFIX_FOR_DELTA_SCORES = "delta_"

# Check if model is deterministic for first NUM_ROWS_DETERMINISTIC rows of dataset
NUM_ROWS_DETERMINISTIC = 5

MAX_ROWS_TO_TAKE = 100000

# The absolute tolerance used when performing approximate numerical comparisons,
# specifically, when comparing EvalScore objects.
ABS_TOL = 1e-3

# Jumpstart
JUMPSTART_MODEL_ID = "jumpstart_model_id"
JUMPSTART_MODEL_VERSION = "jumpstart_model_version"
MODEL_ID = "model_id"
SPEC_KEY = "spec_key"
DEFAULT_PAYLOADS = "default_payloads"
SDK_MANIFEST_FILE = "models_manifest.json"
JUMPSTART_BUCKET_BASE_URL_FORMAT = "https://jumpstart-cache-prod-{}.s3.{}.amazonaws.com"
JUMPSTART_BUCKET_BASE_URL_FORMAT_ENV_VAR = "JUMPSTART_BUCKET_BASE_URL_FORMAT"
GENERATED_TEXT_JMESPATH_EXPRESSION = "*.output_keys.generated_text"
