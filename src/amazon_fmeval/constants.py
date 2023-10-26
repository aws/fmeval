# Output results path
import os

EVAL_RESULTS_PATH = "EVAL_RESULTS_PATH"
DEFAULT_EVAL_RESULTS_PATH = "/tmp/eval_results/"

# Number of actors to use
PARALLELIZATION_FACTOR = "PARALLELIZATION_FACTOR"
PARTITION_MULTIPLIER = 5

# Environment variables for SageMaker endpoint urls
SAGEMAKER_SERVICE_ENDPOINT_URL = "SAGEMAKER_SERVICE_ENDPOINT_URL"
SAGEMAKER_RUNTIME_ENDPOINT_URL = "SAGEMAKER_RUNTIME_ENDPOINT_URL"

# Constants for column names in loaded ray datasets
MODEL_INPUT_COLUMN_NAME = "model_input"
MODEL_OUTPUT_COLUMN_NAME = "model_output"
MODEL_LOG_PROBABILITY_COLUMN_NAME = "model_log_probability"
TARGET_OUTPUT_COLUMN_NAME = "target_output"
CATEGORY_COLUMN_NAME = "category"
SENT_MORE_INPUT_COLUMN_NAME = "sent_more_input"
SENT_LESS_INPUT_COLUMN_NAME = "sent_less_input"
SENT_MORE_PROMPT_COLUMN_NAME = "sent_more_prompt"
SENT_LESS_PROMPT_COLUMN_NAME = "sent_less_prompt"
SENT_LESS_LOG_PROB_COLUMN_NAME = "sent_less_log_prob"
SENT_MORE_LOG_PROB_COLUMN_NAME = "sent_more_log_prob"
SENT_MORE_OUTPUT_COLUMN_NAME = "sent_more_output"
SENT_LESS_OUTPUT_COLUMN_NAME = "sent_less_output"

COLUMN_NAMES = {
    MODEL_INPUT_COLUMN_NAME,
    MODEL_OUTPUT_COLUMN_NAME,
    TARGET_OUTPUT_COLUMN_NAME,
    CATEGORY_COLUMN_NAME,
    SENT_MORE_INPUT_COLUMN_NAME,
    SENT_LESS_INPUT_COLUMN_NAME,
    SENT_MORE_OUTPUT_COLUMN_NAME,
    SENT_LESS_OUTPUT_COLUMN_NAME,
}

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

# The absolute tolerance used when performing approximate numerical comparisons
ABS_TOL = 1e-4

# Jumpstart
JUMPSTART_MODEL_ID = "jumpstart_model_id"
JUMPSTART_MODEL_VERSION = "jumpstart_model_version"
MODEL_ID = "model_id"
SPEC_KEY = "spec_key"
DEFAULT_PAYLOADS = "default_payloads"
JUMPSTART_REGION = os.environ.get("AWS_REGION", "us-east-2")
SDK_MANIFEST_FILE = "models_manifest.json"
JUMPSTART_BUCKET_BASE_URL = "https://jumpstart-cache-prod-{}.s3.{}.amazonaws.com".format(
    JUMPSTART_REGION, JUMPSTART_REGION
)
GENERATED_TEXT_JMESPATH_EXPRESSION = "*.output_keys.generated_text"
