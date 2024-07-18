from enum import Enum
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional

# Output results path

EVAL_RESULTS_PATH = "EVAL_RESULTS_PATH"
DEFAULT_EVAL_RESULTS_PATH = "/tmp/eval_results/"

# Number of actors to use
PARALLELIZATION_FACTOR = "PARALLELIZATION_FACTOR"
PARTITION_MULTIPLIER = 5

# Environment variables for SageMaker endpoint urls
SAGEMAKER_SERVICE_ENDPOINT_URL = "SAGEMAKER_SERVICE_ENDPOINT_URL"
SAGEMAKER_RUNTIME_ENDPOINT_URL = "SAGEMAKER_RUNTIME_ENDPOINT_URL"

# We default the boto3 client region to us-west-2 as the dataset bucket cannot be accessed from opt-in regions.
BUILT_IN_DATASET_PREFIX = "s3://fmeval/datasets"
BUILT_IN_DATASET_DEFAULT_REGION = "us-west-2"

# Mapping of iso region to built in dataset region in the same partition
BUILT_IN_DATASET_ISO_REGIONS = {"us-isof-south-1": "us-isof-south-1", "us-isof-east-1": "us-isof-south-1"}

# Environment variable for disabling telemetry
DISABLE_FMEVAL_TELEMETRY = "DISABLE_FMEVAL_TELEMETRY"


@dataclass(frozen=True)
class Column:
    """
    This class represents a column in the Ray Dataset produced by
    an evaluation algorithm's `evaluate` method.

    Note that some columns are created during the "data loading" stage,
    when the initial Ray Dataset object is created by data_loaders.util.get_dataset,
    while the remaining columns are created during the execution of `evaluate`.
    Only the contents of the columns created during the data loading stage
    have the potential to be casted to strings.

    :param name: The name of the column as it appears in the Ray Dataset.
    :param should_cast: Whether the contents of this column should
        be casted to strings during data loading.
        This parameter is None (as opposed to False) for columns that do
        not exist during data loading to make it clear that casting these
        columns is not even a possibility to begin with.
    """

    name: str
    should_cast: Optional[bool] = None


class DatasetColumns(Enum):
    """
    This Enum represents the columns that appear in the finalized
    Ray Dataset produced during the course of executing an eval algorithm's
    `evaluate` method.

    These are the only columns (aside from score columns) whose
    data gets written to output records by `util.save_dataset`.
    Other algorithm-specific columns that get produced as intermediate
    results (for example, CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME in
    ClassificationAccuracy) are not included here, and thus won't
    get saved by `util.save_dataset`.
    """

    MODEL_INPUT = Column(name="model_input", should_cast=True)
    PROMPT = Column(name="prompt")
    MODEL_OUTPUT = Column(name="model_output", should_cast=True)
    MODEL_LOG_PROBABILITY = Column(name="model_log_probability")
    TARGET_OUTPUT = Column(name="target_output", should_cast=True)
    CATEGORY = Column(name="category", should_cast=True)
    CONTEXT = Column(name="context", should_cast=True)
    SENT_MORE_INPUT = Column(name="sent_more_input", should_cast=True)
    SENT_LESS_INPUT = Column(name="sent_less_input", should_cast=True)
    SENT_MORE_PROMPT = Column(name="sent_more_prompt")
    SENT_LESS_PROMPT = Column(name="sent_less_prompt")
    SENT_MORE_LOG_PROB = Column(name="sent_more_log_prob", should_cast=False)
    SENT_LESS_LOG_PROB = Column(name="sent_less_log_prob", should_cast=False)
    ERROR = Column(name="error", should_cast=False)


DATASET_COLUMNS = OrderedDict((col.value.name, col) for col in DatasetColumns)
COLUMNS_WITH_LISTS = [DatasetColumns.CONTEXT.value.name]

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
JUMPSTART_MODEL_TYPE = "jumpstart_model_type"
MODEL_ID = "model_id"
SPEC_KEY = "spec_key"
DEFAULT_PAYLOADS = "default_payloads"
SDK_MANIFEST_FILE = "models_manifest.json"
PROPRIETARY_SDK_MANIFEST_FILE = "proprietary-sdk-manifest.json"
JUMPSTART_BUCKET_BASE_URL_FORMAT = "https://jumpstart-cache-prod-{}.s3.{}.amazonaws.com"
JUMPSTART_BUCKET_BASE_URL_FORMAT_ENV_VAR = "JUMPSTART_BUCKET_BASE_URL_FORMAT"
GENERATED_TEXT_JMESPATH_EXPRESSION = "*.output_keys.generated_text"
INPUT_LOG_PROBS_JMESPATH_EXPRESSION = "*.output_keys.input_logprobs"
EMBEDDING_JMESPATH_EXPRESSION = "embedding"
IS_EMBEDDING_MODEL = "is_embedding_model"

# BERTScore
BERTSCORE_DEFAULT_MODEL = "microsoft/deberta-xlarge-mnli"


# S3 multi-part upload constants
UPLOAD_ID = "UploadId"
PARTS = "Parts"
E_TAG = "ETag"
PART_NUMBER = "PartNumber"
