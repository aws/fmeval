from typing import Optional
from dataclasses import dataclass
from fmeval.util import require
from fmeval.constants import SUPPORTED_MIME_TYPES


@dataclass
class DataConfig:
    """
    Configures the information required by data-loading components.

    Note that the term "location" used below refers to a string
    that can be used to locate the data that comprises a single
    column in the to-be-produced Ray Dataset. As an example,
    when the dataset MIME type is JSON or JSON Lines, the "location"
    is a JMESPath query.

    **Note**:
        Parsing logic used by data loaders make the assumption that
        attributes in this class with the suffix "_location" correspond
        to a "location" (defined above). When adding new attributes to this class,
        if an attribute corresponds to a location, the attribute name must end
        with "_location"

    :param dataset_name: the dataset name
    :param dataset_uri: either a local path or s3 URI representing where the dataset is stored
    :param dataset_mime_type: the MIME type of the dataset file
    :param model_input_location: the location  for model inputs
    :param model_output_location: the location for model outputs
    :param target_output_location: the location for target outputs
    :param category_location: the location for categories
    :param sent_more_input_location: the location for the "sent more"
            inputs (used by the Prompt Stereotyping evaluation algorithm)
    :param sent_less_input_location: the location for the "sent less"
            inputs (used by the Prompt Stereotyping evaluation algorithm)
    :param sent_more_log_prob_location: the location for the "sent more"
            input log probability (used by the Prompt Stereotyping evaluation algorithm)
    :param sent_less_log_prob_location: the location for the "sent less"
            input log probability (used by the Prompt Stereotyping evaluation algorithm).
    :param context_location: the location of the context for RAG evaluations.
    """

    dataset_name: str
    dataset_uri: str
    dataset_mime_type: str
    model_input_location: Optional[str] = None
    model_output_location: Optional[str] = None
    target_output_location: Optional[str] = None
    category_location: Optional[str] = None
    sent_more_input_location: Optional[str] = None
    sent_less_input_location: Optional[str] = None
    sent_more_log_prob_location: Optional[str] = None
    sent_less_log_prob_location: Optional[str] = None
    context_location: Optional[str] = None

    def __post_init__(self):
        require(
            self.dataset_mime_type in SUPPORTED_MIME_TYPES,
            f"Unsupported MIME type: {self.dataset_mime_type}. "
            f"The following mime types are supported: {SUPPORTED_MIME_TYPES}.",
        )
