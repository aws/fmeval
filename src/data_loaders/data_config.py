from typing import Optional
from dataclasses import dataclass
from util import require
from constants import SUPPORTED_MIME_TYPES


@dataclass
class DataConfig:
    """
    Configures the information required by data-loading components.

    Note that the term "location" used below refers to a string
    that can be used to locate the data that comprises a single
    column in the to-be-produced Ray Dataset. As an example,
    when the dataset MIME type is JSON or JSON Lines, the "location"
    is a JMESPath query.

    Attributes:
        dataset_name: the dataset name
        dataset_uri: either a local path or s3 URI representing where
            the dataset is stored
        dataset_mime_type: the MIME type of the dataset file
        model_input_location: the location  for model inputs
        model_output_location: the location for model outputs
        target_output_location: the location for target outputs
        category_location: the location for categories
        sent_more_input_location: the location for the "sent more"
            inputs (used by the Prompt Stereotyping evaluation algorithm)
        sent_less_input_location: the location for the "sent less"
            inputs (used by the Prompt Stereotyping evaluation algorithm)
        sent_more_output_location: the location for the "sent more"
            outputs (used by the Prompt Stereotyping evaluation algorithm)
        sent_less_output_location: the location for the "sent less"
            outputs (used by the Prompt Stereotyping evaluation algorithm)

    Note:
        Parsing logic used by data loaders make the assumption that
        attributes in this class with the suffix "_location" correspond
        to a "location" (defined above). When adding new attributes to this class,
        if an attribute corresponds to a location, the attribute name must end
        with "_location".
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
    sent_more_output_location: Optional[str] = None
    sent_less_output_location: Optional[str] = None

    def __post_init__(self):
        require(
            self.dataset_mime_type in SUPPORTED_MIME_TYPES,
            f"Unsupported MIME type: {self.dataset_mime_type}. "
            f"The following mime types are supported: {SUPPORTED_MIME_TYPES}.",
        )
