import pytest
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.exceptions import EvalAlgorithmClientError


class TestDataConfig:
    def test_data_config_init_invalid_mime_type(self):
        """
        GIVEN an invalid dataset_mime_type attribute
        WHEN creating a DataConfig
        THEN an EvalAlgorithmClientError is raised by __post_init__
        """
        with pytest.raises(EvalAlgorithmClientError, match="Unsupported MIME type: fake_mime_type."):
            DataConfig(dataset_name="dataset", dataset_uri="path/to/dataset", dataset_mime_type="fake_mime_type")
