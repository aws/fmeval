from amazon_fmeval.constants import MIME_TYPE_JSON
from amazon_fmeval.model_runners.extractors import create_extractor, JsonExtractor


def test_create_extractor():
    assert isinstance(create_extractor(model_accept_type=MIME_TYPE_JSON), JsonExtractor)
