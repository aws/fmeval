from constants import MIME_TYPE_JSON
from model_runners.extractors import create_extractor, JsonExtractor


def test_create_extractor():
    assert isinstance(create_extractor(model_accept_type=MIME_TYPE_JSON), JsonExtractor)
