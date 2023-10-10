from typing import Tuple, List

from amazon_fmeval.model_runners.composers import Composer
from amazon_fmeval.model_runners.extractors import JsonExtractor
from amazon_fmeval.model_runners.model_runner import ModelRunner


class TestModelRunner:
    def test_model_runner(self):
        class MyModelRunner(ModelRunner):
            def __init__(self):
                super().__init__('{"content": $prompt}', "output", "log_probability")

            def predict(self, prompt: str) -> Tuple[str, float]:
                pass

            def batch_predict(self, prompts: List[str]) -> List[Tuple[str, float]]:
                pass

        model_runner = MyModelRunner()
        assert isinstance(model_runner._composer, Composer)
        assert isinstance(model_runner._extractor, JsonExtractor)
