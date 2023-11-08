import os
import json
from pytest import approx
from amazon_fmeval.eval_algorithms.summarization_accuracy import (
    SummarizationAccuracy,
    METEOR_SCORE,
    ROUGE_SCORE,
    BERT_SCORE,
)
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.constants import MIME_TYPE_JSONLINES
from test.integration.models.model_runners import bedrock_model_runner


ABS_TOL = 1e-2  # Bedrock models are not deterministic, so we use a higher tolerance here
os.environ["PARALLELIZATION_FACTOR"] = "2"
eval_algo = SummarizationAccuracy()


def format_input(input_str: str) -> str:
    """
    Formats the input to match what is required by Claude,
    specifically, anthropic.claude-v2.
    """
    return f"Human: {input_str}\n\nAssistant:\n"


class TestSummarizationAccuracy:
    def test_evaluate_sample(self, integration_tests_dir):
        with open(os.path.join(integration_tests_dir, "datasets", "xsum_sample.jsonl")) as fh:
            json_obj = json.loads(fh.readline())
            original_text = json_obj["document"]
            target_output = json_obj["summary"]
            model_input = f"Summarise the following text in one sentence: {original_text}"
            model_output = bedrock_model_runner.predict(format_input(model_input))[0]
            eval_scores = eval_algo.evaluate_sample(target_output, model_output)
            for eval_score in eval_scores:  # pragma: no branch
                if eval_score.name == METEOR_SCORE:
                    assert eval_score.value == approx(0.380, abs=ABS_TOL)
                elif eval_score.name == ROUGE_SCORE:
                    assert eval_score.value == approx(0.250, abs=ABS_TOL)
                elif eval_score.name == BERT_SCORE:
                    assert eval_score.value == approx(0.734, abs=ABS_TOL)

    def test_evaluate(self, integration_tests_dir):
        dataset_config = DataConfig(
            dataset_name="xsum_sample",
            dataset_uri=os.path.join(integration_tests_dir, "datasets", "xsum_sample.jsonl"),
            dataset_mime_type=MIME_TYPE_JSONLINES,
            model_input_location="document",
            target_output_location="summary",
        )
        eval_outputs = eval_algo.evaluate(
            model=bedrock_model_runner,
            dataset_config=dataset_config,
            prompt_template=format_input("Summarise the following text in one sentence: $feature"),
            save=True,
        )
        eval_output = eval_outputs[0]
        for eval_score in eval_output.dataset_scores:
            if eval_score.name == METEOR_SCORE:
                assert eval_score.value == approx(0.279, abs=ABS_TOL)
            elif eval_score.name == ROUGE_SCORE:
                assert eval_score.value == approx(0.084, abs=ABS_TOL)
            elif eval_score.name == BERT_SCORE:
                assert eval_score.value == approx(0.677, abs=ABS_TOL)
