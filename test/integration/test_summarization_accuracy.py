import os
import json
from pytest import approx
from fmeval.eval_algorithms.summarization_accuracy import (
    SummarizationAccuracy,
    METEOR_SCORE,
    ROUGE_SCORE,
    BERT_SCORE,
)
from fmeval.eval_algorithms import (
    DATASET_CONFIGS,
    GIGAWORD,
)
from test.integration.models.model_runners import bedrock_model_runner

ABS_TOL = 5e-2  # Bedrock models are not deterministic, so we use a higher tolerance here
os.environ["PARALLELIZATION_FACTOR"] = "2"


def format_input(input_str: str) -> str:
    """
    Formats the input to match what is required by Claude,
    specifically, anthropic.claude-v2.
    """
    return f"Human: {input_str}\n\nAssistant:\n"


class TestSummarizationAccuracy:
    def test_evaluate_sample_and_evaluate(self, integration_tests_dir):
        """
        Instantiating a SummarizationAccuracy object will create a BertscoreHelperModel
        Ray actor. To avoid doing this twice (which would require us to shut down Ray
        in between tests due to Bert's high memory usage), we use a single SummarizationAccuracy
        object to test evaluate_sample and evaluate back-to-back (instead of following the
        convention of the other tests, where evaluate_sample and evaluate are tested in separate methods).
        """
        eval_algo = SummarizationAccuracy()
        expected_evaluate_sample_scores = {METEOR_SCORE: 0.108, ROUGE_SCORE: 0.0, BERT_SCORE: 0.608}
        expected_evaluate_scores = {METEOR_SCORE: 0.317, ROUGE_SCORE: 0.060, BERT_SCORE: 0.632}

        # Test evaluate_sample
        with open(os.path.join(integration_tests_dir, "datasets", "gigaword_sample.jsonl")) as fh:
            json_obj = json.loads(fh.readline())
            original_text = json_obj["document"]
            target_output = json_obj["summary"]
            model_input = f"Summarise the following text in one sentence: {original_text}"
            model_output = bedrock_model_runner.predict(format_input(model_input))[0]
            eval_scores = eval_algo.evaluate_sample(target_output, model_output)
            for eval_score in eval_scores:
                assert eval_score.value == approx(expected_evaluate_sample_scores[eval_score.name], abs=ABS_TOL)

        # Test evaluate
        dataset_config = DATASET_CONFIGS[GIGAWORD]
        eval_outputs = eval_algo.evaluate(
            model=bedrock_model_runner,
            dataset_config=dataset_config,
            prompt_template=format_input("Summarise the following text in one sentence: $model_input"),
            save=True,
            num_records=20,
        )
        eval_output = eval_outputs[0]
        eval_scores = eval_output.dataset_scores
        for eval_score in eval_scores:
            assert eval_score.value == approx(expected_evaluate_scores[eval_score.name], abs=ABS_TOL)
