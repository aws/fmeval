import os
from pytest import approx
from fmeval.eval_algorithms.qa_accuracy import (
    QAAccuracy,
    QAAccuracyConfig,
    F1_SCORE,
    EXACT_MATCH_SCORE,
    QUASI_EXACT_MATCH_SCORE,
    PRECISION_OVER_WORDS,
    RECALL_OVER_WORDS,
    BERT_SCORE,
)
from fmeval.transforms.qa_accuracy_metrics import BERT_SCORE

from fmeval.data_loaders.data_config import DataConfig
from fmeval.constants import MIME_TYPE_JSONLINES
from test.integration.models.model_runners import bedrock_model_runner
from test.integration.test_summarization_accuracy import format_input


ABS_TOL = 5e-2  # Bedrock models are not deterministic, so we use a higher tolerance here
os.environ["PARALLELIZATION_FACTOR"] = "2"

config = QAAccuracyConfig("<OR>")
eval_algo = QAAccuracy(config)

js_model_runner_prompt_template = """
    Answer the question at the end in as few words as possible.
    Do not repeat the question. Do not answer in complete sentences. <</SYS>>
    Question: $model_input
    """


class TestQAAccuracy:
    def test_evaluate_sample(self):
        model_input = """
            Answer the question at the end in as few words as possible.
            Do not repeat the question. Do not answer in complete sentences.
            Question: London is the capital of
            """
        model_output = bedrock_model_runner.predict(format_input(model_input))[0]
        eval_scores = eval_algo.evaluate_sample(
            target_output="UK<OR>England<OR>United Kingdom", model_output=model_output
        )
        for eval_score in eval_scores:
            if eval_score.name == BERT_SCORE:
                assert eval_score.value == approx(1.0, abs=ABS_TOL)
            else:
                assert eval_score.value == 1.0

    def test_evaluate(self, integration_tests_dir):
        dataset_config = DataConfig(
            dataset_name="triviaQA_sample",
            dataset_uri=os.path.join(integration_tests_dir, "datasets", "triviaQA_sample.jsonl"),
            dataset_mime_type=MIME_TYPE_JSONLINES,
            model_input_location="question",
            target_output_location="answer",
        )
        eval_output = eval_algo.evaluate(
            model=bedrock_model_runner,
            dataset_config=dataset_config,
            prompt_template=format_input(js_model_runner_prompt_template),
            save=True,
            # might take a longer time to run, so thinking about changing to 20 records
            # just like in test_summarization_accuracy.py
        )[0]
        for eval_score in eval_output.dataset_scores:
            if eval_score.name == F1_SCORE:  # pragma: no branch
                assert eval_score.value == approx(0.6588103254769923, abs=ABS_TOL)
            elif eval_score.name == EXACT_MATCH_SCORE:
                assert eval_score.value == approx(0.5858585858585859, abs=ABS_TOL)
            elif eval_score.name == QUASI_EXACT_MATCH_SCORE:
                assert eval_score.value == approx(0.6060606060606061, abs=ABS_TOL)
            elif eval_score.name == PRECISION_OVER_WORDS:
                assert eval_score.value == approx(0.6666666666666666, abs=ABS_TOL)
            elif eval_score.name == RECALL_OVER_WORDS:
                assert eval_score.value == approx(0.6696969696969697, abs=ABS_TOL)
            elif eval_score.name == BERT_SCORE:
                assert eval_score.value == approx(0.8692791720833442, abs=ABS_TOL)
