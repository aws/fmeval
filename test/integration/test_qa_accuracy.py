import os
from pytest import approx
from amazon_fmeval.eval_algorithms.qa_accuracy import (
    QAAccuracy,
    QAAccuracyConfig,
    F1_SCORE,
    EXACT_MATCH_SCORE,
    QUASI_EXACT_MATCH_SCORE,
)
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.constants import MIME_TYPE_JSONLINES
from test.integration.models.model_runners import js_model_runner, js_model_runner_prompt_template

ABS_TOL = 1e-6  # scores and model are deterministic, so approx() should be used purely to handle floating point error
os.environ["PARALLELIZATION_FACTOR"] = "2"

config = QAAccuracyConfig("<OR>")
eval_algo = QAAccuracy(config)


class TestQAAccuracy:
    def test_evaluate_sample(self):
        model_input = """
            <s>[INST] <<SYS>>Answer the question at the end in as few words as possible.
            Do not repeat the question. Do not answer in complete sentences.<</SYS>>
            Question: London is the capital of [/INST]
            """
        model_output = js_model_runner.predict(model_input)[0]
        eval_scores = eval_algo.evaluate_sample(
            target_output="UK<OR>England<OR>United Kingdom", model_output=model_output
        )
        for eval_score in eval_scores:
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
            model=js_model_runner,
            dataset_config=dataset_config,
            prompt_template=js_model_runner_prompt_template,
            save=True,
        )[0]
        for eval_score in eval_output.dataset_scores:
            if eval_score.name == F1_SCORE:  # pragma: no branch
                assert eval_score.value == approx(0.360630, abs=ABS_TOL)
            elif eval_score.name == EXACT_MATCH_SCORE:
                assert eval_score.value == approx(0.060606, abs=ABS_TOL)
            elif eval_score.name == QUASI_EXACT_MATCH_SCORE:
                assert eval_score.value == approx(0.303030, abs=ABS_TOL)
