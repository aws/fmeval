import os
from pytest import approx
from amazon_fmeval.eval_algorithms.qa_accuracy_semantic_robustness import (
    QAAccuracySemanticRobustness,
    QAAccuracySemanticRobustnessConfig,
    DELTA_F1_SCORE,
    DELTA_EXACT_MATCH_SCORE,
    DELTA_QUASI_EXACT_MATCH_SCORE,
)
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.constants import MIME_TYPE_JSONLINES
from ..test_model_runners import sm_model_runner, sm_model_runner_prompt_template

ABS_TOL = 1e-3
os.environ["PARALLELIZATION_FACTOR"] = "2"

config = QAAccuracySemanticRobustnessConfig("<OR>")
eval_algo = QAAccuracySemanticRobustness(config)


class TestQAAccuracySemanticRobustness:
    def test_evaluate_sample(self):
        model_input = "London is the capital of"
        eval_scores = eval_algo.evaluate_sample(
            model_input=model_input,
            model=sm_model_runner,
            target_output="UK<OR>England<OR>United Kingdom",
            prompt_template=sm_model_runner_prompt_template,
        )
        for eval_score in eval_scores:
            assert eval_score.value == approx(0.8, abs=ABS_TOL)

    def test_evaluate(self, integration_tests_dir):
        dataset_config = DataConfig(
            dataset_name="triviaQA_sample",
            dataset_uri=os.path.join(integration_tests_dir, "datasets", "triviaQA_sample.jsonl"),
            dataset_mime_type=MIME_TYPE_JSONLINES,
            model_input_location="question",
            target_output_location="answer",
        )
        eval_output = eval_algo.evaluate(
            model=sm_model_runner,
            dataset_config=dataset_config,
            prompt_template=sm_model_runner_prompt_template,
            save=True,
        )[0]
        for eval_score in eval_output.dataset_scores:
            if eval_score.name == DELTA_F1_SCORE:
                assert eval_score.value == approx(0.188, abs=ABS_TOL)
            elif eval_score.name == DELTA_EXACT_MATCH_SCORE:
                assert eval_score.value == approx(0.038, abs=ABS_TOL)
            elif eval_score.name == DELTA_QUASI_EXACT_MATCH_SCORE:
                assert eval_score.value == approx(0.186, abs=ABS_TOL)
