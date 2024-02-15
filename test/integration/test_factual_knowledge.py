import os
import ray
from typing import NamedTuple, Dict

import pytest
from pytest import approx
from fmeval.eval_algorithms.factual_knowledge import FactualKnowledge, FactualKnowledgeConfig
from fmeval.data_loaders.data_config import DataConfig
from fmeval.constants import MIME_TYPE_JSONLINES
from test.integration.models.model_runners import hf_model_runner

ABS_TOL = 1e-4
os.environ["PARALLELIZATION_FACTOR"] = "2"

config = FactualKnowledgeConfig("<OR>")
eval_algo = FactualKnowledge(config)


class TestFactualKnowledge:
    def test_evaluate_sample(self):
        model_output = hf_model_runner.predict("London is the capital of")[0]
        eval_score = eval_algo.evaluate_sample(
            target_output="UK<OR>England<OR>United Kingdom", model_output=model_output
        )[0]
        assert eval_score.value == 1  # the model produces deterministic output

    class EvaluateTestCase(NamedTuple):
        dataset_name: str
        dataset_score: float
        category_scores: Dict[str, float]

    @pytest.mark.parametrize(
        "test_case",
        [
            EvaluateTestCase(
                dataset_name="trex_sample.jsonl",
                dataset_score=0.0547,
                category_scores={"Capitals": 0.09, "Subsidiary": 0.0198},
            ),
            # The purpose of testing evaluate() on this tiny dataset is to
            # ensure that no issues arise even when the dataset
            # has a very small number of rows. Specifically, issues caused
            # by inconsistent batch formats in the Ray task graph.
            # See https://github.com/ray-project/ray/pull/39960
            EvaluateTestCase(
                dataset_name="trex_sample_small.jsonl", dataset_score=0.0, category_scores={"Capitals": 0.0}
            ),
        ],
    )
    def test_evaluate(self, integration_tests_dir, test_case):
        dataset_config = DataConfig(
            dataset_name="TREX",
            dataset_uri=os.path.join(integration_tests_dir, "datasets", test_case.dataset_name),
            dataset_mime_type=MIME_TYPE_JSONLINES,
            model_input_location="question",
            target_output_location="answers",
            category_location="knowledge_category",
        )
        eval_outputs = eval_algo.evaluate(
            model=hf_model_runner,
            dataset_config=dataset_config,
            prompt_template="$feature",
            save=True,
        )
        eval_output = eval_outputs[0]
        assert eval_output.dataset_scores[0].value == approx(test_case.dataset_score, abs=ABS_TOL)
        for category_score in eval_output.category_scores:  # pragma: no branch
            assert category_score.scores[0].value == approx(test_case.category_scores[category_score.name], abs=ABS_TOL)

    def test_ray_shutdown(self):
        """
        Forcefully shut down Ray to ensure that resources
        used by these tests get freed.
        """
        ray.shutdown()
