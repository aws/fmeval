import os
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

    def test_evaluate(self, integration_tests_dir):
        dataset_config = DataConfig(
            dataset_name="TREX",
            dataset_uri=os.path.join(integration_tests_dir, "datasets", "trex_sample.jsonl"),
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
        assert eval_output.dataset_scores[0].value == approx(0.0547, abs=ABS_TOL)
        for category_score in eval_output.category_scores:  # pragma: no branch
            if category_score.name == "Capitals":
                assert category_score.scores[0].value == approx(0.09, abs=ABS_TOL)
            elif category_score.name == "Subsidiary":
                assert category_score.scores[0].value == approx(0.0198, abs=ABS_TOL)

    def test_ray_shutdown(self):
        """
        Forcefully shut down the Ray session to ensure that resources
        consumed by this session get freed.
        """
        ray.shutdown()
