import os
from pytest import approx
from test.integration.hf_model_runner import HFModelConfig, HuggingFaceCausalLLMModelRunner
from amazon_fmeval import get_eval_algorithm
from amazon_fmeval.eval_algorithms.factual_knowledge import FactualKnowledgeConfig
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.constants import MIME_TYPE_JSONLINES

ABS_TOL = 1e-4
os.environ["PARALLELIZATION_FACTOR"] = "2"

hf_config = HFModelConfig(model_name="gpt2", max_new_tokens=32)
model = HuggingFaceCausalLLMModelRunner(model_config=hf_config)
factual_knowledge_config = FactualKnowledgeConfig("<OR>")
factual_knowledge_algo = get_eval_algorithm("factual_knowledge")(factual_knowledge_config)


class TestFactualKnowledge:
    test_dir = "factual_knowledge"

    def test_evaluate_sample(self):
        model_output = model.predict("London is the capital of")[0]
        eval_score = factual_knowledge_algo.evaluate_sample(
            target_output="UK<OR>England<OR>United Kingdom", model_output=model_output
        )[0]
        assert eval_score.value == 1  # the model produces deterministic output

    def test_evaluate(self, integration_tests_dir):
        dataset_config = DataConfig(
            dataset_name="TREX",
            dataset_uri=os.path.join(integration_tests_dir, self.test_dir, "datasets", "trex_sample.jsonl"),
            dataset_mime_type=MIME_TYPE_JSONLINES,
            model_input_location="question",
            target_output_location="answers",
            category_location="knowledge_category",
        )
        eval_outputs = factual_knowledge_algo.evaluate(
            model=model,
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
