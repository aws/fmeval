import os
from typing import NamedTuple, Dict
import logging

import pytest
from pytest import approx

from fmeval.eval_algorithms import EvalScore, CategoryScore
from fmeval.eval_algorithms.factual_knowledge import (
    FactualKnowledge,
    FactualKnowledgeConfig,
    EXACT_INCLUSION,
    QUASI_EXACT_INCLUSION,
)
from fmeval.data_loaders.data_config import DataConfig
from fmeval.constants import MIME_TYPE_JSONLINES
from test.integration.models.model_runners import hf_model_runner

ABS_TOL = 1e-4
os.environ["PARALLELIZATION_FACTOR"] = "2"

config = FactualKnowledgeConfig("<OR>")
eval_algo = FactualKnowledge(config)

logger = logging.getLogger(__name__)


class TestFactualKnowledge:
    def test_evaluate_sample(self):
        model_output = hf_model_runner.predict("London is the capital of")[0]
        eval_scores = eval_algo.evaluate_sample(
            target_output="UK<OR>England<OR>United Kingdom", model_output=model_output
        )
        # the model produces deterministic output
        for eval_score in eval_scores:
            if eval_score.name == EXACT_INCLUSION:
                assert eval_score.value == 1.0
            elif eval_score.name == QUASI_EXACT_INCLUSION:
                assert eval_score.value == 1.0

    class EvaluateTestCase(NamedTuple):
        dataset_name: str
        dataset_score: Dict[str, float]
        category_scores: Dict[str, Dict[str, float]]

    DATASET_SCORES = [
        EvalScore(name=EXACT_INCLUSION, value=0.547),
        EvalScore(name=QUASI_EXACT_INCLUSION, value=0.547),
    ]

    CATEGORY_SCORES = [
        CategoryScore(
            name="Capitals",
            scores=[
                EvalScore(name=EXACT_INCLUSION, value=0.09),
                EvalScore(name=QUASI_EXACT_INCLUSION, value=0.09),
            ],
        ),
        CategoryScore(
            name="Subsidiary",
            scores=[
                EvalScore(name=EXACT_INCLUSION, value=0.0198),
                EvalScore(name=QUASI_EXACT_INCLUSION, value=0.0198),
            ],
        ),
    ]

    @pytest.mark.parametrize(
        "test_case",
        [
            EvaluateTestCase(
                dataset_name="trex_sample.jsonl",
                dataset_score={EXACT_INCLUSION: 0.0547, QUASI_EXACT_INCLUSION: 0.0547},
                category_scores={
                    "Capitals": {EXACT_INCLUSION: 0.09, QUASI_EXACT_INCLUSION: 0.09},
                    "Subsidiary": {EXACT_INCLUSION: 0.0198, QUASI_EXACT_INCLUSION: 0.0198},
                },
            ),
            # The purpose of testing evaluate() on this tiny dataset is to
            # ensure that no issues arise even when the dataset
            # has a very small number of rows. Specifically, issues caused
            # by inconsistent batch formats in the Ray task graph.
            # See https://github.com/ray-project/ray/pull/39960
            EvaluateTestCase(
                dataset_name="trex_sample_small.jsonl",
                dataset_score={EXACT_INCLUSION: 0.0, QUASI_EXACT_INCLUSION: 0.0},
                category_scores={"Capitals": {EXACT_INCLUSION: 0.0, QUASI_EXACT_INCLUSION: 0.0}},
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
            prompt_template="$model_input",
            save=True,
        )
        eval_output = eval_outputs[0]

        for eval_score in eval_output.dataset_scores:
            # pragma: no branch
            if eval_score.name == EXACT_INCLUSION:
                assert eval_score.value == approx(test_case.dataset_score[EXACT_INCLUSION], abs=ABS_TOL)
            elif eval_score.name == QUASI_EXACT_INCLUSION:
                assert eval_score.value == approx(test_case.dataset_score[QUASI_EXACT_INCLUSION], abs=ABS_TOL)

        for category_score in eval_output.category_scores:  # pragma: no branch
            for eval_score in category_score.scores:
                if eval_score.name == EXACT_INCLUSION:
                    assert eval_score.value == approx(
                        test_case.category_scores[category_score.name][EXACT_INCLUSION], abs=ABS_TOL
                    )
                elif eval_score.name == QUASI_EXACT_INCLUSION:
                    assert eval_score.value == approx(
                        test_case.category_scores[category_score.name][QUASI_EXACT_INCLUSION], abs=ABS_TOL
                    )

    def test_evaluate_multi_datasets(self, integration_tests_dir):
        dataset_config = [
            DataConfig(
                dataset_name="TREXbig",
                dataset_uri=os.path.join(integration_tests_dir, "datasets", "trex_sample.jsonl"),
                dataset_mime_type=MIME_TYPE_JSONLINES,
                model_input_location="question",
                target_output_location="answers",
                category_location="knowledge_category",
            ),
            DataConfig(
                dataset_name="TREXsmall",
                dataset_uri=os.path.join(integration_tests_dir, "datasets", "trex_sample_small.jsonl"),
                dataset_mime_type=MIME_TYPE_JSONLINES,
                model_input_location="question",
                target_output_location="answers",
                category_location="knowledge_category",
            ),
        ]
        eval_outputs = eval_algo.evaluate(
            model=hf_model_runner,
            dataset_config=dataset_config,
            prompt_template="$model_input",
            save=True,
        )
        assert len(eval_outputs) == len(dataset_config)
        assert [eo.dataset_name for eo in eval_outputs] == [dc.dataset_name for dc in dataset_config]
