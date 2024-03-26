import pytest
from typing import List, NamedTuple, Optional
from contextlib import nullcontext as does_not_raise
from fmeval.eval_algorithms import EvalOutput, CategoryScore, EvalScore, get_default_prompt_template


class TestEvalOutput:
    class TestEvalOutputParameters(NamedTuple):
        eval_name: str
        dataset_name: str
        prompt_template: str
        dataset_scores: Optional[List[EvalScore]] = None
        category_scores: Optional[List[CategoryScore]] = None
        output_path: Optional[str] = None
        error: Optional[str] = None

    @pytest.mark.parametrize(
        "eval_output_parameters, expectation",
        [
            (
                TestEvalOutputParameters(
                    eval_name="toxicity",
                    dataset_name="real_toxicity_prompts",
                    prompt_template="toxicity$features",
                    dataset_scores=[EvalScore(name="toxicity", value=0.5), EvalScore(name="obscene", value=0.8)],
                ),
                does_not_raise(),
            ),
            (
                TestEvalOutputParameters(
                    eval_name="toxicity",
                    dataset_name="real_toxicity_prompts",
                    prompt_template="toxicity$features",
                    dataset_scores=[EvalScore(name="toxicity", value=0.5), EvalScore(name="obscene", value=0.8)],
                    category_scores=[
                        CategoryScore(
                            name="Gender",
                            scores=[EvalScore(name="toxicity", value=0.5), EvalScore(name="obscene", value=0.8)],
                        ),
                        CategoryScore(
                            name="Race",
                            scores=[EvalScore(name="toxicity", value=0.3), EvalScore(name="obscene", value=0.4)],
                        ),
                    ],
                ),
                does_not_raise(),
            ),
            (
                TestEvalOutputParameters(
                    eval_name="toxicity",
                    dataset_name="real_toxicity_prompts",
                    prompt_template="toxicity$features",
                    dataset_scores=[EvalScore(name="toxicity", value=0.5), EvalScore(name="obscene", value=0.8)],
                    category_scores=[
                        CategoryScore(name="Gender", scores=[EvalScore(name="toxicity", value=0.5)]),
                        CategoryScore(
                            name="Race",
                            scores=[EvalScore(name="toxicity", value=0.3), EvalScore(name="obscene", value=0.4)],
                        ),
                    ],
                ),
                pytest.raises(AssertionError),
            ),
            (
                TestEvalOutputParameters(
                    eval_name="toxicity",
                    dataset_name="real_toxicity_prompts",
                    prompt_template="toxicity$features",
                    dataset_scores=[EvalScore(name="toxicity", value=0.5), EvalScore(name="obscene", value=0.8)],
                    category_scores=[
                        CategoryScore(
                            name="Gender",
                            scores=[EvalScore(name="toxicity", value=0.5), EvalScore(name="obscene", value=0.8)],
                        ),
                        CategoryScore(
                            name="Race",
                            scores=[EvalScore(name="severe_toxicity", value=0.3), EvalScore(name="obscene", value=0.4)],
                        ),
                    ],
                ),
                pytest.raises(AssertionError),
            ),
            (
                TestEvalOutputParameters(
                    eval_name="toxicity",
                    dataset_name="real_toxicity_prompts",
                    prompt_template="toxicity$features",
                    dataset_scores=None,
                    error="Toxicity eval error message.",
                ),
                does_not_raise(),
            ),
            (
                TestEvalOutputParameters(
                    eval_name="toxicity",
                    dataset_name="real_toxicity_prompts",
                    prompt_template="toxicity$features",
                    dataset_scores=None,
                    error=None,
                ),
                pytest.raises(AssertionError),
            ),
        ],
    )
    def test_eval_output(self, eval_output_parameters, expectation):
        with expectation:
            EvalOutput(
                eval_name=eval_output_parameters.eval_name,
                dataset_name=eval_output_parameters.dataset_name,
                prompt_template=eval_output_parameters.prompt_template,
                dataset_scores=eval_output_parameters.dataset_scores,
                category_scores=eval_output_parameters.category_scores,
                output_path=eval_output_parameters.output_path,
                error=eval_output_parameters.error,
            )

    def test_equality(self):
        assert EvalOutput(
            dataset_name="dataset1",
            eval_name="toxicity",
            output_path="/output/path",
            prompt_template="$model_input",
            dataset_scores=[EvalScore(name="score1", value=0.7), EvalScore("score2", 0.2)],
            category_scores=[
                CategoryScore(name="cat1", scores=[EvalScore("score1", 1), EvalScore("score2", 2)]),
                CategoryScore(name="cat2", scores=[EvalScore("score1", 1), EvalScore("score2", 2)]),
            ],
        ) == EvalOutput(
            dataset_name="dataset1",
            eval_name="toxicity",
            output_path="/output/path",
            prompt_template="$model_input",
            dataset_scores=[EvalScore(name="score1", value=0.7), EvalScore("score2", 0.2)],
            category_scores=[
                CategoryScore(name="cat2", scores=[EvalScore("score1", 1), EvalScore("score2", 2)]),
                CategoryScore(name="cat1", scores=[EvalScore("score1", 1), EvalScore("score2", 2)]),
            ],
        )

    def test_inequality(self):
        assert EvalOutput(
            dataset_name="dataset1",
            eval_name="toxicity",
            output_path="/output/path",
            prompt_template="$model_input",
            dataset_scores=[EvalScore(name="score1", value=0.7), EvalScore("score2", 0.2)],
            category_scores=[
                CategoryScore(name="cat1", scores=[EvalScore("score1", 1), EvalScore("score2", 2)]),
                CategoryScore(name="cat2", scores=[EvalScore("score1", 1), EvalScore("score2", 2)]),
            ],
        ) != EvalOutput(
            dataset_name="dataset2",
            eval_name="toxicity",
            output_path="/output/path",
            prompt_template="$model_input",
            dataset_scores=[EvalScore(name="score1", value=0.7), EvalScore("score2", 0.2)],
            category_scores=[
                CategoryScore(name="cat2", scores=[EvalScore("score1", 1), EvalScore("score2", 2)]),
                CategoryScore(name="cat1", scores=[EvalScore("score1", 1), EvalScore("score2", 2)]),
            ],
        )


class TestCategoryScore:
    def test_equality(self):
        assert CategoryScore(name="name1", scores=[EvalScore("score1", 1), EvalScore("score2", 2)]) == CategoryScore(
            name="name1", scores=[EvalScore("score2", 2), EvalScore("score1", 1)]
        )

    def test_inequality(self):
        assert CategoryScore(name="name1", scores=[EvalScore("score1", 1), EvalScore("score2", 2)]) != CategoryScore(
            name="name2", scores=[EvalScore("score2", 2), EvalScore("score1", 1)]
        )
        assert CategoryScore(name="name1", scores=[EvalScore("score1", 1), EvalScore("score2", 2)]) != CategoryScore(
            name="name2", scores=[EvalScore("score2", 2), EvalScore("score1", 2)]
        )
        assert CategoryScore(name="name1", scores=[EvalScore("score1", 1), EvalScore("score2", 2)]) != CategoryScore(
            name="name1", scores=[EvalScore("score1", 1), EvalScore("score2", 2), EvalScore(name="score1", value=1)]
        )


class TestEvalScore:
    def test_equality(self):
        assert EvalScore(name="name1", value=0.7) == EvalScore(name="name1", value=0.7)
        assert EvalScore(name="name1", value=0.7) == EvalScore(name="name1", value=0.6999999999999)

    def test_inequality(self):
        assert EvalScore(name="name1", value=0.7) != EvalScore(name="name2", value=0.7)
        assert EvalScore(name="name1", value=0.7) != EvalScore(name="name1", value=0.69)


def test_get_default_prompt_template():
    """
    GIVEN a dataset name
    WHEN get_default_prompt_template() method is called
    THEN expected default prompt template is returned
    """
    assert (
        get_default_prompt_template("trivia_qa")
        == "Respond to the following question with a short answer: $model_input"
    )
    assert get_default_prompt_template("my_custom_dataset") == "$model_input"
