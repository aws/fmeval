import pytest
from typing import NamedTuple
from amazon_fmeval.eval_algorithms import EvalScore, CategoryScore
from amazon_fmeval.constants import ABS_TOL


class TestDataClasses:
    class TestCaseEvalScore(NamedTuple):
        eval_score_1: EvalScore
        eval_score_2: EvalScore
        expected: bool

    @pytest.mark.parametrize(
        "eval_score_1, eval_score_2, expected",
        [
            TestCaseEvalScore(
                eval_score_1=EvalScore(name="my_score", value=0.42),
                eval_score_2=EvalScore(name="my_score", value=0.42),
                expected=True,
            ),
            TestCaseEvalScore(
                eval_score_1=EvalScore(name="my_score", value=0.42),
                eval_score_2=EvalScore(name="my_score", value=0.42 + ABS_TOL),
                expected=True,
            ),
            TestCaseEvalScore(
                eval_score_1=EvalScore(name="my_score", value=0.42),
                eval_score_2=EvalScore(name="my_score", value=0.42 - ABS_TOL),
                expected=True,
            ),
            TestCaseEvalScore(
                eval_score_1=EvalScore(name="name_1", value=0.42),
                eval_score_2=EvalScore(name="name_2", value=0.42),
                expected=False,
            ),
            TestCaseEvalScore(
                eval_score_1=EvalScore(name="my_score", value=0.42),
                eval_score_2=EvalScore(name="my_score", value=0.42 + 2 * ABS_TOL),
                expected=False,
            ),
            TestCaseEvalScore(
                eval_score_1=EvalScore(name="my_score", value=0.42),
                eval_score_2=EvalScore(name="my_score", value=0.42 - 2 * ABS_TOL),
                expected=False,
            ),
        ],
    )
    def test_eval_score_eq(self, eval_score_1, eval_score_2, expected):
        """
        Given two EvalScore objects
        WHEN __eq__ is called
        THEN __eq__ returns the correct value
        """
        assert (eval_score_1 == eval_score_2) == expected

    class TestCaseCategoryScore(NamedTuple):
        category_score_1: CategoryScore
        category_score_2: CategoryScore
        expected: bool

    @pytest.mark.parametrize(
        "category_score_1, category_score_2, expected",
        [
            # CategoryScores are identical
            TestCaseCategoryScore(
                category_score_1=CategoryScore(
                    name="category_name",
                    scores=[EvalScore(name="eval_1", value=0.42), EvalScore(name="eval_2", value=0.162)],
                ),
                category_score_2=CategoryScore(
                    name="category_name",
                    scores=[EvalScore(name="eval_1", value=0.42), EvalScore(name="eval_2", value=0.162)],
                ),
                expected=True,
            ),
            # `scores` list order differs
            TestCaseCategoryScore(
                category_score_1=CategoryScore(
                    name="category_name",
                    scores=[
                        EvalScore(name="eval_1", value=0.42),
                        EvalScore(name="eval_2", value=0.162),
                    ],
                ),
                category_score_2=CategoryScore(
                    name="category_name",
                    scores=[
                        EvalScore(name="eval_2", value=0.162),
                        EvalScore(name="eval_1", value=0.42),
                    ],
                ),
                expected=True,
            ),
            # EvalScore values differ, but within math.isclose tolerance
            TestCaseCategoryScore(
                category_score_1=CategoryScore(
                    name="category_name",
                    scores=[EvalScore(name="eval_1", value=0.42), EvalScore(name="eval_2", value=0.162)],
                ),
                category_score_2=CategoryScore(
                    name="category_name",
                    scores=[
                        EvalScore(name="eval_1", value=0.42 + ABS_TOL),
                        EvalScore(name="eval_2", value=0.162 - ABS_TOL),
                    ],
                ),
                expected=True,
            ),
            # CategoryScore names differ
            TestCaseCategoryScore(
                category_score_1=CategoryScore(
                    name="category_1",
                    scores=[EvalScore(name="eval_1", value=0.42), EvalScore(name="eval_2", value=0.162)],
                ),
                category_score_2=CategoryScore(
                    name="category_2",
                    scores=[EvalScore(name="eval_1", value=0.42), EvalScore(name="eval_2", value=0.162)],
                ),
                expected=False,
            ),
            # at least one EvalScore differs beyond allowed tolerance
            TestCaseCategoryScore(
                category_score_1=CategoryScore(
                    name="category_name",
                    scores=[EvalScore(name="eval_1", value=0.42), EvalScore(name="eval_2", value=0.162)],
                ),
                category_score_2=CategoryScore(
                    name="category_name",
                    scores=[EvalScore(name="eval_1", value=0.42), EvalScore(name="eval_2", value=0.162 + 2 * ABS_TOL)],
                ),
                expected=False,
            ),
        ],
    )
    def test_category_score_eq(self, category_score_1, category_score_2, expected):
        assert (category_score_1 == category_score_2) == expected
