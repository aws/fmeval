from unittest.mock import Mock

import pytest
from typing import Any, List, NamedTuple, Dict

from fmeval.constants import TRANSFORM_PIPELINE_MAX_SIZE
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.transforms.common import GeneratePrompt
from fmeval.transforms.transform import Transform
from fmeval.transforms.transform_pipeline import TransformPipeline, NestedTransform


TRANSFORM_1 = GeneratePrompt(["input_1"], ["output_1"], "1")
TRANSFORM_2 = GeneratePrompt(["input_2"], ["output_2"], "2")
TRANSFORM_3 = GeneratePrompt(["input_3"], ["output_3"], "3")


class TestCaseInit(NamedTuple):
    transforms: List[NestedTransform]
    expected: List[Transform]


@pytest.mark.parametrize(
    "transforms, expected",
    [
        TestCaseInit(transforms=[TRANSFORM_1, TRANSFORM_2], expected=[TRANSFORM_1, TRANSFORM_2]),
        TestCaseInit(transforms=[TransformPipeline([TRANSFORM_1]), TRANSFORM_2], expected=[TRANSFORM_1, TRANSFORM_2]),
        TestCaseInit(
            transforms=[TransformPipeline([TRANSFORM_1]), TransformPipeline([TRANSFORM_2])],
            expected=[TRANSFORM_1, TRANSFORM_2],
        ),
        TestCaseInit(
            transforms=[
                TransformPipeline([TRANSFORM_1, TransformPipeline([TRANSFORM_2, TransformPipeline([TRANSFORM_3])])])
            ],
            expected=[TRANSFORM_1, TRANSFORM_2, TRANSFORM_3],
        ),
    ],
)
def test_init_success(transforms, expected):
    pipeline = TransformPipeline(transforms)
    assert pipeline.transforms == expected


class TestCaseInitFailure(NamedTuple):
    transforms: Any
    err_msg: str


@pytest.mark.parametrize(
    "transforms, err_msg",
    [
        TestCaseInitFailure(
            transforms=TRANSFORM_1,
            err_msg="TransformPipeline initializer accepts a list containing Transforms or TransformPipelines.",
        ),
        TestCaseInitFailure(
            transforms=[TRANSFORM_1, TRANSFORM_2, 123], err_msg="nested_transform has type <class 'int'>"
        ),
        TestCaseInitFailure(
            transforms=[TRANSFORM_1, TRANSFORM_1],
            err_msg="TransformPipeline contains Transforms with the same output keys as other Transforms.",
        ),
        TestCaseInitFailure(
            transforms=[
                GeneratePrompt([f"input_{i}"], [f"output_{i}"], str(i)) for i in range(TRANSFORM_PIPELINE_MAX_SIZE + 1)
            ],
            err_msg=f"TransformPipeline initialized with {TRANSFORM_PIPELINE_MAX_SIZE + 1} Transforms.",
        ),
    ],
)
def test_init_failure(transforms, err_msg):
    with pytest.raises(EvalAlgorithmClientError, match=err_msg):
        TransformPipeline(transforms)


class DummyTransform(Transform):
    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        pos_arg: int,
        kw_arg: str = "Hi",
    ):
        super().__init__(input_keys, output_keys, pos_arg, kw_arg=kw_arg)

    def __call__(self, record: Dict[str, Any]):
        return record


def test_execute():
    """
    GIVEN a valid TransformPipeline.
    WHEN its execute method is called on a dataset.
    THEN the Ray Dataset `map` method is called with the correct arguments.
    """
    pipeline = TransformPipeline([DummyTransform(["input"], ["output"], 42, "Hello there")])
    dataset = Mock()
    dataset.map = Mock()
    pipeline.execute(dataset)
    dataset.map.assert_called_once_with(
        DummyTransform,
        fn_constructor_args=(
            ["input"],
            ["output"],
            42,
        ),
        fn_constructor_kwargs={"kw_arg": "Hello there"},
        num_cpus=(1 / TRANSFORM_PIPELINE_MAX_SIZE),
        concurrency=1.0,
    )
