import re
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
    """
    GIVEN valid arguments.
    WHEN a Transform is initialized.
    THEN the Transform's `transforms` attribute is the correct, flat list of Transforms.
    """
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
            err_msg=re.escape(
                "TransformPipeline contains Transforms with the same output keys as other Transforms. "
                "Here are the problematic Transforms, paired with their offending keys: "
                "{GeneratePrompt(input_keys=['input_1'], output_keys=['output_1'], "
                "args=[['input_1'], ['output_1'], '1'], kwargs={}): ['output_1']}"
            ),
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
    """
    GIVEN invalid arguments.
    WHEN a TransformPipeline is initialized.
    THEN errors with the appropriate message are raised.
    """
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
        self.register_input_output_keys(input_keys, output_keys)
        self.pos_arg = pos_arg
        self.kw_arg = kw_arg

    def __call__(self, record: Dict[str, Any]):
        input_key = self.input_keys[0]
        record[self.output_keys[0]] = f"{record[input_key]}_{self.pos_arg}_{self.kw_arg}"
        return record


def test_mutating_nested_pipelines():
    """
    GIVEN a TransformPipeline containing a child pipeline.
    WHEN the child pipeline's `transforms` list is mutated.
    THEN the parent pipeline is not affected.
    """
    child = TransformPipeline([TRANSFORM_1, TRANSFORM_2])
    parent = TransformPipeline([child, TRANSFORM_3])
    child.transforms.pop(0)
    assert parent.transforms == [TRANSFORM_1, TRANSFORM_2, TRANSFORM_3]


def test_mutating_child_transforms():
    """
    GIVEN a TransformPipeline containing a child pipeline.
    WHEN Transform objects within the child pipeline's `transforms` list are mutated.
        Note that it is bad practice to mutate Transform objects after initialization,
        and this should never be done when defining a pipeline.
    THEN the parent pipeline's `transforms` reflect the same changes.
    """
    transform_1 = GeneratePrompt(["input_1"], ["output_1"], "1")
    transform_2 = GeneratePrompt(["input_2"], ["output_2"], "2")
    transform_3 = GeneratePrompt(["input_3"], ["output_3"], "3")
    child = TransformPipeline([transform_1, transform_2])
    parent = TransformPipeline([child, transform_3])
    child.transforms[0].args = ("Hello", "there")
    assert parent.transforms[0].args == ("Hello", "there")


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


def test_execute_record():
    """
    GIVEN a valid TransformPipeline.
    WHEN its execute_record method is called on a record.
    THEN all Transforms in the TransformPipeline are applied to the input record.
    """
    pipeline = TransformPipeline(
        [DummyTransform(["input"], ["output_1"], 1, kw_arg="a"), DummyTransform(["input"], ["output_2"], 2, kw_arg="b")]
    )
    record = {"input": "asdf"}
    output_record = pipeline.execute_record(record)
    assert output_record == {"input": "asdf", "output_1": "asdf_1_a", "output_2": "asdf_2_b"}
