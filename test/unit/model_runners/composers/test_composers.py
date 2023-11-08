import re
from typing import NamedTuple, Union, List, Dict

import pytest

from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.composers.composers import (
    JsonContentComposer,
    PromptComposer,
)


class TestJsonContentComposer:
    class TestCaseCompose(NamedTuple):
        template: str
        data: str
        expected_result: Union[str, List, Dict]

    @pytest.mark.parametrize(
        "template, data, expected_result",
        [
            TestCaseCompose(
                template="$prompt",
                data="hello there",
                expected_result="hello there",
            ),
            TestCaseCompose(
                template='{"data":$prompt}',
                data='["John",40]',
                expected_result={"data": '["John",40]'},
            ),
            TestCaseCompose(
                template="[$prompt, $prompt]", data="hello there", expected_result=["hello there", "hello there"]
            ),
        ],
    )
    def test_compose(self, template, data, expected_result):
        composer = JsonContentComposer(template=template)
        result = composer.compose(data)
        assert result == expected_result

    def test_invalid_template(self):
        composer = JsonContentComposer(template='{"data":$invalid}')
        data = '["John",40]'
        with pytest.raises(
            EvalAlgorithmClientError,
            match=re.escape(
                "('Unable to load a JSON object with template \\'{\"data\":$invalid}\\' using data [\"John\",40] ', KeyError('invalid'))"
            ),
        ):
            composer.compose(data)


class TestPromptComposer:
    def test_compose(self):
        composer = PromptComposer(template="Answer the following question: $feature")
        prompt = "London is the capital of?"
        expected_result = "Answer the following question: London is the capital of?"
        result = composer.compose(prompt)
        assert result == expected_result

    def test_invalid_template(self):
        composer = PromptComposer(template="Answer the following question: $invalid")
        prompt = "London is the capital of?"
        with pytest.raises(KeyError):
            composer.compose(prompt)
