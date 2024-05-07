import re
from typing import NamedTuple, Union, List, Dict, Optional

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
    class TestCaseCompose(NamedTuple):
        template: str
        prompt: Optional[str]
        kwargs: Dict
        expected_result: str

    @pytest.mark.parametrize(
        "test_case",
        [
            # Test case to verify composing a prompt with `data`
            TestCaseCompose(
                template="Answer the following question: $model_input",
                prompt="London is the capital of?",
                kwargs={},
                expected_result="Answer the following question: London is the capital of?",
            ),
            # Test case verify composing a prompt with multiple keyword arguments
            TestCaseCompose(
                template="Question: $model_input \n context: $context \n statement: $statements",
                prompt=None,
                kwargs={"model_input": "sample question", "context": "sample context", "statements": "statement1"},
                expected_result="Question: sample question \n context: sample context \n statement: statement1",
            ),
            # Test case verify composing a prompt with keyword argument takes higher priority
            TestCaseCompose(
                template="Question: $model_input",
                prompt="question from prompt",
                kwargs={"model_input": "question from kwargs"},
                expected_result="Question: question from kwargs",
            ),
            # Test case verify composing a prompt with both `data` and keyword arguments
            TestCaseCompose(
                template="Question: $model_input \n Context: $context",
                prompt="question from prompt",
                kwargs={"context": "some context"},
                expected_result="Question: question from prompt \n Context: some context",
            ),
        ],
    )
    def test_compose(self, test_case):
        composer = PromptComposer(template=test_case.template)
        result = composer.compose(test_case.prompt, **test_case.kwargs)
        assert result == test_case.expected_result

    def test_invalid_template(self):
        composer = PromptComposer(template="Answer the following question: $invalid")
        prompt = "London is the capital of?"
        with pytest.raises(KeyError):
            composer.compose(prompt)
