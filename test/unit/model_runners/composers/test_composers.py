import re
import pytest
from amazon_fmeval.model_runners.composers.composers import (
    JsonContentComposer,
    StringContentComposer,
    PromptComposer,
    EvalAlgorithmClientError,
)


class TestJsonContentComposer:
    def test_compose(self):
        composer = JsonContentComposer(template='{"data":$prompt}')
        data = '["John",40]'
        expected_result = {"data": '["John",40]'}
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


class TestStringContentComposer:
    def test_compose(self):
        composer = StringContentComposer(template="$prompt")
        data = "London is the capital of?"
        expected_result = "London is the capital of?"
        result = composer.compose(data)
        assert result == expected_result

    def test_invalid_template(self):
        composer = StringContentComposer(template="$invalid")
        data = "London is the capital of?"
        with pytest.raises(KeyError):
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
