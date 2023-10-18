import re

import pytest

from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.composers.composers import JsonContentComposer, PromptComposer


class TestContentComposerComposer:
    # Fixture to create a SingleRequestComposer instance with a predefined template
    @pytest.fixture
    def composer(self):
        template = '{"data":$prompt}'
        return JsonContentComposer(template)

    # Test case to verify composing a single prompt
    @pytest.mark.parametrize(
        "prompts, expected_result",
        [('["John",40]', {"data": '["John",40]'})],
    )
    def test_compose_single_prompt(self, composer, prompts, expected_result):
        result = composer.compose(prompts)
        assert result == expected_result

    # Test case to verify error is raised for an invalid template
    def test_invalid_template(self):
        template = '{"data":$invalid}'
        composer = JsonContentComposer(template)
        prompt = '["John",40]'
        with pytest.raises(
            EvalAlgorithmClientError,
            match=re.escape(
                "('Unable to load a JSON object with content_template \\'{\"data\":$invalid}\\' for prompt [\"John\",40] ', KeyError('invalid'))"
            ),
        ):
            composer.compose(prompt)


class TestPromptComposer:
    # Fixture to create a SingleRequestComposer instance with a predefined template
    @pytest.fixture
    def composer(self):
        template = '{"data":$feature}'
        return PromptComposer(template)

    # Test case to verify composing a single prompt
    @pytest.mark.parametrize(
        "prompts, expected_result",
        [('["John",40]', '{"data":["John",40]}')],
    )
    def test_compose_single_prompt(self, composer, prompts, expected_result):
        result = composer.compose(prompts)
        assert result == expected_result

    # Test case to verify error is raised when composing multiple prompts in SingleRequestComposer
    def test_compose_multiple_prompts(self, composer):
        prompts = ['["John",40]', '["Alice",30]']
        message = "prompt_template with \$feature placeholder can only handle single prompt at a time"
        with pytest.raises(
            EvalAlgorithmClientError,
            match="Prompt must be an instance of string",
        ):
            composer.compose(prompts)

    # Test case to verify error is raised for an invalid template
    def test_invalid_template(self):
        template = '{"data":$invalid}'
        composer = PromptComposer(template)
        prompt = '["John",40]'
        with pytest.raises(KeyError):
            composer.compose(prompt)
