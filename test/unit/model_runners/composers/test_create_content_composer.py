import pytest

from typing import NamedTuple

from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.model_runners.composers import (
    create_content_composer,
    Composer,
    JsonContentComposer,
    JumpStartComposer,
)


class TestCreateContentComposer:
    class TestCaseGetComposerType(NamedTuple):
        template: str
        expected_composer_type: Composer

    @pytest.mark.parametrize(
        "test_case",
        [
            # Test case to verify that create_content_composer correctly creates a SingleRequestComposer
            TestCaseGetComposerType('{"data":$prompt}', JsonContentComposer),
        ],
    )
    def test_create_content_composer(self, test_case):
        composer = create_content_composer(test_case.template)
        assert isinstance(composer, test_case.expected_composer_type)

    def test_create_content_composer_jumpstart(self):
        assert isinstance(create_content_composer(jumpstart_model_id="model_id"), JumpStartComposer)

    # Test case to verify that create_content_composer raises CustomerError for an invalid template
    def test_invalid_template(self):
        template = '{"data":$invalid}'
        message = "Invalid input - unable to create a content composer"
        with pytest.raises(EvalAlgorithmClientError, match=message):
            create_content_composer(template)

    # Test case to verify that create_content_composer raises CustomerError for a template with no placeholder
    def test_no_placeholder(self):
        template = '{"data":"some data"}'
        with pytest.raises(EvalAlgorithmClientError):
            create_content_composer(template)

    @pytest.mark.parametrize(
        "template, expected_composer_type, prompts",
        [
            ('"data":$prompt', JsonContentComposer, ['["John",40]']),
        ],
    )
    def test_not_stringified_json(self, template, expected_composer_type, prompts):
        """
        GIVEN an invalid template that cannot create JSON like object string
        WHEN compose data
        THEN raise EvalAlgorithmClientError
        """
        composer = create_content_composer(template=template)
        assert isinstance(composer, expected_composer_type)
        error_message = "Unable to load a JSON object with template "
        with pytest.raises(EvalAlgorithmClientError, match=error_message):
            composer.compose(prompts)
