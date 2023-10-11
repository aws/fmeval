import json
from unittest.mock import MagicMock, patch

import io
from botocore.response import StreamingBody

from amazon_fmeval.constants import MIME_TYPE_JSON
from amazon_fmeval.model_runners.bedrock_model_runner import BedrockModelRunner

MODEL_ID = "AwesomeModel"

CONTENT_TEMPLATE = '{"data":"$prompt"}'
PROMPT = "This is the model input"
MODEL_INPUT = '{"data": "' + PROMPT + '"}'

OUTPUT = "This is the model output"
LOG_PROBABILITY = 0.9
OUTPUT_JMES_PATH = "predictions.output"
LOG_PROBABILITY_JMES_PATH = "predictions.log_prob"
MODEL_OUTPUT = {"predictions": {"output": OUTPUT, "log_prob": LOG_PROBABILITY}}


class TestBedrockModelRunner:
    def mock_boto3_session_client(*_, **kwargs):
        client = MagicMock()
        client.service_name = kwargs.get("service_name")
        model_output_json = json.dumps(MODEL_OUTPUT)
        model_output_stream = io.StringIO(model_output_json)
        response = {"body": StreamingBody(model_output_stream, len(model_output_json))}
        client.invoke_model.return_value = response
        return client

    @patch("boto3.session.Session.client", side_effect=mock_boto3_session_client, autospec=True)
    def test_bedrock_model_runner_init(self, boto3_client):
        """
        GIVEN valid Bedrock model runner parameters
        WHEN try to create BedrockModelRunner
        THEN Bedrock client is created once with expected parameters
        """
        bedrock_model_runner = BedrockModelRunner(
            model_id=MODEL_ID,
            content_template=CONTENT_TEMPLATE,
            output=OUTPUT_JMES_PATH,
            log_probability=LOG_PROBABILITY_JMES_PATH,
            content_type=MIME_TYPE_JSON,
            accept_type=MIME_TYPE_JSON,
        )
        boto3_client.assert_called_once()

    @patch("boto3.session.Session.client", side_effect=mock_boto3_session_client, autospec=True)
    def test_bedrock_model_runner_predict(self, boto3_client):
        """
        GIVEN valid BedrockModelRunner
        WHEN predict() called
        THEN Bedrock invoke method is called once with expected parameters,
            and extract output and log probability as expected
        """
        bedrock_model_runner = BedrockModelRunner(
            model_id=MODEL_ID,
            content_template=CONTENT_TEMPLATE,
            output=OUTPUT_JMES_PATH,
            log_probability=LOG_PROBABILITY_JMES_PATH,
            content_type=MIME_TYPE_JSON,
            accept_type=MIME_TYPE_JSON,
        )
        # Mocking Bedrock invoke model serializing byte into JSON
        result = bedrock_model_runner.predict(PROMPT)
        assert result == (OUTPUT, LOG_PROBABILITY)
