import pickle
from typing import NamedTuple, Optional
from unittest.mock import Mock, patch

import pytest
import sagemaker

from fmeval.constants import MIME_TYPE_JSON
from fmeval.model_runners.sm_model_runner import SageMakerModelRunner

ENDPOINT_NAME = "valid_endpoint_name"
CUSTOM_ATTRIBUTES = "CustomAttributes"
INFERENCE_COMPONENT_NAME = "valid_inference_component_name"

CONTENT_TEMPLATE = '{"data":$prompt}'
PROMPT = "This is the model input"
MODEL_INPUT = '{"data": "' + PROMPT + '"}'

OUTPUT = "This is the model output"
LOG_PROBABILITY = 0.9
OUTPUT_JMES_PATH = "predictions.output"
LOG_PROBABILITY_JMES_PATH = "predictions.log_prob"
MODEL_OUTPUT = {"predictions": {"output": OUTPUT, "log_prob": LOG_PROBABILITY}}
EMBEDDING_JMES_PATH = "embedding"
VECTOR = [-0.64453125, -0.20996094, 0.4296875, 0.29296875, 0.484375, 0.29296875]
EMBEDDING_MODEL_OUTPUT = {"embedding": VECTOR, "inputTextTokenCount": 10}

# NOTE: Here the original class name can be used to mock the class, because sm_model_runner.py uses "import sagemaker"
# and then refers to Predictor class by its full path. If the module uses "from sagemaker.predictor import Predictor"
# then class name below should be changed to "model_runners.Predictor". see [1] for the reasons.
# [1] https://docs.python.org/3/library/unittest.mock.html#where-to-patch


class TestSageMakerModelRunner:
    @patch("sagemaker.session.Session")
    @patch("sagemaker.deserializers.JSONDeserializer")
    @patch("sagemaker.serializers.JSONSerializer")
    @patch("sagemaker.predictor.Predictor")
    def test_sm_model_runner_init(
        self, sagemaker_predictor_class, json_serializer_class, json_deserializer_class, sagemaker_session_class
    ):
        """
        GIVEN valid SageMaker model runner parameters
        WHEN try to create SageMakerModelRunner
        THEN SageMaker Predictor class is created once with expected parameters
        """
        mock_sagemaker_session = sagemaker_session_class()
        mock_sagemaker_session.sagemaker_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}

        sm_model_runner = SageMakerModelRunner(
            endpoint_name=ENDPOINT_NAME,
            content_template=CONTENT_TEMPLATE,
            custom_attributes=CUSTOM_ATTRIBUTES,
            output=OUTPUT_JMES_PATH,
            log_probability=LOG_PROBABILITY_JMES_PATH,
            embedding=EMBEDDING_JMES_PATH,
            content_type=MIME_TYPE_JSON,
            accept_type=MIME_TYPE_JSON,
        )
        sagemaker_predictor_class.assert_called_once_with(
            endpoint_name=ENDPOINT_NAME,
            sagemaker_session=sagemaker_session_class.return_value,
            serializer=json_serializer_class.return_value,
            deserializer=json_deserializer_class.return_value,
        )

    class TestCasePredict(NamedTuple):
        output_jmespath: Optional[str]
        log_probability_jmespath: Optional[str]
        output: Optional[str]
        log_probability: Optional[float]
        component_name: Optional[str]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCasePredict(
                output_jmespath=OUTPUT_JMES_PATH,
                log_probability_jmespath=LOG_PROBABILITY_JMES_PATH,
                output=OUTPUT,
                log_probability=LOG_PROBABILITY,
                component_name=None,
            ),
            TestCasePredict(
                output_jmespath=None,
                log_probability_jmespath=LOG_PROBABILITY_JMES_PATH,
                output=None,
                log_probability=LOG_PROBABILITY,
                component_name=None,
            ),
            TestCasePredict(
                output_jmespath=OUTPUT_JMES_PATH,
                log_probability_jmespath=None,
                output=OUTPUT,
                log_probability=None,
                component_name=None,
            ),
            TestCasePredict(
                output_jmespath=OUTPUT_JMES_PATH,
                log_probability_jmespath=LOG_PROBABILITY_JMES_PATH,
                output=OUTPUT,
                log_probability=LOG_PROBABILITY,
                component_name=INFERENCE_COMPONENT_NAME,
            ),
        ],
    )
    @patch("sagemaker.session.Session")
    def test_sm_model_runner_predict(self, sagemaker_session_class, test_case):
        """
        GIVEN valid SageMakerModelRunner
        WHEN predict() called
        THEN SageMaker Predictor predict method is called once with expected parameters,
            and extract output and log probability as expected
        """
        mock_sagemaker_session = sagemaker_session_class()
        mock_sagemaker_session.sagemaker_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}

        sm_model_runner = SageMakerModelRunner(
            endpoint_name=ENDPOINT_NAME,
            content_template=CONTENT_TEMPLATE,
            custom_attributes=CUSTOM_ATTRIBUTES,
            output=test_case.output_jmespath,
            log_probability=test_case.log_probability_jmespath,
            content_type=MIME_TYPE_JSON,
            accept_type=MIME_TYPE_JSON,
            component_name=test_case.component_name,
        )
        # Mocking sagemaker.predictor serializing byte into JSON
        sm_model_runner._predictor.deserializer.deserialize = Mock(return_value=MODEL_OUTPUT)
        result = sm_model_runner.predict(PROMPT)
        assert mock_sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called
        call_args, kwargs = mock_sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
        expected_kwargs = {
            "Accept": MIME_TYPE_JSON,
            "Body": MODEL_INPUT,
            "ContentType": MIME_TYPE_JSON,
            "CustomAttributes": CUSTOM_ATTRIBUTES,
            "EndpointName": ENDPOINT_NAME,
        }
        if test_case.component_name:
            expected_kwargs["InferenceComponentName"] = test_case.component_name
        assert kwargs == expected_kwargs
        assert result == (test_case.output, test_case.log_probability)

    @patch("sagemaker.session.Session")
    def test_sm_model_runner_predict_embedding_model(self, sagemaker_session_class):
        """
        GIVEN valid embeddding model SageMakerModelRunner
        WHEN predict() called
        THEN SageMaker Predictor predict method is called once with expected parameters,
            and extract embedding as expected
        """
        mock_sagemaker_session = sagemaker_session_class()
        mock_sagemaker_session.sagemaker_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}

        sm_model_runner = SageMakerModelRunner(
            endpoint_name=ENDPOINT_NAME,
            content_template=CONTENT_TEMPLATE,
            custom_attributes=CUSTOM_ATTRIBUTES,
            embedding=EMBEDDING_JMES_PATH,
            content_type=MIME_TYPE_JSON,
            accept_type=MIME_TYPE_JSON,
        )
        # Mocking sagemaker.predictor serializing byte into JSON
        sm_model_runner._predictor.deserializer.deserialize = Mock(return_value=EMBEDDING_MODEL_OUTPUT)
        result = sm_model_runner.predict(PROMPT)
        assert mock_sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called
        call_args, kwargs = mock_sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
        expected_kwargs = {
            "Accept": MIME_TYPE_JSON,
            "Body": MODEL_INPUT,
            "ContentType": MIME_TYPE_JSON,
            "CustomAttributes": CUSTOM_ATTRIBUTES,
            "EndpointName": ENDPOINT_NAME,
        }
        assert kwargs == expected_kwargs
        assert result == VECTOR

    @patch("sagemaker.session.Session")
    def test_sm_model_runner_serializer(self, sagemaker_session_class):
        """
        GIVEN a valid SageMakerModelRunner
        WHEN it is serialized (via pickle.dumps)
        THEN its __reduce__ method produces the correct output, which is verified by
            pickle.loads returning a correct SageMakerModelRunner instance
        """
        mock_sagemaker_session = sagemaker_session_class()
        mock_sagemaker_session.sagemaker_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}

        sm_model_runner = SageMakerModelRunner(
            endpoint_name=ENDPOINT_NAME,
            content_template=CONTENT_TEMPLATE,
            custom_attributes=CUSTOM_ATTRIBUTES,
            output=OUTPUT_JMES_PATH,
            log_probability=LOG_PROBABILITY_JMES_PATH,
            embedding=EMBEDDING_JMES_PATH,
            content_type=MIME_TYPE_JSON,
            accept_type=MIME_TYPE_JSON,
            component_name=INFERENCE_COMPONENT_NAME,
        )
        deserialized: SageMakerModelRunner = pickle.loads(pickle.dumps(sm_model_runner))
        assert deserialized._endpoint_name == sm_model_runner._endpoint_name
        assert deserialized._content_template == sm_model_runner._content_template
        assert deserialized._custom_attributes == sm_model_runner._custom_attributes
        assert deserialized._output == sm_model_runner._output
        assert deserialized._log_probability == sm_model_runner._log_probability
        assert deserialized._embedding == sm_model_runner._embedding
        assert deserialized._content_type == sm_model_runner._content_type
        assert deserialized._accept_type == sm_model_runner._accept_type
        assert deserialized._component_name == sm_model_runner._component_name
        assert isinstance(deserialized._predictor, sagemaker.predictor.Predictor)
