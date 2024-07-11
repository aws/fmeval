import pickle
from typing import NamedTuple, Optional
from unittest.mock import Mock, patch

import pytest
import sagemaker

from sagemaker.jumpstart.enums import JumpStartModelType
from fmeval.constants import MIME_TYPE_JSON
from fmeval.model_runners.sm_jumpstart_model_runner import JumpStartModelRunner

ENDPOINT_NAME = "valid_endpoint_name"
CUSTOM_ATTRIBUTES = "CustomAttributes"
INFERENCE_COMPONENT_NAME = "valid_inference_component_name"
MODEL_ID = "AwesomeModel"
PROPRIETARY_MODEL_ID = "cohere-gpt-medium"
MODEL_VERSION = "v1.2.3"

CONTENT_TEMPLATE = '{"data":$prompt}'
PROMPT = "This is the model input"
MODEL_INPUT = '{"data": "' + PROMPT + '"}'

OUTPUT = "This is the model output"
LOG_PROBABILITY = 0.9
OUTPUT_JMES_PATH = "predictions.output"
LOG_PROBABILITY_JMES_PATH = "predictions.log_prob"
EMBEDDING_JMES_PATH = "embedding"
MODEL_OUTPUT = {"predictions": {"output": OUTPUT, "log_prob": LOG_PROBABILITY}}
VECTOR = [-0.64453125, -0.20996094, 0.4296875, 0.29296875, 0.484375, 0.29296875]
EMBEDDING_MODEL_OUTPUT = {"embedding": VECTOR, "inputTextTokenCount": 10}

# NOTE: Here the original class name can be used to mock the class, because sm_model_runner.py uses "import sagemaker"
# and then refers to Predictor class by its full path. If the module uses "from sagemaker.predictor import Predictor"
# then class name below should be changed to "model_runners.Predictor". see [1] for the reasons.
# [1] https://docs.python.org/3/library/unittest.mock.html#where-to-patch


class TestJumpStartModelRunner:
    class TestCaseInit(NamedTuple):
        model_id: str

    @pytest.mark.parametrize(
        "test_case_init",
        [
            TestCaseInit(model_id=MODEL_ID),
            TestCaseInit(
                model_id=PROPRIETARY_MODEL_ID,
            ),
        ],
    )
    @patch("sagemaker.session.Session")
    @patch("sagemaker.predictor.Predictor")
    @patch("fmeval.model_runners.sm_jumpstart_model_runner.is_text_embedding_js_model", return_value=False)
    def test_jumpstart_model_runner_init(
        self, is_text_embedding_js_model, sagemaker_predictor_class, sagemaker_session_class, test_case_init
    ):
        """
        GIVEN valid Jumpstart model runner parameters
        WHEN try to create JumpStartModelRunner
        THEN SageMaker Predictor class is created once with expected parameters
        """
        mock_sagemaker_session = sagemaker_session_class()
        mock_sagemaker_session.sagemaker_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}
        mock_sagemaker_session.boto_region_name = "us-west-2"

        with patch("sagemaker.predictor.get_default_predictor") as default_predictor_fn:
            default_predictor = Mock()
            default_predictor_fn.return_value = default_predictor
            default_predictor.accept = MIME_TYPE_JSON
            default_predictor.content_type = MIME_TYPE_JSON

            js_model_runner = JumpStartModelRunner(
                endpoint_name=ENDPOINT_NAME,
                model_id=test_case_init.model_id,
                model_version=MODEL_VERSION,
                content_template=CONTENT_TEMPLATE,
                custom_attributes=CUSTOM_ATTRIBUTES,
                output=OUTPUT_JMES_PATH,
                log_probability=LOG_PROBABILITY_JMES_PATH,
            )

            if test_case_init.model_id == PROPRIETARY_MODEL_ID:
                default_predictor_fn.assert_called_with(
                    predictor=sagemaker_predictor_class.return_value,
                    model_id=test_case_init.model_id,
                    model_version=MODEL_VERSION,
                    model_type=JumpStartModelType.PROPRIETARY,
                    sagemaker_session=sagemaker_session_class.return_value,
                    region=None,
                    hub_arn=None,
                    config_name=None,
                    tolerate_deprecated_model=False,
                    tolerate_vulnerable_model=False,
                )
            else:
                default_predictor_fn.assert_called_with(
                    predictor=sagemaker_predictor_class.return_value,
                    model_id=test_case_init.model_id,
                    model_version=MODEL_VERSION,
                    model_type=JumpStartModelType.OPEN_WEIGHTS,
                    sagemaker_session=sagemaker_session_class.return_value,
                    region=None,
                    hub_arn=None,
                    config_name=None,
                    tolerate_deprecated_model=False,
                    tolerate_vulnerable_model=False,
                )

            sagemaker_predictor_class.assert_called_once_with(
                endpoint_name=ENDPOINT_NAME, sagemaker_session=sagemaker_session_class.return_value, component_name=None
            )

    class TestCasePredict(NamedTuple):
        output_jmespath: Optional[str]
        log_probability_jmespath: Optional[str]
        output: Optional[str]
        log_probability: Optional[float]

    @pytest.mark.parametrize(
        "test_case",
        [
            TestCasePredict(
                output_jmespath=OUTPUT_JMES_PATH,
                log_probability_jmespath=LOG_PROBABILITY_JMES_PATH,
                output=OUTPUT,
                log_probability=LOG_PROBABILITY,
            ),
            TestCasePredict(
                output_jmespath=OUTPUT_JMES_PATH, log_probability_jmespath=None, output=OUTPUT, log_probability=None
            ),
        ],
    )
    @patch("sagemaker.session.Session")
    @patch("fmeval.model_runners.sm_jumpstart_model_runner.is_text_embedding_js_model", return_value=False)
    def test_jumpstart_model_runner_predict(self, is_text_embedding_js_model, sagemaker_session_class, test_case):
        """
        GIVEN valid JumpStartModelRunner
        WHEN predict() called
        THEN SageMaker Predictor predict method is called once with expected parameters,
            and extract output and log probability as expected
        """
        mock_sagemaker_session = sagemaker_session_class()
        mock_sagemaker_session.sagemaker_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}
        mock_sagemaker_session.boto_region_name = "us-west-2"

        with patch.object(
            sagemaker.serializers, "retrieve_default", return_value=sagemaker.serializers.JSONSerializer()
        ) as default_serializer, patch.object(
            sagemaker.deserializers, "retrieve_default", return_value=sagemaker.deserializers.JSONDeserializer()
        ) as default_deserializer, patch.object(
            sagemaker.accept_types, "retrieve_default", return_value=MIME_TYPE_JSON
        ) as default_accept_type, patch.object(
            sagemaker.content_types, "retrieve_default", return_value=MIME_TYPE_JSON
        ):
            js_model_runner = JumpStartModelRunner(
                endpoint_name=ENDPOINT_NAME,
                model_id=MODEL_ID,
                model_version=MODEL_VERSION,
                content_template=CONTENT_TEMPLATE,
                custom_attributes=CUSTOM_ATTRIBUTES,
                output=test_case.output_jmespath,
                log_probability=test_case.log_probability_jmespath,
                component_name=INFERENCE_COMPONENT_NAME,
            )
            # Mocking sagemaker.predictor serializing byte into JSON
            js_model_runner._predictor.deserializer.deserialize = Mock(return_value=MODEL_OUTPUT)

            result = js_model_runner.predict(PROMPT)
            assert mock_sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called
            call_args, kwargs = mock_sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
            assert kwargs == {
                "Accept": MIME_TYPE_JSON,
                "Body": MODEL_INPUT,
                "ContentType": MIME_TYPE_JSON,
                "CustomAttributes": CUSTOM_ATTRIBUTES,
                "EndpointName": ENDPOINT_NAME,
                "InferenceComponentName": INFERENCE_COMPONENT_NAME,
            }
            assert result == (test_case.output, test_case.log_probability)

    @patch("sagemaker.session.Session")
    @patch("fmeval.model_runners.sm_jumpstart_model_runner.is_text_embedding_js_model", return_value=False)
    def test_jumpstart_model_runner_predict_no_output(self, is_text_embedding_js_model, sagemaker_session_class):
        """
        GIVEN valid JumpStartModelRunner
        WHEN predict() called
        THEN SageMaker Predictor predict method is called once with expected parameters,
            and extract output and log probability as expected
        """
        mock_sagemaker_session = sagemaker_session_class()
        mock_sagemaker_session.sagemaker_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}
        mock_sagemaker_session.boto_region_name = "us-west-2"

        with patch.object(
            sagemaker.serializers, "retrieve_default", return_value=sagemaker.serializers.JSONSerializer()
        ) as default_serializer, patch.object(
            sagemaker.deserializers, "retrieve_default", return_value=sagemaker.deserializers.JSONDeserializer()
        ) as default_deserializer, patch.object(
            sagemaker.accept_types, "retrieve_default", return_value=MIME_TYPE_JSON
        ) as default_accept_type, patch.object(
            sagemaker.content_types, "retrieve_default", return_value=MIME_TYPE_JSON
        ):
            js_model_runner = JumpStartModelRunner(
                endpoint_name=ENDPOINT_NAME,
                model_id=MODEL_ID,
                model_version=MODEL_VERSION,
                content_template=CONTENT_TEMPLATE,
                custom_attributes=CUSTOM_ATTRIBUTES,
                output=OUTPUT_JMES_PATH,
                log_probability=LOG_PROBABILITY_JMES_PATH,
            )
            # Mocking sagemaker.predictor serializing byte into JSON
            js_model_runner._predictor.deserializer.deserialize = Mock(return_value=MODEL_OUTPUT)

            result = js_model_runner.predict(PROMPT)
            assert mock_sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called
            call_args, kwargs = mock_sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
            assert kwargs == {
                "Accept": MIME_TYPE_JSON,
                "Body": MODEL_INPUT,
                "ContentType": MIME_TYPE_JSON,
                "CustomAttributes": CUSTOM_ATTRIBUTES,
                "EndpointName": ENDPOINT_NAME,
            }
            assert result == (OUTPUT, LOG_PROBABILITY)

    @patch("sagemaker.session.Session")
    @patch("fmeval.model_runners.sm_jumpstart_model_runner.is_text_embedding_js_model", return_value=True)
    def test_jumpstart_model_runner_predict_embedding(self, is_text_embedding_js_model, sagemaker_session_class):
        """
        GIVEN valid embedding model JumpStartModelRunner
        WHEN predict() called
        THEN SageMaker Predictor predict method is called once with expected parameters,
            and extract embedding as expected
        """
        mock_sagemaker_session = sagemaker_session_class()
        mock_sagemaker_session.sagemaker_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}
        mock_sagemaker_session.boto_region_name = "us-west-2"

        with patch.object(
            sagemaker.serializers, "retrieve_default", return_value=sagemaker.serializers.JSONSerializer()
        ) as default_serializer, patch.object(
            sagemaker.deserializers, "retrieve_default", return_value=sagemaker.deserializers.JSONDeserializer()
        ) as default_deserializer, patch.object(
            sagemaker.accept_types, "retrieve_default", return_value=MIME_TYPE_JSON
        ) as default_accept_type, patch.object(
            sagemaker.content_types, "retrieve_default", return_value=MIME_TYPE_JSON
        ):
            js_model_runner = JumpStartModelRunner(
                endpoint_name=ENDPOINT_NAME,
                model_id=MODEL_ID,
                model_version=MODEL_VERSION,
                content_template=CONTENT_TEMPLATE,
                custom_attributes=CUSTOM_ATTRIBUTES,
                component_name=INFERENCE_COMPONENT_NAME,
            )
            # Mocking sagemaker.predictor serializing byte into JSON
            js_model_runner._predictor.deserializer.deserialize = Mock(return_value=EMBEDDING_MODEL_OUTPUT)

            result = js_model_runner.predict(PROMPT)
            assert mock_sagemaker_session.sagemaker_runtime_client.invoke_endpoint.called
            call_args, kwargs = mock_sagemaker_session.sagemaker_runtime_client.invoke_endpoint.call_args
            assert kwargs == {
                "Accept": MIME_TYPE_JSON,
                "Body": MODEL_INPUT,
                "ContentType": MIME_TYPE_JSON,
                "CustomAttributes": CUSTOM_ATTRIBUTES,
                "EndpointName": ENDPOINT_NAME,
                "InferenceComponentName": INFERENCE_COMPONENT_NAME,
            }
            assert result == VECTOR

    @patch("sagemaker.session.Session")
    @patch("fmeval.model_runners.sm_jumpstart_model_runner.is_text_embedding_js_model", return_value=False)
    def test_jumpstart_model_runner_serializer(self, is_text_embedding_js_model, sagemaker_session_class):
        """
        GIVEN a valid JumpStartModelRunner
        WHEN it is serialized (via pickle.dumps)
        THEN its __reduce__ method produces the correct output, which is verified by
            pickle.loads returning a correct JumpStartModelRunner instance
        """
        mock_sagemaker_session = sagemaker_session_class()
        mock_sagemaker_session.sagemaker_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}
        mock_sagemaker_session.boto_region_name = "us-west-2"

        with patch.object(
            sagemaker.serializers, "retrieve_default", return_value=sagemaker.serializers.JSONSerializer()
        ) as default_serializer, patch.object(
            sagemaker.deserializers, "retrieve_default", return_value=sagemaker.deserializers.JSONDeserializer()
        ) as default_deserializer, patch.object(
            sagemaker.accept_types, "retrieve_default", return_value=MIME_TYPE_JSON
        ) as default_accept_type, patch.object(
            sagemaker.content_types, "retrieve_default", return_value=MIME_TYPE_JSON
        ):
            js_model_runner = JumpStartModelRunner(
                endpoint_name=ENDPOINT_NAME,
                model_id=MODEL_ID,
                model_version=MODEL_VERSION,
                content_template=CONTENT_TEMPLATE,
                custom_attributes=CUSTOM_ATTRIBUTES,
                output=OUTPUT_JMES_PATH,
                log_probability=LOG_PROBABILITY_JMES_PATH,
                embedding=EMBEDDING_JMES_PATH,
                component_name=INFERENCE_COMPONENT_NAME,
            )
            deserialized: JumpStartModelRunner = pickle.loads(pickle.dumps(js_model_runner))
            assert deserialized._endpoint_name == js_model_runner._endpoint_name
            assert deserialized._model_id == js_model_runner._model_id
            assert deserialized._model_version == js_model_runner._model_version
            assert deserialized._content_template == js_model_runner._content_template
            assert deserialized._custom_attributes == js_model_runner._custom_attributes
            assert deserialized._output == js_model_runner._output
            assert deserialized._log_probability == js_model_runner._log_probability
            assert deserialized._embedding == js_model_runner._embedding
            assert deserialized._component_name == js_model_runner._component_name
            assert isinstance(deserialized._predictor, sagemaker.predictor.Predictor)
