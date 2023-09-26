"""
Module to manage the Sagemaker Jumpstart model runners.
"""
import atexit
import logging
import time

from abc import ABC, abstractmethod

import sagemaker
from sagemaker.utils import unique_name_from_base
from sagemaker.jumpstart.model import JumpStartModel as SDKJumpStartModel
from sagemaker.predictor import retrieve_default
from typing import Optional
from dataclasses import dataclass
from util import SageMakerModelDeploymentConfig, SagemakerEndpointUsageState, OUTPUT, INPUT_LOG_PROB
from model_runner import ModelRunner

logger = logging.getLogger(__name__)

class JumpStartModelRunner(ModelRunner):
    """
    A class to manage the deployment and deletion of SageMaker JumpStart model when user provides
    ModelType.SAGEMAKER_JUMPSTART_MODEL.
    """

    def __init__(
        self,
        self,
        model_deployment_config: SageMakerModelDeploymentConfig,
        sagemaker_session: sagemaker.session.Session,
        endpoint_name: str,
        model_id: str,
        model_name: str,
        content_template: str,
        output: str,
        probability: str,
        content_type: str = "application/json",
        accept_type: str = "application/json"
    ):
        """
        :param sagemaker_session: SageMaker session to be reused.
        :param model_deployment_config: SageMakerModelDeploymentConfig that defines how the model is deployed
        """
        super().__init__()
        self._sagemaker_session: sagemaker.session.Session = sagemaker_session
        self._model_deployment_config = model_deployment_config
        self._predictor = None
        self._endpoint_name = self._model_deployment_config.endpoint_name
        self._composer = create_content_composer(content_template, content_type)
        self._extractor = create_extractor(output, probability, accept_type)

        # to pass mypy check
        assert (
            self._model_deployment_config.model_identifier
        ), "Missing `model_identifier` for ModelType.SAGEMAKER_JUMPSTART_ENDPOINT"
        self._model_id = self._model_deployment_config.model_identifier

        assert (
            self._model_deployment_config.model_identifier
        ), "Missing `model_identifier` for ModelType.SAGEMAKER_JUMPSTART_MODEL"
        self._model_id = self._model_deployment_config.model_identifier

        self.endpoint_usage_state = SagemakerEndpointUsageState(start_time=time.time(), num_calls=0, in_service=False)

        self._sm_predictor = sagemaker.predictor.Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=self._sagemaker_session,
            # we only support JSON format model input/output now
            serializer=sagemaker.serializers.JSONSerializer(),
            deserializer=sagemaker.deserializers.JSONDeserializer(),
        )

        # Register cleanup functions to delete shadow endpoint on termination
        # by exception or sys.exit (https://docs.python.org/library/atexit.html)
        atexit.register(self.clean_up)

    def create_from_model(self):
        """
        Create JumpStart model
        :return SagemakerModelInvocationParameters: parameters related to JumpStart model invocation
        """
        logger.info("Spinning up JumpStart shadow endpoint")
        model = SDKJumpStartModel(
            model_id=self._model_deployment_config.model_identifier,
            model_version=self._model_deployment_config.model_version,
        )
        # Generate the endpoint name from our end in case user specify an endpoint_name_prefix
        endpoint_name = gen_shadow_endpoint_unique_name(
            self._model_deployment_config.endpoint_name_prefix, self._model_id
        )
        # wait is True by default
        self._predictor = model.deploy(
            endpoint_name=endpoint_name,
            initial_instance_count=self._model_deployment_config.instance_count,
            instance_type=self._model_deployment_config.instance_type,
            accelerator_type=self._model_deployment_config.accelerator_type,
            kms_key=self._model_deployment_config.kms_key_id,
            tags=self._model_deployment_config.tags,
        )
        self._endpoint_name = endpoint_name

    def create_from_endpoint(self):
        """
        Create SageMaker predictor with JumpStart endpoint
        :return SagemakerModelInvocationParameters: parameters related to JumpStart model invocation
        """
        # Check endpoint status first before create predictor
        desc = self._sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=self._endpoint_name)
        if not desc or "EndpointStatus" not in desc or desc["EndpointStatus"] != "InService":
            raise UserError(f"Endpoint `{self._model_deployment_config.endpoint_name}` is not in service.")
        self._predictor = retrieve_default(
            endpoint_name=self._model_deployment_config.endpoint_name,
            model_id=self._model_id,
            model_version=self._model_deployment_config.model_version,
            sagemaker_session=self._sagemaker_session,
        )
        self.endpoint_usage_state.in_service = True

    def predict(self, data: Union[str, List[str]]) -> Dict:
        """
        Invoke the SageMaker endpoint and parse the model response.
        :param data: Input data for which you want the model to provide inference.
        :return extracted output/probability dict from model output
        """
        output_dict = {}
        composed_data = self._composer.compose(data)
        num_records = 1 if isinstance(data, str) else len(data)
        model_output = self._sm_predictor.predict(
            data=composed_data, custom_attributes=self._model_deployment_config.custom_attributes
        )
        output_dict[OUTPUT] = self.extract_output(data=model_output, num_records=num_records)
        output_dict[INPUT_LOG_PROB] = self.extract_probability(
            data=model_output, num_records=num_records
        )
        return output_dict

    def clean_up(self):
        """
        Clean up shadow endpoint and related resources.
        """
        end_time = time.time()
        runtime_in_seconds = end_time - self.endpoint_usage_state.start_time
        logger.info(
            "Model endpoint delivered %.5f requests per second and a total of %d requests over %.0f seconds",
            self.endpoint_usage_state.num_calls / (+0.00001 + runtime_in_seconds),
            self.endpoint_usage_state.num_calls,
            runtime_in_seconds,
        )
        self._clean_up_resources()
    def clean_up_resources(self):
        """
        Clean up SageMaker JumpStart endpoint - endpoint, endpoint config and model
        """
        if self._predictor is not None:  # pragma: no branch
            if self._endpoint_name is not None:  # pragma: no branch
                try:
                    logger.debug(f"Deleting endpoint and endpoint configuration: {self._predictor.endpoint_name}")
                    self._predictor.delete_endpoint()
                    self._endpoint_name = None
                except:
                    logger.error(
                        "Failed to delete endpoint and endpoint config for endpoint: %s", self._predictor.endpoint_name
                    )
            try:
                logger.debug(f"Deleting SageMaker JumpStart model for endpoint: {self._predictor.endpoint_name}")
                self._predictor.delete_model()
            except:
                logger.error("Failed to delete JumpStart model for endpoint: %s", self._predictor.endpoint_name)
            self._predictor = None

