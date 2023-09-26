"""
Module to manage the creation and deletion of shadow endpoint.
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
from util import SageMakerModelDeploymentConfig
from model_runner import ModelRunner

logger = logging.getLogger(__name__)


class SageMakerModelRunner(ModelRunner):
    """
    A class to manage the creation and deletion of SageMaker shadow endpoint when user provides
    ModelType.SAGEMAKER_MODEL. Endpoint is created upon object construction, and ideally it is deleted right before
    program exits. There are cases an endpoint cannot be deleted, e.g. not allowed by service when the endpoint is being
    created; force-kill by signal or Python fatal internal error leaves the program no chance. In these cases, we need a
    way to clean up leaked endpoints.
    """

    def __init__(
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
        :param model_deployment_config: SageMakerModelDeploymentConfig that defines how the model is deployed
        :param sagemaker_session: SageMaker session to be reused.
        """
        super().__init__()
        self._sagemaker_session: sagemaker.session.Session = sagemaker_session
        self._model_deployment_config = model_deployment_config
        self._endpoint_name: Optional[str] = None
        self._endpoint_name = self._model_deployment_config.endpoint_name
        self._composer = create_content_composer(content_template, content_type)
        self._extractor = create_extractor(output, probability, accept_type)

        # Register cleanup functions to delete shadow endpoint on termination
        # by exception or sys.exit (https://docs.python.org/library/atexit.html)
        # We only register if we create Shadow Endpoint.
        atexit.register(self.clean_up)
        # To create shadow endpoint by model name provided by caller
        logger.info("Spinning up shadow endpoint")

        # to pass mypy check
        assert (
            self._model_deployment_config.model_identifier
        ), "Missing `model_identifier` for ModelType.SAGEMAKER_MODEL"
        self._model_name = self._model_deployment_config.model_identifier
        endpoint_config_name = unique_name_from_base(self._model_deployment_config.endpoint_name_prefix + "-config")
        self._sagemaker_session.create_endpoint_config(
            name=endpoint_config_name,
            model_name=self._model_name,
            initial_instance_count=self._model_deployment_config.instance_count,
            instance_type=self._model_deployment_config.instance_type,
            accelerator_type=self._model_deployment_config.accelerator_type,
            kms_key=self._model_deployment_config.kms_key_id,
            tags=self._model_deployment_config.tags,
        )
        self._endpoint_config_name = endpoint_config_name

    def create_from_model(self) -> SageMakerEndpointInvocationParameters:
        """
        Create SageMaker shadow endpoint
        :return SageMakerEndpointInvocationParameters: parameters related to SageMaker endpoint model invocation
        """
        endpoint_name = gen_shadow_endpoint_unique_name(
            self._model_deployment_config.endpoint_name_prefix, self._model_name
        )
        logger.info("Creating endpoint: '%s'", endpoint_name)
        # Sagemaker python sdk still doesn't support creating TTL enabled endpoints
        # Need to figure out a way for possible memory leak without specifying StoppingCondition
        self._sagemaker_session.sagemaker_client.create_endpoint(  # type: ignore
            EndpointName=endpoint_name,
            EndpointConfigName=self._endpoint_config_name,
            Tags=self._model_deployment_config.tags,
        )
        self.endpoint_usage_state = wait_for_sagemaker_endpoint(
            sagemaker_session=self._sagemaker_session,
            endpoint_name=endpoint_name,
            endpoint_usage_state=self.endpoint_usage_state,
        )
        self._endpoint_name = endpoint_name
        model_invocation_parameters = SageMakerEndpointInvocationParameters(endpoint_name=self._endpoint_name)
        return model_invocation_parameters

    def create_from_endpoint(self) -> SageMakerEndpointInvocationParameters:
        """
        Create SageMaker endpoint invocation parameters
        :return SageMakerEndpointInvocationParameters: parameters related to SageMaker endpoint invocation
        """
        # Check endpoint status first before create predictor
        desc = self._sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=self._endpoint_name)
        if not desc or "EndpointStatus" not in desc or desc["EndpointStatus"] != "InService":
            raise UserError(f"Endpoint `{self._endpoint_name}` is not in service.")
        self.endpoint_usage_state.in_service = True
        model_invocation_parameters = SageMakerEndpointInvocationParameters(endpoint_name=self._endpoint_name)
        return model_invocation_parameters

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
    def _clean_up_resources(self):
        """
        Clean up SageMaker shadow endpoint resources - endpoint config and endpoint
        """
        if self._endpoint_config_name is not None:
            # noinspection PyBroadException
            try:
                logger.debug(f"Deleting endpoint configuration: {self._endpoint_config_name}")
                self._sagemaker_session.delete_endpoint_config(self._endpoint_config_name)
                self._endpoint_config_name = None
            except:
                logger.error("Failed to delete endpoint config %s", self._endpoint_config_name)
        if self._endpoint_name is not None:
            # noinspection PyBroadException
            try:
                logger.debug(f"Deleting endpoint: {self._endpoint_name}")
                self._sagemaker_session.delete_endpoint(self._endpoint_name)
                self._endpoint_name = None
            except:
                logger.error("Failed to delete endpoint %s", self._endpoint_name)
