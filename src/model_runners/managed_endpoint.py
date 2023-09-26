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
from model_runners.model_invocation_parameters import (
    ModelInvocationParameters,
    SageMakerEndpointInvocationParameters,
    JumpStartModelInvocationParameters,
)
from orchestrator.configs.analysis_config import ModelDeploymentConfig
from orchestrator.utils.constants import (
    ENDPOINT_NAME_MAX_LEN,
)
from orchestrator.utils.util import CustomerError

logger = logging.getLogger(__name__)


@dataclass
class EndpointUsageState:
    """Endpoint usage state config use by MangedEndpoint.

    :param start_time: start time of managing this shadow endpoint.
    :param num_calls: Number of calls of waiting for this endpoint spinning up.
    :param in_service: whether the endpoint is in service or not.
    """

    start_time: float
    num_calls: int
    in_service: bool


class ManagedEndpoint(ABC):
    """
    Interface for a ManagedEndpoint.
    """

    def __init__(self):
        self.endpoint_usage_state = EndpointUsageState(start_time=time.time(), num_calls=0, in_service=False)

    @abstractmethod
    def create(self) -> ModelInvocationParameters:
        """
        Create shadow endpoint and make sure deployment completed.
        :return ModelInvocationParameters: parameters related to model invocation
        """

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
        # TODO revisit after re-invent
        self._clean_up_resources()

    @abstractmethod
    def _clean_up_resources(self):
        """
        Clean up shadow endpoint resources
        """


class ManagedSageMakerEndpoint(ManagedEndpoint):
    """
    A class to manage the creation and deletion of SageMaker shadow endpoint when user provides
    ModelType.SAGEMAKER_MODEL. Endpoint is created upon object construction, and ideally it is deleted right before
    program exits. There are cases an endpoint cannot be deleted, e.g. not allowed by service when the endpoint is being
    created; force-kill by signal or Python fatal internal error leaves the program no chance. In these cases, we need a
    way to clean up leaked endpoints.
    """

    def __init__(
        self,
        model_deployment_config: ModelDeploymentConfig,
        sagemaker_session: sagemaker.session.Session,
    ):
        """
        :param model_deployment_config: ModelDeploymentConfig that defines how the model is deployed
        :param sagemaker_session: SageMaker session to be reused.
        """
        super().__init__()
        self._sagemaker_session: sagemaker.session.Session = sagemaker_session
        self._model_deployment_config = model_deployment_config
        self._endpoint_name: Optional[str] = None
        self._endpoint_config_name: Optional[str] = None

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

    def create(self) -> SageMakerEndpointInvocationParameters:
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


class SageMakerEndpoint(ManagedEndpoint):
    """
    A class to manage SageMaker endpoint when user provides ModelType.SAGEMAKER_ENDPOINT.
    """

    def __init__(
        self,
        model_deployment_config: ModelDeploymentConfig,
        sagemaker_session: sagemaker.session.Session,
    ):
        """
        :param model_deployment_config: ModelDeploymentConfig that defines how the model is deployed
        :param sagemaker_session: SageMaker session to be reused.
        """
        super().__init__()
        self._sagemaker_session: sagemaker.session.Session = sagemaker_session
        self._model_deployment_config = model_deployment_config

        # to pass mypy check
        assert (
            self._model_deployment_config.endpoint_name
        ), "Missing `model_identifier` for ModelType.SAGEMAKER_ENDPOINT"
        self._endpoint_name = self._model_deployment_config.endpoint_name

    def create(self) -> SageMakerEndpointInvocationParameters:
        """
        Create SageMaker endpoint invocation parameters
        :return SageMakerEndpointInvocationParameters: parameters related to SageMaker endpoint invocation
        """
        # Check endpoint status first before create predictor
        desc = self._sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=self._endpoint_name)
        if not desc or "EndpointStatus" not in desc or desc["EndpointStatus"] != "InService":
            raise CustomerError(f"Endpoint `{self._endpoint_name}` is not in service.")
        self.endpoint_usage_state.in_service = True
        model_invocation_parameters = SageMakerEndpointInvocationParameters(endpoint_name=self._endpoint_name)
        return model_invocation_parameters

    def _clean_up_resources(self):
        """
        Clean up SageMaker endpoint - pass
        """


class JumpStartModel(ManagedEndpoint):
    """
    A class to manage the deployment and deletion of SageMaker JumpStart model when user provides
    ModelType.SAGEMAKER_JUMPSTART_MODEL.
    """

    def __init__(
        self,
        model_deployment_config: ModelDeploymentConfig,
        sagemaker_session: sagemaker.session.Session,
    ):
        """
        :param sagemaker_session: SageMaker session to be reused.
        :param model_deployment_config: ModelDeploymentConfig that defines how the model is deployed
        """
        super().__init__()
        self._sagemaker_session: sagemaker.session.Session = sagemaker_session
        self._model_deployment_config = model_deployment_config
        self._predictor = None
        self._endpoint_name: Optional[str] = None

        # to pass mypy check
        assert (
            self._model_deployment_config.model_identifier
        ), "Missing `model_identifier` for ModelType.SAGEMAKER_JUMPSTART_MODEL"
        self._model_id = self._model_deployment_config.model_identifier

        # Register cleanup functions to delete shadow endpoint on termination
        # by exception or sys.exit (https://docs.python.org/library/atexit.html)
        atexit.register(self.clean_up)

    def create(self) -> JumpStartModelInvocationParameters:
        """
        Create JumpStart model
        :return JumpStartModelInvocationParameters: parameters related to JumpStart model invocation
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
        model_invocation_parameters = JumpStartModelInvocationParameters(predictor=self._predictor)
        return model_invocation_parameters

    def _clean_up_resources(self):
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


class JumpStartEndpoint(ManagedEndpoint):
    """
    A class to manage the SageMaker JumpStart endpoint when user provides
    ModelType.SAGEMAKER_JUMPSTART_ENDPOINT.
    """

    def __init__(
        self,
        model_deployment_config: ModelDeploymentConfig,
        sagemaker_session: sagemaker.session.Session,
    ):
        """
        :param sagemaker_session: SageMaker session to be reused.
        :param model_deployment_config: ModelDeploymentConfig that defines how the model is deployed
        """
        super().__init__()
        self._sagemaker_session: sagemaker.session.Session = sagemaker_session
        self._model_deployment_config = model_deployment_config
        self._predictor = None
        self._endpoint_name = self._model_deployment_config.endpoint_name

        # to pass mypy check
        assert (
            self._model_deployment_config.model_identifier
        ), "Missing `model_identifier` for ModelType.SAGEMAKER_JUMPSTART_ENDPOINT"
        self._model_id = self._model_deployment_config.model_identifier

    def create(self) -> JumpStartModelInvocationParameters:
        """
        Create SageMaker predictor with JumpStart endpoint
        :return JumpStartModelInvocationParameters: parameters related to JumpStart model invocation
        """
        # Check endpoint status first before create predictor
        desc = self._sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=self._endpoint_name)
        if not desc or "EndpointStatus" not in desc or desc["EndpointStatus"] != "InService":
            raise CustomerError(f"Endpoint `{self._model_deployment_config.endpoint_name}` is not in service.")
        self._predictor = retrieve_default(
            endpoint_name=self._model_deployment_config.endpoint_name,
            model_id=self._model_id,
            model_version=self._model_deployment_config.model_version,
            sagemaker_session=self._sagemaker_session,
        )
        self.endpoint_usage_state.in_service = True
        model_invocation_parameters = JumpStartModelInvocationParameters(predictor=self._predictor)
        return model_invocation_parameters

    def _clean_up_resources(self):
        """
        Clean up SageMaker JumpStart endpoint - pass
        """


# Model Util functions (Stayed in the same file to avoid circular import error)
def wait_for_sagemaker_endpoint(
    sagemaker_session: sagemaker.session.Session,
    endpoint_name: str,
    endpoint_usage_state: EndpointUsageState,
) -> EndpointUsageState:
    """
    Wait for SageMaker shadow endpoint deployment to complete.
    :param sagemaker_session: SageMaker session to be reused.
    :param endpoint_name: SageMaker shadow endpoint name.
    :param endpoint_usage_state: Endpoint usage state config use by MangedEndpoint.
    :return EndpointUsageState: Updated endpoint usage state config use by MangedEndpoint.
    """
    from inspect import cleandoc

    if not endpoint_usage_state.in_service:  # pragma: no branch
        # When should_wait is true we always make sure the endpoint is ready.
        # Otherwise, we only wait if we created a new shadow endpoint (in this case
        # model_name is set).
        wait_time_start = time.time()
        logger.info(
            cleandoc(
                """Checking endpoint status:
        Legend:
        (OutOfService: x, Creating: -, Updating: -, InService: !, RollingBack: <, Deleting: o, Failed: *)"""
            )
        )
        # Do initial check to see if endpoint is already up and running before wait polling
        # See https://tiny.amazon.com/6uiwux44/issuamazissuRAI4 for why this is needed
        desc = sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        if not desc or "EndpointStatus" not in desc or desc["EndpointStatus"] != "InService":
            # NOTE: Endpoint creation usually takes 5 to 12 minutes, here poll status every minute.
            sagemaker_session.wait_for_endpoint(endpoint_name, poll=60)
        wait_time = time.time() - wait_time_start
        logger.info("Endpoint is in service after %.0f seconds", wait_time)
    endpoint_usage_state.in_service = True
    endpoint_usage_state.start_time = time.time()
    endpoint_usage_state.num_calls += 1
    return endpoint_usage_state


def gen_shadow_endpoint_unique_name(prefix: str, model_name: str, unique_len: int = 15) -> str:
    """
    Generate shadow endpoint unique name.
    :param prefix: endpoint name prefix
    :param model_name: model name
    :param unique_len: unique characters length
    :return: endpoint name
    """
    rem_len = ENDPOINT_NAME_MAX_LEN - unique_len
    if len(prefix) + len(model_name) > rem_len:
        prefix_len = (rem_len // 2) - 1
        prefix = prefix[:prefix_len]
        model_name_len = rem_len - len(prefix)
        model_name = model_name[:model_name_len]
    endpoint_name = unique_name_from_base(prefix + "-" + model_name, ENDPOINT_NAME_MAX_LEN)
    return endpoint_name
