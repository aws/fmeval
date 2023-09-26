"""
Utilities for model runners.
"""
import atexit
import logging
import time

from abc import ABC, abstractmethod

import sagemaker
from sagemaker.utils import unique_name_from_base
from sagemaker.jumpstart.model import JumpStartModel as SDKJumpStartModel
from sagemaker.predictor import retrieve_default
from sagemaker.base_predictor import PredictorBase
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Model
DEFAULT_ENDPOINT_NAME_PREFIX = "amazon-fm-eval"
ENDPOINT_NAME_PREFIX_PATTERN = "^[a-zA-Z0-9](-*[a-zA-Z0-9])"
BEDROCK_MODEL_ID_PATTERN = (
    "^(arn:aws(-[^:]+)?:bedrock:[a-z0-9-]{1,20}:(([0-9]{12}:custom-model/[a-z0-9-]{1,63}[.]{1}"
    "[a-z0-9-]{1,63}/[a-z0-9]{12})|(:foundation-model/[a-z0-9-]{1,63}[.]{1}[a-z0-9-]{1,63})))|"
    "(([0-9a-zA-Z][_-]?)+)|([a-z0-9-]{1,63}[.]{1}[a-z0-9-]{1,63})$"
)
MODEL_PACKAGE_ARN_PATTERN = "arn:aws[a-z\-]*:sagemaker:[a-z0-9\-]*:[0-9]{12}:model-package/.*"

# https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpoint.html
ENDPOINT_NAME_MAX_LEN = 63

# Predictor constants
OUTPUT = "output"
INPUT_LOG_PROB = "input_log_probability"

@dataclass
class SagemakerEndpointUsageState:
    """Endpoint usage state config use by MangedEndpoint.

    :param start_time: start time of managing this shadow endpoint.
    :param num_calls: Number of calls of waiting for this endpoint spinning up.
    :param in_service: whether the endpoint is in service or not.
    """

    start_time: float
    num_calls: int
    in_service: bool

class SageMakerModelDeploymentConfig:
    """
    Config class that is used to defines how the model is deployed or should be deployed for different model types.

    :param model_type: The ModelType enum that specifies how the model is deployed or should be deployed
    :param model_identifier: The name or ID of the model. The value in this field depends on the `model_type`. For
                             SageMaker Model, it would be the model name. For SageMaker Jumpstart, this would be the
                             model ID, for SageMaker Registry Models, it would be the model package name. For AWS
                             Bedrock models, it would be the Bedrock model ID.
    :param model_version: Version of the SageMaker JumpStart model. If not provided, we automatically use the latest version of the model.
    :param  endpoint_name: The name of the deployed endpoint. This would be provided for model types SageMaker Endpoint,
                           and SageMaker Jumpstart. For SageMaker Jumpstart, this should also be accompanied by
                           Jumpstart model ID used to deploy this endpoint.
    :param endpoint_name_prefix: The prefix to be used for the endpoint name while creating a SageMaker Endpoint. This
                                 would be applicable to all model types where an endpoint needs to be deployed
                                 (Sagemaker Model, Sagemaker Registry Model, Sagemaker Jumpstart (without an endpoint))
    :param instance_count: The number of instances to be used to deploy an endpoint.  This would be applicable to all
                           model types where an endpoint needs to be deployed (Sagemaker Model, Sagemaker Registry
                           Model, Sagemaker Jumpstart (without an endpoint))
    :param instance_type: The instance type to be used to deploy an endpoint.  This would be applicable to all model
                          types where an endpoint needs to be deployed (Sagemaker Model, Sagemaker Registry Model,
                          Sagemaker Jumpstart (without an endpoint))
    :param accelerator_type: SageMaker Elastic Inference accelerator type to deploy to the model endpoint instance for
                             making inferences to the model.  This would be applicable to all model types where an
                             endpoint needs to be deployed (Sagemaker Model, Sagemaker Registry Model, Sagemaker
                             Jumpstart (without an endpoint))
    :param custom_attributes: Provides additional information about a request for an inference submitted to a model
                              hosted at an Amazon SageMaker endpoint. Not applicable for AWS Bedrock Models. The
                              information is an opaque value that is forwarded verbatim. You could use this value, for
                              example, to provide an ID that you can use to track a request or to provide other metadata
                              that a service endpoint was programmed to process. The value must consist of no more than
                              1024 visible US-ASCII characters as specified in Section 3.3.6. Field Value Components of
                              the Hypertext Transfer Protocol (HTTP/1.1).
    :param  tags: List of tags to be added to be added to resources created by Margaret container (For instance, if we
                  create a SageMaker Endpoint, we pass on these tags to it as well). This is not a field in the analysis
                  config, but will be populated from the JobConfig
    :param kms_key_id: The AWS Key Management Service (AWS KMS) key that Amazon SageMaker uses to encrypt data on the
                       the storage volume attached to the ML compute instance(s) that run the processing job. This is
                       not a field in the analysis config, but will be populated from the JobConfig
    """

    model_type: ModelType
    model_identifier: Optional[str] = None
    model_version: Optional[str] = "*"
    endpoint_name: Optional[str] = None
    endpoint_name_prefix: Annotated[
        str, AfterValidator(check_endpoint_name_prefix_matches_regex)
    ] = DEFAULT_ENDPOINT_NAME_PREFIX
    instance_count: Optional[int] = None
    instance_type: Optional[str] = None
    accelerator_type: Optional[str] = None
    custom_attributes: Optional[str] = None
    # These two fields should not be read from analysis config. Instead, they will be read from the job config.
    # Pydantic ignores fields starting with an underscore
    _tags: Optional[List[Tag]] = None
    _kms_key_id: Optional[str] = None

def wait_for_sagemaker_endpoint(
    sagemaker_session: sagemaker.session.Session,
    endpoint_name: str,
    endpoint_usage_state: SagemakerEndpointUsageState,
) -> SagemakerEndpointUsageState:
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

def create_content_composer(template: str) -> Optional[ContentComposer]:
    composer: Optional[ContentComposer] = None
    vanilla_template = VanillaTemplate(template)

    if identifiers := vanilla_template.get_unique_identifiers():
        if SinglePromptContentComposer.KEYWORD in identifiers:
            composer = SinglePromptContentComposer(vanilla_template)
        elif MultiPromptsContentComposer.KEYWORD in identifiers:
            composer = MultiPromptsContentComposer(vanilla_template)
        else:
            logger.error(f"Found placeholders {identifiers} in template '{template}'.")
    else:
        logger.error(f"Could not find any identifier in template '{template}'.")

    if composer is None:
        raise UserError(
            "Invalid content_template. View job logs for details. "
            "The template must contain the placeholder $prompts for an array of prompts, "
            "or the placeholder $prompt for single prompt at a time."
        )
    return composer
