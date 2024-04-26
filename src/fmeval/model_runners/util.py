"""
Utilities for model runners.
"""
import logging
import os
from typing import Literal
import boto3
import botocore.session
import botocore.config
import sagemaker

from fmeval.constants import SAGEMAKER_SERVICE_ENDPOINT_URL, SAGEMAKER_RUNTIME_ENDPOINT_URL, DISABLE_FMEVAL_TELEMETRY
from fmeval.util import get_fmeval_package_version
from sagemaker.user_agent import determine_prefix
from mypy_boto3_bedrock.client import BedrockClient


logger = logging.getLogger(__name__)


def get_user_agent_extra() -> str:
    """Return a string containing various user-agent headers to be passed to a botocore config.

    This string will always contain SageMaker Python SDK headers obtained using the determine_prefix
    utility function from sagemaker.user_agent. If fmeval telemetry is enabled, this string will
    additionally contain an fmeval-specific header.

    :return: A string to be used as the user_agent_extra parameter in a botocore config.
    """
    # Obtain user-agent headers for information such as SageMaker notebook instance type and SageMaker Studio app type.
    # We manually obtain these headers, so we can pass them in the user_agent_extra parameter of botocore.config.Config.
    # This is because although these headers are already obtained in the sagemaker.session.Session initializer,
    # there is currently a bug in the sagemaker.session.Session code where these headers get assigned to an instance
    # attribute, but don't actually show up in the user-agent header when making API calls.
    sagemaker_python_sdk_headers = determine_prefix()
    return (
        sagemaker_python_sdk_headers
        if os.getenv(DISABLE_FMEVAL_TELEMETRY)
        else f"{sagemaker_python_sdk_headers} fmeval/{get_fmeval_package_version()}"
    )


def get_boto_session(
    boto_retry_mode: Literal["legacy", "standard", "adaptive"],
    retry_attempts: int,
) -> boto3.session.Session:
    """
    Get boto3 session with adaptive retry config
    :return: The new session
    """
    botocore_session: botocore.session.Session = botocore.session.get_session()
    botocore_session.set_default_client_config(
        botocore.config.Config(
            # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/retries.html
            retries={"mode": boto_retry_mode, "max_attempts": retry_attempts},
            # https://botocore.amazonaws.com/v1/documentation/api/latest/reference/config.html
            user_agent_extra=get_user_agent_extra(),
        )
    )
    return boto3.session.Session(botocore_session=botocore_session)


def get_sagemaker_session(
    boto_retry_mode: Literal["legacy", "standard", "adaptive"] = "adaptive",
    retry_attempts: int = 10,
) -> sagemaker.Session:
    """
    Get SageMaker session with adaptive retry config.
    :param boto_retry_mode: retry mode used for botocore config (legacy/standard/adaptive).
    :param retry_attempts: max retry attempts used for botocore client failures
    :return: The new session
    """
    boto_session = get_boto_session(boto_retry_mode, retry_attempts)
    sagemaker_service_endpoint_url = os.getenv(SAGEMAKER_SERVICE_ENDPOINT_URL)
    sagemaker_runtime_endpoint_url = os.getenv(SAGEMAKER_RUNTIME_ENDPOINT_URL)
    sagemaker_client = boto_session.client(
        service_name="sagemaker",
        endpoint_url=sagemaker_service_endpoint_url,
    )
    sagemaker_runtime_client = boto_session.client(
        service_name="sagemaker-runtime",
        endpoint_url=sagemaker_runtime_endpoint_url,
    )
    sagemaker_session = sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=sagemaker_runtime_client,
    )
    return sagemaker_session


def get_bedrock_runtime_client(
    boto_retry_mode: Literal["legacy", "standard", "adaptive"] = "adaptive",
    retry_attempts: int = 10,
) -> BedrockClient:
    """
    Get Bedrock runtime client with adaptive retry config.
    :param boto_retry_mode: retry mode used for botocore config (legacy/standard/adaptive).
    :param retry_attempts: max retry attempts used for botocore client failures
    :return: The new session
    """
    boto_session = get_boto_session(boto_retry_mode, retry_attempts)
    bedrock_runtime_client = boto_session.client(service_name="bedrock-runtime")
    return bedrock_runtime_client


def is_endpoint_in_service(
    sagemaker_session: sagemaker.session.Session,
    endpoint_name: str,
) -> bool:
    """
    :param sagemaker_session: SageMaker session to be reused.
    :param endpoint_name: SageMaker endpoint name.
    :return: Whether the endpoint is in service
    """
    in_service = True
    desc = sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    if not desc or "EndpointStatus" not in desc or desc["EndpointStatus"] != "InService":
        in_service = False
    return in_service
