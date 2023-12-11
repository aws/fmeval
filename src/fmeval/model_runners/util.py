"""
Utilities for model runners.
"""
import logging
import os
from typing import Literal
import boto3
import botocore
import sagemaker

from fmeval.constants import SAGEMAKER_SERVICE_ENDPOINT_URL, SAGEMAKER_RUNTIME_ENDPOINT_URL
from mypy_boto3_bedrock.client import BedrockClient

logger = logging.getLogger(__name__)


def get_boto_session() -> boto3.session.Session:
    """
    Get boto3 session with adaptive retry config
    :return: The new session
    """
    botocore_session: botocore.session.Session = botocore.session.get_session()
    botocore_session.set_default_client_config(
        botocore.config.Config(
            # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/retries.html
            retries={"mode": "adaptive", "max_attempts": 10}
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
    boto_session = get_boto_session()
    boto_config = botocore.client.Config(retries={"mode": boto_retry_mode, "max_attempts": retry_attempts})
    sagemaker_service_endpoint_url = os.getenv(SAGEMAKER_SERVICE_ENDPOINT_URL)
    sagemaker_runtime_endpoint_url = os.getenv(SAGEMAKER_RUNTIME_ENDPOINT_URL)
    sagemaker_client = boto_session.client(
        service_name="sagemaker",
        endpoint_url=sagemaker_service_endpoint_url,
        config=boto_config,
    )
    sagemaker_runtime_client = boto_session.client(
        service_name="sagemaker-runtime",
        endpoint_url=sagemaker_runtime_endpoint_url,
        config=boto_config,
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
    boto_session = get_boto_session()
    bedrock_runtime_client = boto_session.client(  # type: ignore
        service_name="bedrock-runtime",
        config=botocore.client.Config(retries={"mode": boto_retry_mode, "max_attempts": retry_attempts}),
    )
    return bedrock_runtime_client


def is_endpoint_in_service(
    sagemaker_session: sagemaker.session.Session,
    endpoint_name: str,
) -> bool:
    """
    :param sagemaker_session: SageMaker session to be reused.
    :param endpoint_name: SageMaker endpoint name.
    :return None
    """
    in_service = True
    desc = sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    if not desc or "EndpointStatus" not in desc or desc["EndpointStatus"] != "InService":
        in_service = False
    return in_service
