"""
Utilities for model runners.
"""
import logging
from typing import Literal
import boto3
import botocore
import sagemaker

logger = logging.getLogger(__name__)


def get_boto_session() -> boto3.session.Session:  # pragma: no cover
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
    boto_retry_mode: Literal["legacy", "standard", "adaptive"] = "standard",
    retry_attempts: int = 0,
) -> sagemaker.Session:  # pragma: no cover
    """
    Get SageMaker session with adaptive retry config.
    :param boto_retry_mode: retry mode used for botocore config (standard/adaptive).
    :param retry_attempts: max retry attempts used for botocore client failures
    :return: The new session
    """
    boto_session = get_boto_session()
    # noinspection PyTypeChecker
    sagemaker_client = boto_session.client(service_name="sagemaker")
    # noinspection PyTypeChecker
    sagemaker_runtime_client = boto_session.client(  # type: ignore
        service_name="sagemaker-runtime",
        config=botocore.client.Config(
            # Disable retry because our Predictor class has its own retry logic
            retries={"mode": boto_retry_mode, "max_attempts": retry_attempts}
        ),
    )
    sagemaker_session = sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=sagemaker_runtime_client,
    )
    return sagemaker_session


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
