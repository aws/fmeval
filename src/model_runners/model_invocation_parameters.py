from abc import ABC

from sagemaker.base_predictor import PredictorBase
from dataclasses import dataclass


class ModelInvocationParameters(ABC):
    """
    Parameter(s) to be used for model invocation.
    """


@dataclass(frozen=True)
class SageMakerEndpointInvocationParameters(ModelInvocationParameters):
    """
    SageMaker endpoint invocation parameters.

    :param endpoint_name: name of the SageMaker endpoint.
    """

    endpoint_name: str


@dataclass(frozen=True)
class BedrockModelInvocationParameters(ModelInvocationParameters):
    """
    Bedrock model invocation parameters.

    :param model_id: identifier of the Bedrock model.
    """

    model_id: str


@dataclass(frozen=True)
class JumpStartModelInvocationParameters(ModelInvocationParameters):
    """
    SageMaker JumpStart model invocation parameters.

    :param predictor: JumpStart-specific predictor.
    """

    predictor: PredictorBase
