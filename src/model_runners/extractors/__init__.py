from infra.utils.sm_exceptions import CustomerError
from orchestrator.configs.analysis_config import ModelAcceptType
from model_runners.extractors.json_extractor import JsonExtractorFacade


def create_extractor(model_accept_type):
    if model_accept_type == ModelAcceptType.JSON.value:
        extractor = JsonExtractorFacade
    else:
        raise CustomerError(
            f"Invalid accept type: {model_accept_type}. "
            f"Please specify one of the following accept types. "
            f"1. {ModelAcceptType.JSON.value}"
        )

    return extractor
