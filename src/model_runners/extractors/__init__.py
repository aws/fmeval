
def create_extractor(model_accept_type):
    if model_accept_type == ModelAcceptType.JSON.value:
        extractor = JsonExtractor
    else:
        raise UserError(
            f"Invalid accept type: {model_accept_type}. "
            f"Please specify one of the following accept types. "
            f"1. {ModelAcceptType.JSON.value}"
        )

    return extractor
