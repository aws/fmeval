from typing import List, Optional

from data_loaders.headers.header_generator import HeaderGeneratorConfig, HeaderGenerator
from orchestrator.utils import util


class HeaderFactory:
    """
    This class provides a list of dataset headers.
    """

    @staticmethod
    def get_headers(
        headers: List[str],
        header_generator_config: Optional[HeaderGeneratorConfig] = None,
    ) -> List[str]:
        """
        :param headers: list of customer provided headers for features
        :param header_generator_config: header generator config for target_output, inference_output and category
        :return: list of dataset headers, including customer provided features headers and artificial headers for
                 target_output, inference_output and category columns
        """
        util.assert_condition(headers, "headers cannot be None or an empty list.")
        dataset_headers = headers[:]
        if header_generator_config:  # pragma: no branch
            generated_headers = HeaderGenerator().generate_headers(header_generator_config)
            dataset_headers.extend(generated_headers)
        return dataset_headers
