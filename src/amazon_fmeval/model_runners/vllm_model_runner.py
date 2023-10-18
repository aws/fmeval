"""
Module to manage model runners for vLLM API Server as a target.
  https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html#api-server

Right now vLLM is a serving engine for a single model - thus, we don't take a model_id.
"""
import json
import logging
from amazon_fmeval.util import require
from typing import Optional, Tuple
from amazon_fmeval.constants import MIME_TYPE_JSON
from amazon_fmeval.model_runners.model_runner import ModelRunner

import requests

logger = logging.getLogger(__name__)


class VllmModelRunner(ModelRunner):
    """
    A class to manage the creation and deletion of vLLM model runner.
    """

    #
    # For more about the available arguments, see:
    #   https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    #
    def __init__(
        self,
        content_template: str,
        output: Optional[str] = None,
        log_probability: Optional[str] = None,
        remote_uri: Optional[str] = "http://localhost:8000/generate",
        use_beam_search: Optional[bool] = False,
        num_completions: Optional[int] = 1,
        temperature: Optional[float] = 0,
        top_p: Optional[float] = 1,
    ):
        """
        :param content_template: String template to compose the model input from the prompt.
        :param output: JMESPath expression of output in the model output.
        :param log_probability: JMESPath expression of log probability in the model output.
        :param remote_uri: Remote URI for the connection.
        :param use_beam_search: Whether to use beam search instead of sampling.
        :param num_completions: Number of output sequences to return for the given prompt.
        :param temperature: Sampling randomness - lower is more deterministic, zero is greedy sampling.
        :param top_p: Cumulative probability of the top tokens to consider - must be in (0, 1] and must be 1 if greedy sampling.
        """
        super().__init__(content_template, output, log_probability, MIME_TYPE_JSON, MIME_TYPE_JSON)

        self._content_template = content_template
        self._output = output
        self._log_probability = log_probability
        self._remote_uri = remote_uri
        self._use_beam_search = use_beam_search
        self._num_completions = num_completions
        self._temperature = temperature
        self._top_p = top_p

        require(
            output is not None or log_probability is not None,
            "One of output jmespath expression or log probability jmespath expression must be provided",
        )

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Invoke the vLLM endpoint and parse the model response.
        :param prompt: Input data for which you want the model to provide inference.
        """

        request = {
            "prompt": prompt,
            "use_beam_search": self._use_beam_search,
            "n": self._num_completions,  # number of completions
            "temperature": self._temperature,
            "top_p": self._top_p,
        }

        response = requests.post(self._remote_uri, json=request)
        model_output = json.loads(response.text)

        output = (
            self._extractor.extract_output(data=model_output, num_records=1)
            if self._extractor.output_jmespath_expression
            else None
        )

        log_probability = (
            self._extractor.extract_log_probability(data=model_output, num_records=1)
            if self._extractor.log_probability_jmespath_expression
            else None
        )

        return output, log_probability
