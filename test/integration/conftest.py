import os
import sys
import logging

from pytest import fixture

from amazon_fmeval.model_runners.sm_jumpstart_model_runner import JumpStartModelRunner
from amazon_fmeval.model_runners.sm_model_runner import SageMakerModelRunner
from amazon_fmeval.util import project_root
from test.integration.models.hf_model_runner import HFModelConfig, HuggingFaceCausalLLMModelRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@fixture(scope="session")
def integration_tests_dir():
    return os.path.join(project_root(__file__), "test", "integration")


@fixture(scope="session", autouse=True)
def append_integration_dir(integration_tests_dir):
    sys.path.append(integration_tests_dir)


@fixture(scope="session")
def js_model_runner():
    js_endpoint_name = "meta-textgeneration-llama-2-7b-f-integration-test-endpoint"
    js_model_id, js_model_version = "meta-textgeneration-llama-2-7b-f", "*"
    return JumpStartModelRunner(
        endpoint_name=js_endpoint_name,
        model_id=js_model_id,
        model_version=js_model_version,
        output="[0].generation.content",
        content_template='{"inputs": [[{"role":"user", "content": $prompt}]], "parameters": {"max_new_tokens": 10, "top_p": 0.9, "temperature": 1e-20, "do_sample" : false}}',
        custom_attributes="accept_eula=true",
    )


@fixture(scope="session")
def js_model_runner_prompt_template():
    return """
        <s>[INST] <<SYS>>Answer the question at the end in as few words as possible.
        Do not repeat the question. Do not answer in complete sentences. <</SYS>>
        Question: $feature [/INST]
        """


@fixture(scope="session")
def sm_model_runner():
    sm_endpoint_name = "meta-textgeneration-llama-2-7b-f-integration-test-endpoint"
    return SageMakerModelRunner(
        endpoint_name=sm_endpoint_name,
        output="[0].generation.content",
        content_template='{"inputs": [[{"role":"user", "content": $prompt}]], "parameters": {"max_new_tokens": 10, "top_p": 0.9, "temperature": 1e-20, "do_sample" : false}}',
        custom_attributes="accept_eula=true",
    )


@fixture(scope="session")
def sm_model_runner_prompt_template():
    return """
        <s>[INST] <<SYS>>Answer the question at the end in as few words as possible.
        Do not repeat the question. Do not answer in complete sentences. <</SYS>>
        Question: $feature [/INST]
        """


@fixture(scope="session")
def hf_model_runner():
    hf_config = HFModelConfig(model_name="gpt2", max_new_tokens=10)
    return HuggingFaceCausalLLMModelRunner(model_config=hf_config)
