from amazon_fmeval.model_runners.sm_jumpstart_model_runner import JumpStartModelRunner
from amazon_fmeval.model_runners.sm_model_runner import SageMakerModelRunner
from test.integration.models.hf_model_runner import HFModelConfig, HuggingFaceCausalLLMModelRunner

"""
These model runners get used by each of the integration tests.
"""

# JumpStart model runner
js_endpoint_name = "meta-textgeneration-llama-2-7b-f-integration-test-endpoint"
js_model_id, js_model_version = "meta-textgeneration-llama-2-7b-f", "*"
js_model_runner = JumpStartModelRunner(
    endpoint_name=js_endpoint_name,
    model_id=js_model_id,
    model_version=js_model_version,
    output="[0].generation.content",
    content_template='{"inputs": [[{"role":"user", "content": $prompt}]], "parameters": {"max_new_tokens": 10, "top_p": 0.9, "temperature": 1e-20, "do_sample" : false}}',
    custom_attributes="accept_eula=true",
)

# SageMaker model runner
sm_endpoint_name = "meta-textgeneration-llama-2-7b-f-integration-test-endpoint"
sm_model_runner = SageMakerModelRunner(
    endpoint_name=sm_endpoint_name,
    output="[0].generation.content",
    content_template='{"inputs": [[{"role":"user", "content": $prompt}]], "parameters": {"max_new_tokens": 10, "top_p": 0.9, "temperature": 1e-20, "do_sample" : false}}',
    custom_attributes="accept_eula=true",
)

# Huggingface model runner
hf_config = HFModelConfig(model_name="gpt2", max_new_tokens=10)
hf_model_runner = HuggingFaceCausalLLMModelRunner(model_config=hf_config)
