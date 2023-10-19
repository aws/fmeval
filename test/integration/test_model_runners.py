from amazon_fmeval.model_runners.sm_jumpstart_model_runner import JumpStartModelRunner
from test.integration.hf_model_runner import HFModelConfig, HuggingFaceCausalLLMModelRunner

"""
These model runners get used by various different integration tests.
"""

# JumpStart model runner
js_endpoint_name = "meta-textgeneration-llama-2-7b-f-2023-10-18-18-31-53-528"
js_model_id, js_model_version = "meta-textgeneration-llama-2-7b-f", "*"
js_model_runner = JumpStartModelRunner(
    endpoint_name=js_endpoint_name,
    model_id=js_model_id,
    model_version=js_model_version,
    output="[0].generation.content",
    content_template='{"inputs": [[{"role":"user", "content": $prompt}]], "parameters": {"max_new_tokens": 10, "top_p": 0.9, "temperature": 1e-20, "do_sample" : false}}',
    custom_attributes="accept_eula=true",
)

# Huggingface model runner
hf_config = HFModelConfig(model_name="gpt2", max_new_tokens=32)
hf_model_runner = HuggingFaceCausalLLMModelRunner(model_config=hf_config)
