from amazon_fmeval.model_runners.sm_jumpstart_model_runner import JumpStartModelRunner
from amazon_fmeval.model_runners.sm_model_runner import SageMakerModelRunner
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
js_model_runner_prompt_template = """
    <s>[INST] <<SYS>>Answer the question at the end in as few words as possible.
    Do not repeat the question. Do not answer in complete sentences. <</SYS>>
    Question: $feature [/INST]
    """

# SageMaker model runner
sm_endpoint_name = "meta-textgeneration-llama-2-7b-f-2023-10-18-18-31-53-528"
sm_model_runner = SageMakerModelRunner(
    endpoint_name=sm_endpoint_name,
    output="[0].generation.content",
    content_template='{"inputs": [[{"role":"user", "content": $prompt}]], "parameters": {"max_new_tokens": 10, "top_p": 0.9, "temperature": 1e-20, "do_sample" : false}}',
    custom_attributes="accept_eula=true",
)
sm_model_runner_prompt_template = """
    <s>[INST] <<SYS>>Answer the question at the end in as few words as possible.
    Do not repeat the question. Do not answer in complete sentences. <</SYS>>
    Question: $feature [/INST]
    """

# Huggingface model runner
hf_config = HFModelConfig(model_name="gpt2", max_new_tokens=10)
hf_model_runner = HuggingFaceCausalLLMModelRunner(model_config=hf_config)
