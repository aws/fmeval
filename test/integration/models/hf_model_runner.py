import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import Tuple, Optional

from fmeval.model_runners.model_runner import ModelRunner


@dataclass(frozen=True)
class HFModelConfig:
    """
    Configures a HuggingFaceCausalLLMModelRunner instance.

    :param model_name: A unique identifier tied to a HuggingFace model.
            See https://huggingface.co/docs/transformers/v4.34.1/en/model_doc/auto#transformers.AutoModel.from_pretrained
    :param max_new_tokens: The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    :param remove_prompt_from_generated_text: Whether to remove the prompt from text that is generated by the model.
    :param do_sample: Whether to use sampling; greedy decoding is used during generation if False.
    """

    model_name: str
    max_new_tokens: int
    remove_prompt_from_generated_text: bool = True
    do_sample: bool = False


class HuggingFaceCausalLLMModelRunner(ModelRunner):
    def __init__(self, model_config: HFModelConfig):
        self.config = model_config
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

    def predict(self, prompt: str) -> Tuple[Optional[str], Optional[float]]:
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generations = self.model.generate(
            **input_ids,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=self.config.do_sample,
        )
        generation_contains_input = (
            input_ids["input_ids"][0] == generations[0][: input_ids["input_ids"].shape[1]]
        ).all()
        if self.config.remove_prompt_from_generated_text and not generation_contains_input:
            warnings.warn(
                "Your model does not return the prompt as part of its generations. "
                "`remove_prompt_from_generated_text` does nothing."
            )
        if self.config.remove_prompt_from_generated_text and generation_contains_input:
            output = self.tokenizer.batch_decode(generations[:, input_ids["input_ids"].shape[1] :])[0]
        else:
            output = self.tokenizer.batch_decode(generations, skip_special_tokens=True)[0]

        with torch.inference_mode():
            input_ids = self.tokenizer(self.tokenizer.bos_token + prompt, return_tensors="pt")["input_ids"]
            model_output = self.model(input_ids, labels=input_ids)
            probability = -model_output[0].item()

        return output, probability
