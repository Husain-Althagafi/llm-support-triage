from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def get_model(model_name, load_in_4bit=True):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def get_lora_config(r, alpha):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias='none',
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        task_type="CAUSAL_LM",
    )
    return lora_config

def get_lora_model(model, lora_config):
    lora_model = get_peft_model(model, lora_config)
    return lora_model

