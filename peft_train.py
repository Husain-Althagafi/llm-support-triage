from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_from_disk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import get_model, get_lora_config, get_lora_model


def tokenize_masked(sample: Dict, tokenizer: AutoTokenizer, max_length: int = 512) -> Dict:
    text = sample['text']

    if '\n' in text:
        prompt, response = text.rsplit('\n', 1)
        prompt = prompt + '\n'

    else:
        prompt, response = text, ''

    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length).input_ids[0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run peft training"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name or path"
    )

    args = parser.parse_args()

    # load model and tokenizer
    print(f'Loading model and tokenizer for {args.model_name}...')
    model, tokenizer = get_model(args.model_name)
    print(f"Model and tokenizer for {args.model_name} loaded.")

    # load lora config and model
    lora_config = get_lora_config(r=8, alpha=16)
    print(f'Loading LoRA configuration...')
    lora_model = get_lora_model(model, lora_config)
    print(f'LoRa model loaded.')

    # load datasets
    print(f'Loading datasets...')
    train_ds = load_from_disk('data/triage_train_dataset')
    eval_ds = load_from_disk('data/triage_test_dataset')
    print(f'Datasets loaded.')

    print(train_ds[0])



    
