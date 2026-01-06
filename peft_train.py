from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_from_disk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import get_model, get_lora_config, get_lora_model
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm

SYSTEM_MSG = (
    f'You are a support triage assistant\n'
    f'Return STRICT JSON with keys: category, priority, first_response_steps (array of strings).'
)

def build_prompt(ticket: str) -> str:
    return (
        f'{SYSTEM_MSG}\n'
        f'Ticket: {ticket}\n'
        f'JSON:'
    )

def extract_ticket_from_processed_text(full_text: str) -> str:
    # format is prompt + \n + json
    for line in full_text.splitlines():
        if line.startswith('Ticket: '):
            return line[len('Ticket: '):]  
    return full_text

def try_parse_json(response_text: str) -> Optional[Dict[str, Any]]:
    try:
        start_idx = response_text.index('{')
        end_idx = response_text.rindex('}') + 1
        json_str = response_text[start_idx:end_idx]
        parsed_json = json.loads(json_str)
        return parsed_json
    except (ValueError, json.JSONDecodeError):
        return None

def tokenize_masked(sample: Dict, tokenizer: AutoTokenizer, max_length: int = 512) -> Dict:
    text = sample['text']

    if '\n' in text:
        prompt, response = text.rsplit('\n', 1)
        prompt += '\n'

    else:
        prompt, response = text, ''

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids

    if len(response_ids) == 0 or response_ids[-1] != tokenizer.eos_token_id:
        response_ids = response_ids + [tokenizer.eos_token_id]
    
    input_ids = (prompt_ids + response_ids)

    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        if overflow >= len(prompt_ids):
            input_ids = input_ids[overflow:]
            prompt_ids = []
        else:
            prompt_ids = prompt_ids[overflow:]
            input_ids = prompt_ids + response_ids

    labels = len(prompt_ids) * [-100] + len(response_ids) * [1]
    mask = [1] * len(input_ids)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': mask,
    }

def make_collate(tokenizer):
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]

        max_len = max(len(ids) for ids in input_ids)   




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

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/weights/lora_triage_model",
        help="Directory to save the trained LoRA model"
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
    train_ds = load_from_disk('data/banking77/processed_v1/train')
    eval_ds = load_from_disk('data/banking77/processed_v1/test')
    print(f'Datasets loaded.')

    



    
