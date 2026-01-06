from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from model import get_model, get_lora_config, get_lora_model
import argparse
import json
import os
from typing import List, Dict, Any, Optional
from datasets import load_from_disk
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

@torch.inference_mode()
def generate_batch(model, tokenizer, texts: List[str], max_new_tokens: int = 256) -> List[str]:
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        padding_side='left',
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Inference script parser"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        '--use_lora',
        action='store_true',
        help='Enable LoRa'
    )

    args = parser.parse_args()

    model, tokenizer = get_model(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TODO: load LoRa weights if specified
    if args.use_lora:
        pass

    test_ds = load_from_disk('data/banking77/processed_v1/test')

    test_ds = test_ds.map(
        lambda sample: {
            'full_prompt': build_prompt(extract_ticket_from_processed_text(sample['text'])),
            'gt_cat': sample['category'],
            'gt_prio': sample['priority']
        },
        remove_columns = test_ds.column_names
    )

    # TODO make better inferencing and evaluation (eval script or code can be in eval.py) also should save results
    total = 0
    valid = 0
    batch_size = 8

    for i in tqdm(range(0, len(test_ds), batch_size), desc=f'Batch inference'):
        batch = test_ds[i:i+batch_size]
        prompts = batch['full_prompt']
        responses = generate_batch(model, tokenizer, prompts)

        for j in range(len(responses)):
            total += 1
            resp_json = try_parse_json(responses[j])
            if resp_json is not None:
                print(f'GT Category: {batch["gt_cat"][j]}, GT Priority: {batch["gt_prio"][j]}')
                print(f'Predicted Response JSON: {resp_json["category"]}, {resp_json["priority"]}')
                pred_cat = resp_json.get('category', None)
                pred_prio = resp_json.get('priority', None)

                if pred_cat == batch['gt_cat'][j] and pred_prio == batch['gt_prio'][j]:
                    valid += 1
    print(f'Accuracy: {valid}/{total} = {valid/total:.4f}')


    


    
    
