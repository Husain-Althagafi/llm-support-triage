from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_from_disk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import get_model, get_lora_config, get_lora_model
import json
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import GradScaler, autocast


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

    labels = [-100] * len(prompt_ids)  + input_ids[len(prompt_ids):]
    mask = [1] * len(input_ids)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': mask,
    }

def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def pad(sequences: List[List[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(seq) for seq in sequences)
        return torch.tensor(
            [seq + [pad_value] * (max_len - len(seq)) for seq in sequences],
            dtype=torch.long
        )

    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]

        return {
            'input_ids': pad(input_ids, pad_id),
            'labels': pad(labels, -100),
            'attention_mask': pad(attention_mask, 0),
        }
    
    return collate_fn

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

    if not torch.cuda.is_available():
        raise RuntimeError(f'CUDA not available.')

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
    train_ds = train_ds.select(range(100))
    test_ds = load_from_disk('data/banking77/processed_v1/test')
    test_ds = test_ds.select(range(50))
    print(f'Datasets loaded.')

    print(f'Tokenizing datasets...')
    train_tok = train_ds.map(lambda x: tokenize_masked(x, tokenizer), remove_columns=train_ds.column_names)
    test_tok = test_ds.map(lambda x: tokenize_masked(x, tokenizer), remove_columns=test_ds.column_names)

    train_loader = DataLoader(train_tok, batch_size=1, shuffle=True, collate_fn=make_collate_fn(tokenizer))
    test_loader = DataLoader(test_tok, batch_size=1, shuffle=False, collate_fn=make_collate_fn(tokenizer))

    device = torch.device('cuda')
    lora_model.to(device)

    optimizer = AdamW((p for p in lora_model.parameters() if p.requires_grad), lr=1e-4)
    grad_accum = 8
    scaler = GradScaler()
    num_epochs = 2

    lora_model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        eval_loss = 0.0
        training_steps = 0
        testing_steps = 0
        lora_model.train()

        #   Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(device_type="cuda", dtype=torch.float16):
                loss = lora_model(**batch).loss / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * grad_accum
            training_steps += 1
            

        if (step + 1) % grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)



        #   Evaluation 
        lora_model.eval()
        with torch.inference_mode():  
            pbar = tqdm(test_loader, desc=f"Eval Epoch {epoch+1}/{num_epochs}")
            for step, batch in enumerate(pbar):
                batch = {k: v.to(device) for k, v in batch.items()}

                with autocast(device_type="cuda", dtype=torch.float16):
                    loss = lora_model(**batch).loss

                eval_loss += loss.item()
                testing_steps += 1
                pbar.set_postfix(loss=eval_loss / testing_steps)
                
        print(f"Epoch {epoch+1} avg train loss: {total_loss/training_steps:.4f}")
        print(f"Epoch {epoch+1} avg test loss: {eval_loss/testing_steps:.4f}")

    lora_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved adapter to:", args.output_dir)


    



    
