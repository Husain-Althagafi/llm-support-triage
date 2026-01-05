from datasets import load_from_disk, load_dataset
import os
from dict_mappings import LABEL_TO_TRIAGE, FIRST_STEPS
# from prompt import prompt
import json
import shutil
import gc

RAW_DIR = "data/banking77/raw"
PROC_DIR = "data/banking77/processed_v1"

def load_banking77():
    train_path = os.path.join(RAW_DIR, "train")
    test_path  = os.path.join(RAW_DIR, "test")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f'Banking77 dataset found, loading from disk...')
        train_ds = load_from_disk(train_path)
        test_ds = load_from_disk(test_path)
    else:
        print(f'Banking77 dataset not found on disk, downloading from huggingface...')
        ds = load_dataset('PolyAI/banking77')
        train_ds = ds['train']
        test_ds = ds['test']
        os.makedirs(RAW_DIR, exist_ok=True)
        train_ds.save_to_disk(train_path)
        test_ds.save_to_disk(test_path)

    return train_ds, test_ds

def map_labels(sample, labels):
    label = labels[sample['label']]
    category, priority = LABEL_TO_TRIAGE.get(label, ("General Inquiry", "P3"))

    output = {
        'category': category,
        'priority': priority,
        "first_response_steps": FIRST_STEPS
    }

    prompt = (
        "You are a support triage assistant.\n"
        "Return STRICT JSON with keys: category, priority, first_response_steps (array of strings).\n"
        f"Ticket: {sample['text']}\n"
        "JSON:"
    )

    text = prompt + '\n' + json.dumps(output, ensure_ascii=False)

    return {
        'text': text,
        'category': category,
        'priority': priority,
        'first_response_steps': FIRST_STEPS
    }

if __name__ == '__main__':
    print(f'Loading datasets...')
    train_ds, test_ds = load_banking77()
    print(f'Datasets loaded.')

    print(f'Running preprocessing...')
    labels = train_ds.features['label'].names
    proc_train = train_ds.map(lambda x: map_labels(x, labels), remove_columns=['label'])
    proc_test = test_ds.map(lambda x: map_labels(x, labels), remove_columns=['label'])
    print(f'Preprocessing done.')

    print(f'Saving processed datasets...')
    os.makedirs(PROC_DIR, exist_ok=True)
    train_path = os.path.join(PROC_DIR, "train")
    test_path = os.path.join(PROC_DIR, "test")

    if not os.path.exists(train_path) and not os.path.exists(test_path):
        proc_train.save_to_disk(train_path)
        proc_test.save_to_disk(test_path)
        print(f'Processed datasets saved to {PROC_DIR} /train and /test.')

    else:
        print(f'Saving of preprocessed dataset failed, delete existing preprocessed dataset first.')