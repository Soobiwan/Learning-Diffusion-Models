from typing import Dict, List

from datasets import load_dataset
import torch
from torch.utils.data import Dataset


def format_alpaca(ex: Dict[str, str]) -> Dict[str, str]:
    inst = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    if inp:
        prompt = f"Instruction:\n{inst}\n\nInput:\n{inp}\n\nResponse:\n"
    else:
        prompt = f"Instruction:\n{inst}\n\nResponse:\n"
    return {"prompt": prompt, "response": out}


class AlpacaDataset(Dataset):
    def __init__(self, n: int = 1000, seed: int = 42):
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        self.items = [format_alpaca(dict(x)) for x in ds]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = dict(self.items[idx])
        item["mode"] = "alpaca"
        return item


def alpaca_collate(batch: List[Dict], tokenizer, max_length: int = 192, forbidden_start_id: int = None):
    input_ids, labels = [], []
    for ex in batch:
        prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False).input_ids
        resp_ids = tokenizer(ex["response"], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        ids = [tokenizer.bos_token_id] + prompt_ids + resp_ids
        lab = [-100] * (1 + len(prompt_ids)) + resp_ids
        ids, lab = ids[:max_length], lab[:max_length]
        if forbidden_start_id is not None and any(i >= forbidden_start_id for i in ids):
            raise ValueError("Visual token leaked into Alpaca replay batch.")
        input_ids.append(torch.tensor(ids, dtype=torch.long))
        labels.append(torch.tensor(lab, dtype=torch.long))
    return _pad(input_ids, labels, tokenizer.pad_token_id, left_pad=tokenizer.padding_side == "left")


def _pad(input_ids, labels, pad_id, left_pad: bool = False):
    max_len = max(x.numel() for x in input_ids)
    ids = torch.full((len(input_ids), max_len), pad_id, dtype=torch.long)
    labs = torch.full((len(input_ids), max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(input_ids), max_len), dtype=torch.long)
    for i, (x, y) in enumerate(zip(input_ids, labels)):
        start = max_len - x.numel() if left_pad else 0
        ids[i, start:start + x.numel()] = x
        labs[i, start:start + y.numel()] = y
        mask[i, start:start + x.numel()] = 1
    return {"input_ids": ids, "labels": labs, "attention_mask": mask}
