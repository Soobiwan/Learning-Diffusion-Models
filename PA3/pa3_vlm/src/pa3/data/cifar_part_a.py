from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from transformers import CLIPImageProcessor


CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
VEHICLES = {"airplane", "automobile", "ship", "truck"}
FLYING = {"airplane", "bird"}
ANIMALS = set(CIFAR_CLASSES) - VEHICLES
CAPTION_TEMPLATES = [
    "A small photo of a {c}.",
    "This image contains a {c}.",
    "A low-resolution picture showing a {c}.",
    "The main object is a {c}.",
    "A CIFAR-10 example of a {c}.",
    "There is a {c} in the image.",
]


def stratified_indices(targets, per_class: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for i, y in enumerate(targets):
        buckets[int(y)].append(i)
    out = []
    for y in sorted(buckets):
        ids = np.array(buckets[y])
        rng.shuffle(ids)
        out.extend(ids[:per_class].tolist())
    rng.shuffle(out)
    return out


class CIFARCaptionDataset(Dataset):
    def __init__(self, root="./data", train=True, per_class=1000, seed=42, clip_name="openai/clip-vit-base-patch32"):
        base = CIFAR10(root=root, train=train, download=True)
        self.indices = stratified_indices(base.targets, per_class, seed)
        self.base = base
        self.processor = CLIPImageProcessor.from_pretrained(clip_name)
        self.patch_cache = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        raw_idx = self.indices[idx]
        img, y = self.base[raw_idx]
        c = CIFAR_CLASSES[int(y)]
        caption = CAPTION_TEMPLATES[idx % len(CAPTION_TEMPLATES)].format(c=c)
        out = {"caption": caption, "class_name": c, "image": img, "raw_idx": raw_idx}
        if self.patch_cache is not None and raw_idx in self.patch_cache:
            out["clip_patches"] = self.patch_cache[raw_idx]
        else:
            out["pixel_values"] = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return out


def vqa_items_from_indices(base, indices) -> List[Dict]:
    items = []
    for raw_idx in indices:
        img, y = base[raw_idx]
        c = CIFAR_CLASSES[int(y)]
        query_class = c if raw_idx % 2 == 0 else CIFAR_CLASSES[(int(y) + 1) % len(CIFAR_CLASSES)]
        qas = [
            ("What object is shown?", c, "recognition"),
            (f"Is there a {query_class}?", "yes" if query_class == c else "no", "binary"),
            ("Vehicle or living thing?", "vehicle" if c in VEHICLES else "living", "abstraction"),
            ("Can it fly?", "yes" if c in FLYING else "no", "attribute"),
            ("Is this an animal?", "yes" if c in ANIMALS else "no", "category"),
        ]
        for q, a, skill in qas:
            items.append({"image": img, "question": q, "answer": a, "skill": skill, "class_name": c, "raw_idx": raw_idx})
    return items


class CIFARVQADataset(Dataset):
    def __init__(self, root="./data", train=True, per_class=1000, seed=42, clip_name="openai/clip-vit-base-patch32"):
        base = CIFAR10(root=root, train=train, download=True)
        self.indices = stratified_indices(base.targets, per_class, seed)
        self.items = vqa_items_from_indices(base, self.indices)
        self.processor = CLIPImageProcessor.from_pretrained(clip_name)
        self.patch_cache = None
        self.base = base

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        out = {k: v for k, v in ex.items() if k != "image"}
        raw_idx = ex["raw_idx"]
        if self.patch_cache is not None and raw_idx in self.patch_cache:
            out["clip_patches"] = self.patch_cache[raw_idx]
        else:
            out["pixel_values"] = self.processor(images=ex["image"], return_tensors="pt")["pixel_values"][0]
        return out


def caption_collate(batch, tokenizer, max_length=96):
    out = {}
    if "clip_patches" in batch[0]:
        out["clip_patches"] = torch.stack([b["clip_patches"] for b in batch])
    else:
        out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
    caps = [b["caption"] for b in batch]
    tok = tokenizer(caps, add_special_tokens=False, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    out.update({"caption_ids": tok.input_ids, "caption_mask": tok.attention_mask, "captions": caps})
    return out


def vqa_text_collate(batch, tokenizer, max_length=128):
    out = {}
    if "clip_patches" in batch[0]:
        out["clip_patches"] = torch.stack([b["clip_patches"] for b in batch])
    else:
        out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
    q = [b["question"] for b in batch]
    a = [b["answer"] for b in batch]
    qtok = tokenizer(q, add_special_tokens=False, padding=True, truncation=True, max_length=max_length // 2, return_tensors="pt")
    answer_rows = [
        torch.tensor(tokenizer(x, add_special_tokens=False, truncation=True, max_length=15).input_ids + [tokenizer.eos_token_id], dtype=torch.long)
        for x in a
    ]
    max_answer = max(row.numel() for row in answer_rows)
    answer_ids = torch.full((len(answer_rows), max_answer), tokenizer.pad_token_id, dtype=torch.long)
    answer_mask = torch.zeros((len(answer_rows), max_answer), dtype=torch.long)
    for i, row in enumerate(answer_rows):
        answer_ids[i, :row.numel()] = row
        answer_mask[i, :row.numel()] = 1
    out.update({
        "question_ids": qtok.input_ids, "question_mask": qtok.attention_mask,
        "answer_ids": answer_ids, "answer_mask": answer_mask,
        "questions": q, "answers": a, "skills": [b["skill"] for b in batch], "classes": [b["class_name"] for b in batch],
    })
    return out


@torch.no_grad()
def extract_clip_patches(clip, pixel_values):
    outputs = clip(pixel_values=pixel_values)
    print_once = getattr(clip, "_pa3_printed_tokens", False)
    if not print_once:
        print(f"Sanity: CLIP should output 50 tokens before discarding CLS -> {tuple(outputs.last_hidden_state.shape)}")
        clip._pa3_printed_tokens = True
    patches = outputs.last_hidden_state[:, 1:, :]
    if not getattr(clip, "_pa3_printed_patches", False):
        print(f"Sanity: Part A patch tokens should be [B, 49, 768] -> {tuple(patches.shape)}")
        clip._pa3_printed_patches = True
    return patches


@torch.no_grad()
def cache_clip_patches(dataset, clip, device, batch_size: int = 128, dtype: torch.dtype = torch.float16):
    """Cache frozen CLIP patch tokens once per unique CIFAR image."""
    if getattr(dataset, "patch_cache", None):
        return dataset
    raw_indices = list(dict.fromkeys(dataset.indices))
    cache = {}
    was_training = clip.training
    clip.eval()
    processor = dataset.processor
    for start in range(0, len(raw_indices), batch_size):
        ids = raw_indices[start:start + batch_size]
        images = [dataset.base[i][0] for i in ids]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        patches = extract_clip_patches(clip, pixels).detach().to("cpu", dtype=dtype)
        for raw_idx, patch in zip(ids, patches):
            cache[raw_idx] = patch
    dataset.patch_cache = cache
    if was_training:
        clip.train()
    print(f"Cached CLIP patches: {len(cache)} unique images for {dataset.__class__.__name__}")
    return dataset
