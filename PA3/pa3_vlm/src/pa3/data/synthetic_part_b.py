from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


CLASSES = ["spiral", "triangle", "circle", "cross", "checkerboard", "gradient"]
GEOMETRIC = {"triangle", "circle", "cross"}
SYMM = {"spiral": "0", "triangle": "1", "circle": "infinite", "cross": "4", "checkerboard": "4", "gradient": "0"}


def _draw(name: str, rng) -> np.ndarray:
    img = np.zeros((16, 16, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:16, 0:16]
    color = rng.uniform(0.25, 1.0, size=3)
    if name == "circle":
        mask = (xx - 7.5) ** 2 + (yy - 7.5) ** 2 <= rng.integers(20, 35)
        img[mask] = color
    elif name == "triangle":
        mask = (yy > 3) & (yy < 13) & (np.abs(xx - 8) < yy - 2)
        img[mask] = color
    elif name == "cross":
        img[6:10, :] = color
        img[:, 6:10] = color
    elif name == "checkerboard":
        mask = ((xx // 4 + yy // 4) % 2) == 0
        img[mask] = color
        img[~mask] = color[::-1] * 0.4
    elif name == "gradient":
        img[..., 0] = xx / 15
        img[..., 1] = yy / 15
        img[..., 2] = (xx + yy) / 30
    elif name == "spiral":
        cx, cy = 7.5, 7.5
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        th = np.arctan2(yy - cy, xx - cx)
        mask = np.abs(((r * 1.8 - th) % (2 * np.pi)) - np.pi) < 0.45
        img[mask] = color
    img += rng.normal(0, 0.025, img.shape)
    return np.clip(img, 0, 1)


def generate_dataset(n_per_class: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for i, c in enumerate(CLASSES):
        for _ in range(n_per_class):
            images.append(_draw(c, rng))
            labels.append(i)
    images = np.stack(images)
    labels = np.array(labels)
    train_idx, val_idx = [], []
    for i in range(len(CLASSES)):
        idx = np.where(labels == i)[0]
        rng.shuffle(idx)
        split = int(0.8 * len(idx))
        train_idx.extend(idx[:split])
        val_idx.extend(idx[split:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return images, labels, np.array(train_idx), np.array(val_idx)


def save_synthetic_grid(images, labels, path: str):
    fig, axes = plt.subplots(6, 5, figsize=(6, 7))
    for c in range(6):
        ids = np.where(labels == c)[0][:5]
        for j, idx in enumerate(ids):
            axes[c, j].imshow(images[idx])
            axes[c, j].axis("off")
            if j == 0:
                axes[c, j].set_title(CLASSES[c])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


class SyntheticImageDataset(Dataset):
    def __init__(self, split="train", n_per_class=1000, seed=42):
        images, labels, train_idx, val_idx = generate_dataset(n_per_class, seed)
        idx = train_idx if split == "train" else val_idx
        self.images = images[idx]
        self.labels = labels[idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx]).permute(2, 0, 1).float()
        label = int(self.labels[idx])
        return {"image": img, "class_name": CLASSES[label], "label": label}


def build_vqa(ex: Dict, sample_idx: int = 0) -> List[Dict]:
    c = ex["class_name"]
    label = int(ex.get("label", CLASSES.index(c)))
    query_class = c if sample_idx % 2 == 0 else CLASSES[(label + 1) % len(CLASSES)]
    qas = [
        ("What shape is in this image?", c, "shape"),
        (f"Is there a {query_class}?", "yes" if query_class == c else "no", "binary"),
        ("Geometric or non-geometric?", "geometric" if c in GEOMETRIC else "non-geometric", "geometry"),
        ("How many axes of symmetry?", SYMM[c], "symmetry"),
    ]
    return [{"image": ex["image"], "class_name": c, "question": q, "answer": a, "skill": s} for q, a, s in qas]


class SyntheticVQADataset(Dataset):
    def __init__(self, split="train", n_per_class=1000, seed=42):
        base = SyntheticImageDataset(split, n_per_class, seed)
        self.items = []
        for i in range(len(base)):
            self.items.extend(build_vqa(base[i], i))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class SyntheticImageGenDataset(Dataset):
    templates = ["Draw a 16 by 16 {c}.", "Generate a tiny {c} image.", "Create visual tokens for a {c}."]

    def __init__(self, split="train", n_per_class=1000, seed=42):
        self.base = SyntheticImageDataset(split, n_per_class, seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        ex = self.base[idx]
        return {"image": ex["image"], "class_name": ex["class_name"], "prompt": self.templates[idx % 3].format(c=ex["class_name"])}


@torch.no_grad()
def encode_images_to_visual_ids(vqvae, images, device, original_vocab: int):
    zq, ids, *_ = vqvae.encode(images.to(device))
    return ids.reshape(ids.shape[0], -1).cpu() + original_vocab + 2


def _encode_multimodal_sample(b: Dict, tokenizer, visual_ids: torch.Tensor, original_vocab: int, mode: str, max_length: int):
    image_id, image_end_id = original_vocab, original_vocab + 1
    vis = visual_ids.tolist()
    if mode == "vqa":
        q = tokenizer(b["question"], add_special_tokens=False).input_ids
        a = tokenizer(b["answer"], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        ids = [tokenizer.bos_token_id, image_id] + vis + [image_end_id] + q + a
        lab = [-100] * (2 + len(vis) + 1 + len(q)) + a
        token_types = ["BOS", "<image>"] + ["VIS"] * len(vis) + ["</image>"] + ["Q"] * len(q) + ["A"] * len(a)
    elif mode == "imggen":
        p = tokenizer(b["prompt"], add_special_tokens=False).input_ids
        ids = [tokenizer.bos_token_id] + p + [image_id] + vis + [image_end_id, tokenizer.eos_token_id]
        lab = [-100] * (1 + len(p) + 1) + vis + [image_end_id, tokenizer.eos_token_id]
        token_types = ["BOS"] + ["P"] * len(p) + ["<image>"] + ["VIS"] * len(vis) + ["</image>", "EOS"]
    else:
        raise ValueError(mode)
    ids, lab, token_types = ids[:max_length], lab[:max_length], token_types[:max_length]
    meta = {k: v for k, v in b.items() if k != "image"}
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(lab, dtype=torch.long),
        "token_types": token_types,
        "meta": meta,
    }


class PreencodedMultimodalDataset(Dataset):
    """Tokenise Part B samples once so the LM stage no longer calls the VQ-VAE per batch."""

    def __init__(
        self,
        base: Dataset,
        tokenizer,
        vqvae,
        device,
        original_vocab: int,
        mode: str,
        max_length: int = 192,
        batch_size: int = 128,
    ):
        self.items = []
        was_training = vqvae.training
        vqvae.eval()
        for start in range(0, len(base), batch_size):
            batch = [base[i] for i in range(start, min(start + batch_size, len(base)))]
            images = torch.stack([b["image"] for b in batch])
            vis = encode_images_to_visual_ids(vqvae, images, device, original_vocab)
            for ex, visual_ids in zip(batch, vis):
                self.items.append(_encode_multimodal_sample(ex, tokenizer, visual_ids, original_vocab, mode, max_length))
        if was_training:
            vqvae.train()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def tokenized_multimodal_collate(batch, tokenizer):
    seqs = [b["input_ids"] for b in batch]
    labs = [b["labels"] for b in batch]
    max_len = max(x.numel() for x in seqs)
    input_ids = torch.full((len(seqs), max_len), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((len(seqs), max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    left_pad = tokenizer.padding_side == "left"
    for i, (x, y) in enumerate(zip(seqs, labs)):
        start = max_len - x.numel() if left_pad else 0
        input_ids[i, start:start + x.numel()] = x
        labels[i, start:start + y.numel()] = y
        mask[i, start:start + x.numel()] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": mask,
        "meta": [b.get("meta", {}) for b in batch],
        "token_types": [b.get("token_types", []) for b in batch],
    }


def multimodal_collate(batch, tokenizer, vqvae, device, original_vocab: int, mode: str, max_length=192):
    images = torch.stack([b["image"] for b in batch])
    vis = encode_images_to_visual_ids(vqvae, images, device, original_vocab)
    encoded = [_encode_multimodal_sample(b, tokenizer, vis[i], original_vocab, mode, max_length) for i, b in enumerate(batch)]
    return tokenized_multimodal_collate(encoded, tokenizer)
