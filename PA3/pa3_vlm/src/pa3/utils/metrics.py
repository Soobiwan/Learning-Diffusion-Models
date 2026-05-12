from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np
import torch


def normalize_answer(text: str) -> str:
    return " ".join(text.lower().strip().replace(".", "").split())


def exact_match(preds: Iterable[str], targets: Iterable[str]) -> float:
    pairs = list(zip(preds, targets))
    if not pairs:
        return 0.0
    return float(np.mean([normalize_answer(p) == normalize_answer(t) for p, t in pairs]))


def grouped_accuracy(preds: List[str], targets: List[str], groups: List[str]) -> Dict[str, float]:
    buckets = defaultdict(list)
    for p, t, g in zip(preds, targets, groups):
        buckets[g].append(normalize_answer(p) == normalize_answer(t))
    return {k: float(np.mean(v)) for k, v in sorted(buckets.items())}


@torch.no_grad()
def perplexity_from_loss(losses: List[float]) -> float:
    if not losses:
        return float("inf")
    return float(np.exp(np.mean(losses)))


def topk_tokens(logits: torch.Tensor, tokenizer, k: int = 5):
    values, ids = logits.float().topk(k)
    return [(tokenizer.decode([int(i)]), float(v)) for i, v in zip(ids, values)]

