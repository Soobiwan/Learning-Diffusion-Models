"""General-purpose utilities used across modules."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count module parameters."""
    params = module.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def tensor_dict_to_float(values: dict[str, Any]) -> dict[str, float]:
    """Detach tensors and convert scalar metrics to Python floats."""
    converted: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            converted[key] = float(value.detach().cpu().item())
        else:
            converted[key] = float(value)
    return converted


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Compute the mean over masked elements."""
    mask = mask.to(values.dtype)
    total = (values * mask).sum()
    denom = mask.sum().clamp_min(eps)
    return total / denom
