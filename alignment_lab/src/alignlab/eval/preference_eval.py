"""Preference optimization evaluation helpers."""

from __future__ import annotations

import torch


def preference_accuracy(logits: torch.Tensor) -> float:
    """Compute binary preference accuracy from signed margins."""
    return float((logits > 0).float().mean().item())
