"""Reward model evaluation helpers."""

from __future__ import annotations

import torch


def reward_preference_accuracy(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> float:
    """Compute pairwise reward accuracy."""
    return float((chosen_scores > rejected_scores).float().mean().item())
