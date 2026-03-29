"""Reward model evaluation helpers."""

from __future__ import annotations

from typing import Sequence

import torch


def reward_preference_accuracy(chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> float:
    """Compute pairwise reward accuracy."""
    return float((chosen_scores > rejected_scores).float().mean().item())


def reward_model_win_rate_vs_sft(
    aligned_scores: Sequence[float] | torch.Tensor,
    sft_scores: Sequence[float] | torch.Tensor,
) -> float:
    """Compute reward-model win rate of aligned responses against an SFT baseline."""
    aligned = torch.as_tensor(aligned_scores, dtype=torch.float32)
    baseline = torch.as_tensor(sft_scores, dtype=torch.float32)
    return float((aligned > baseline).float().mean().item())
