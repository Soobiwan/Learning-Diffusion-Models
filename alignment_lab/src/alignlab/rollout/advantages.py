"""Advantage transforms for PPO and GRPO."""

from __future__ import annotations

import torch


def normalize_advantages(advantages: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Normalize advantages across all valid elements."""
    mean = advantages.mean()
    std = advantages.std(unbiased=False).clamp_min(eps)
    return (advantages - mean) / std


def group_relative_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute group-relative normalized advantages from scalar rewards.

    Expects rewards shaped as `[num_groups * group_size]`.
    """
    if rewards.numel() % group_size != 0:
        raise ValueError("Reward count must be divisible by group size.")
    grouped = rewards.view(-1, group_size)
    group_mean = grouped.mean(dim=-1, keepdim=True)
    group_std = grouped.std(dim=-1, unbiased=False, keepdim=True)
    advantages = (grouped - group_mean) / group_std.clamp_min(eps)
    return advantages.view(-1), group_std.squeeze(-1)


def broadcast_sequence_advantages(
    sequence_advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Broadcast per-sequence scalar advantages across response tokens."""
    return sequence_advantages.unsqueeze(-1) * response_mask.to(sequence_advantages.dtype)
