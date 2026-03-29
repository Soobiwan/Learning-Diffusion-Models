"""Generalized Advantage Estimation."""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute token-wise GAE advantages and returns."""
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)
    next_value = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)

    for timestep in reversed(range(rewards.size(1))):
        non_terminal = 1.0 - dones[:, timestep]
        delta = rewards[:, timestep] + gamma * next_value * non_terminal - values[:, timestep]
        next_advantage = delta + gamma * gae_lambda * non_terminal * next_advantage
        advantages[:, timestep] = next_advantage
        next_value = values[:, timestep]

    returns = advantages + values
    return advantages, returns
