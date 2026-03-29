"""PPO losses for policy and critic updates."""

from __future__ import annotations

import torch

from ..common.utils import masked_mean
from .base import LossOutput, Objective


def clipped_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PPO clipped surrogate loss."""
    ratios = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -masked_mean(torch.min(unclipped, clipped), mask)
    clipped_fraction = ((ratios < 1.0 - clip_range) | (ratios > 1.0 + clip_range)).float().mean()
    return policy_loss, ratios, clipped_fraction


def clipped_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    old_values: torch.Tensor | None = None,
    clip_range: float | None = None,
) -> torch.Tensor:
    """Compute optional clipped value loss."""
    if old_values is None or clip_range is None:
        return masked_mean((values - returns).square(), mask)
    clipped_values = old_values + torch.clamp(values - old_values, -clip_range, clip_range)
    unclipped = (values - returns).square()
    clipped = (clipped_values - returns).square()
    return 0.5 * masked_mean(torch.max(unclipped, clipped), mask)


class PPOObjective(Objective):
    """Token-level PPO objective with value loss."""

    name = "ppo"

    def __init__(
        self,
        clip_range: float = 0.2,
        value_clip_range: float | None = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.0,
    ) -> None:
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def compute(
        self,
        new_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor,
        old_values: torch.Tensor | None = None,
        entropy: torch.Tensor | None = None,
    ) -> LossOutput:
        policy_loss, ratios, clipped_fraction = clipped_policy_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask,
            clip_range=self.clip_range,
        )
        value_loss = clipped_value_loss(
            values=values,
            returns=returns,
            mask=mask,
            old_values=old_values,
            clip_range=self.value_clip_range,
        )
        entropy_bonus = masked_mean(entropy, mask) if entropy is not None else torch.tensor(
            0.0, device=new_logprobs.device
        )
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_bonus
        return LossOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "policy_loss": policy_loss.detach(),
                "value_loss": value_loss.detach(),
                "ratio_mean": ratios.mean().detach(),
                "clipped_fraction": clipped_fraction.detach(),
                "entropy": entropy_bonus.detach(),
            },
        )
