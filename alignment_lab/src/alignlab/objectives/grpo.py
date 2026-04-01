"""GRPO-style policy objective."""

from __future__ import annotations

import torch

from .base import LossOutput, Objective


def _sequence_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    token_totals = (values * mask_f).sum(dim=-1)
    token_counts = mask_f.sum(dim=-1).clamp_min(1.0)
    return token_totals / token_counts


def clipped_group_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the clipped GRPO/RLVR surrogate with per-sequence normalization."""
    ratios = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * advantages
    token_objective = torch.min(unclipped, clipped)
    valid_sequences = mask.any(dim=-1)
    if not torch.any(valid_sequences):
        zero = torch.tensor(0.0, device=new_logprobs.device)
        return zero, ratios, zero
    policy_loss = -_sequence_mean(token_objective, mask)[valid_sequences].mean()
    clipped_fraction = (
        ((ratios < 1.0 - clip_range) | (ratios > 1.0 + clip_range)).to(new_logprobs.dtype) * mask.to(new_logprobs.dtype)
    ).sum() / mask.to(new_logprobs.dtype).sum().clamp_min(1.0)
    return policy_loss, ratios, clipped_fraction


class GRPOObjective(Objective):
    """Group-relative policy optimization objective."""

    name = "grpo"

    def __init__(self, beta_kl: float = 0.02, clip_range: float = 0.2) -> None:
        self.beta_kl = beta_kl
        self.clip_range = clip_range

    def compute(
        self,
        new_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        kl_values: torch.Tensor | None = None,
    ) -> LossOutput:
        policy_loss, ratios, clipped_fraction = clipped_group_policy_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask,
            clip_range=self.clip_range,
        )
        valid_sequences = mask.any(dim=-1)
        if kl_values is None:
            kl_penalty = torch.tensor(0.0, device=new_logprobs.device)
        else:
            kl_penalty = _sequence_mean(kl_values, mask)[valid_sequences].mean()
        loss = policy_loss + self.beta_kl * kl_penalty
        return LossOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "policy_loss": policy_loss.detach(),
                "kl_penalty": kl_penalty.detach(),
                "ratio_mean": ratios[mask].mean().detach() if torch.any(mask) else torch.tensor(1.0, device=ratios.device),
                "clipped_fraction": clipped_fraction.detach(),
            },
        )
