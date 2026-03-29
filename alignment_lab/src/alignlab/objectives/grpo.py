"""GRPO-style policy objective."""

from __future__ import annotations

import torch

from ..common.utils import masked_mean
from .base import LossOutput, Objective


class GRPOObjective(Objective):
    """Group-relative policy optimization objective."""

    name = "grpo"

    def __init__(self, beta_kl: float = 0.02) -> None:
        self.beta_kl = beta_kl

    def compute(
        self,
        token_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        kl_values: torch.Tensor | None = None,
    ) -> LossOutput:
        advantage_term = token_logprobs * advantages
        policy_loss = -masked_mean(advantage_term, mask)
        if kl_values is None:
            kl_penalty = torch.tensor(0.0, device=token_logprobs.device)
        else:
            kl_penalty = masked_mean(kl_values, mask)
        loss = policy_loss + self.beta_kl * kl_penalty
        return LossOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "policy_loss": policy_loss.detach(),
                "kl_penalty": kl_penalty.detach(),
            },
        )
