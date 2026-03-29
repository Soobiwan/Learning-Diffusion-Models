"""Direct Preference Optimization objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import LossOutput, Objective


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> LossOutput:
    """Compute the DPO logistic loss."""
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_logratios - reference_logratios)
    positive = -(1.0 - label_smoothing) * F.logsigmoid(logits)
    negative = -label_smoothing * F.logsigmoid(-logits)
    loss = (positive + negative).mean()
    accuracy = (logits > 0).float().mean()
    return LossOutput(
        loss=loss,
        metrics={
            "loss": loss.detach(),
            "preference_accuracy": accuracy.detach(),
            "z_margin": logits.mean().detach(),
            "policy_logratio": policy_logratios.mean().detach(),
            "reference_logratio": reference_logratios.mean().detach(),
        },
    )


class DPOObjective(Objective):
    """Reference-based pairwise preference optimization."""

    name = "dpo"

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0) -> None:
        self.beta = beta
        self.label_smoothing = label_smoothing

    def compute(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> LossOutput:
        return dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            beta=self.beta,
            label_smoothing=self.label_smoothing,
        )
