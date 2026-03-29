"""Reward model ranking loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import LossOutput, Objective


def pairwise_ranking_loss(
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
    beta: float = 1.0,
    regularization: float = 0.0,
) -> LossOutput:
    """Bradley-Terry style pairwise ranking loss."""
    margin = beta * (chosen_scores - rejected_scores)
    ranking_loss = -F.logsigmoid(margin).mean()
    regularizer = regularization * 0.5 * (chosen_scores.square().mean() + rejected_scores.square().mean())
    loss = ranking_loss + regularizer
    accuracy = (chosen_scores > rejected_scores).float().mean()
    return LossOutput(
        loss=loss,
        metrics={
            "loss": loss.detach(),
            "ranking_loss": ranking_loss.detach(),
            "reward_regularizer": regularizer.detach(),
            "reward_accuracy": accuracy.detach(),
            "reward_margin": (chosen_scores - rejected_scores).mean().detach(),
        },
    )


class RewardModelObjective(Objective):
    """Pairwise reward ranking objective."""

    name = "reward_model"

    def __init__(self, beta: float = 1.0, regularization: float = 0.0) -> None:
        self.beta = beta
        self.regularization = regularization

    def compute(self, chosen_scores: torch.Tensor, rejected_scores: torch.Tensor) -> LossOutput:
        return pairwise_ranking_loss(
            chosen_scores=chosen_scores,
            rejected_scores=rejected_scores,
            beta=self.beta,
            regularization=self.regularization,
        )
