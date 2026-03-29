"""Trainer for reward models."""

from __future__ import annotations

from typing import Any

import torch

from ..objectives.reward_model import RewardModelObjective
from .base import BaseTrainer


class RewardModelTrainer(BaseTrainer):
    """Pairwise ranking trainer for scalar reward models."""

    def __init__(
        self,
        model: torch.nn.Module,
        beta: float = 1.0,
        regularization: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.objective = RewardModelObjective(beta=beta, regularization=regularization)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        chosen = self.model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        ).logits.squeeze(-1)
        rejected = self.model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        ).logits.squeeze(-1)
        loss_output = self.objective.compute(chosen_scores=chosen, rejected_scores=rejected)
        return loss_output.loss, self.log_output(loss_output.metrics)

    def train_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        batch = self.move_batch_to_device(batch)
        loss, metrics = self.compute_loss(batch)
        self._backward_and_step(loss)
        return metrics
