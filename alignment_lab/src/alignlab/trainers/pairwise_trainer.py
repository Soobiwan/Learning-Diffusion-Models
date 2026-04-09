"""Shared trainer for DPO/SimPO/SamPO-like objectives."""

from __future__ import annotations

from typing import Any

import torch

from ..objectives.base import Objective
from ..rollout.logprobs import sequence_logprobs_from_logits
from .base import BaseTrainer


class PairwiseTrainer(BaseTrainer):
    """Reusable pairwise trainer that delegates loss math to objectives."""

    def __init__(
        self,
        model: torch.nn.Module,
        objective: Objective,
        reference_model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.objective = objective
        self.reference_model = self.prepare_auxiliary_module(reference_model)
        if self.reference_model is not None:
            self.reference_model.eval()

    def _sequence_logprobs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_logprobs, _ = sequence_logprobs_from_logits(outputs.logits, labels=labels)
        return sequence_logprobs

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        policy_chosen = self._sequence_logprobs(
            self.model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        )
        policy_rejected = self._sequence_logprobs(
            self.model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )
        if self.reference_model is None:
            raise ValueError("PairwiseTrainer requires a reference model for DPO-style objectives.")
        with torch.no_grad():
            reference_chosen = self._sequence_logprobs(
                self.reference_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            reference_rejected = self._sequence_logprobs(
                self.reference_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )
        loss_output = self.objective.compute(
            policy_chosen_logps=policy_chosen,
            policy_rejected_logps=policy_rejected,
            reference_chosen_logps=reference_chosen,
            reference_rejected_logps=reference_rejected,
        )
        chosen_response_lengths = batch["chosen_response_mask"].sum(dim=-1).float().mean()
        rejected_response_lengths = batch["rejected_response_mask"].sum(dim=-1).float().mean()
        loss_output.metrics["chosen_response_length"] = chosen_response_lengths.detach()
        loss_output.metrics["rejected_response_length"] = rejected_response_lengths.detach()
        loss_output.metrics["response_length_gap"] = (chosen_response_lengths - rejected_response_lengths).detach()
        return loss_output.loss, self.log_output(loss_output.metrics)

    def train_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        batch = self.move_batch_to_device(batch)
        loss, metrics = self.compute_loss(batch)
        self._backward_and_step(loss)
        return metrics
