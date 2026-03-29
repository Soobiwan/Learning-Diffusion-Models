"""Trainer for supervised fine-tuning."""

from __future__ import annotations

from typing import Any

import torch

from ..models.generation import generate_batched
from ..objectives.sft import SFTObjective
from .base import BaseTrainer


class SFTTrainer(BaseTrainer):
    """Shared training logic for masked-response SFT."""

    def __init__(self, model: torch.nn.Module, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.objective = SFTObjective()

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss_output = self.objective.compute(logits=outputs.logits, labels=batch["labels"])
        metrics = dict(loss_output.metrics)
        metrics["perplexity"] = torch.exp(loss_output.loss.detach())
        return loss_output.loss, self.log_output(metrics)

    def train_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        batch = self.move_batch_to_device(batch)
        loss, metrics = self.compute_loss(batch)
        self._backward_and_step(loss)
        return metrics

    @torch.no_grad()
    def sample_generations(
        self,
        tokenizer: Any,
        prompt_batch: dict[str, torch.Tensor],
        generation_config: dict[str, Any],
    ) -> list[str]:
        self.model.eval()
        prompt_batch = self.move_batch_to_device(prompt_batch)
        generated = generate_batched(
            model=self.model,
            tokenizer=tokenizer,
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            generation_config=generation_config,
        )
        return list(generated["responses"])
