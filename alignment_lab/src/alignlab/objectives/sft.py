"""Masked-response supervised fine-tuning objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import LossOutput, Objective


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Compute response-masked cross entropy for decoder-only models."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


class SFTObjective(Objective):
    """Compute masked next-token loss over response tokens only."""

    name = "sft"

    def compute(self, logits: torch.Tensor, labels: torch.Tensor) -> LossOutput:
        loss = masked_cross_entropy(logits=logits, labels=labels)
        return LossOutput(loss=loss, metrics={"loss": loss.detach()})
