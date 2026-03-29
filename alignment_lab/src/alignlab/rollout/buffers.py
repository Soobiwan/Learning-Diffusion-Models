"""Rollout batch containers with CPU caching support."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Iterator

import torch


@dataclass
class RolloutBatch:
    """Container for PPO/GRPO/RLVR updates."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor | None = None
    values: torch.Tensor | None = None
    old_values: torch.Tensor | None = None
    rewards: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    sequence_advantages: torch.Tensor | None = None
    prompts: list[str] | None = None
    responses: list[str] | None = None
    meta: dict[str, Any] | None = None

    def to(self, device: torch.device | str) -> "RolloutBatch":
        """Move tensor fields to the target device."""
        updates: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                updates[field.name] = value.to(device)
        return replace(self, **updates)

    def cpu(self) -> "RolloutBatch":
        """Move rollout storage to CPU."""
        return self.to("cpu")

    def iter_minibatches(self, batch_size: int) -> Iterator["RolloutBatch"]:
        """Yield simple contiguous minibatches."""
        total = self.input_ids.size(0)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            updates: dict[str, Any] = {}
            for field in fields(self):
                value = getattr(self, field.name)
                if isinstance(value, torch.Tensor):
                    updates[field.name] = value[start:end]
                elif isinstance(value, list):
                    updates[field.name] = value[start:end]
                else:
                    updates[field.name] = value
            yield replace(self, **updates)
