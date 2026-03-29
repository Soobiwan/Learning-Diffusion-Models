"""Base objective abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch


@dataclass
class LossOutput:
    """Structured loss output returned by objective functions."""

    loss: torch.Tensor
    metrics: dict[str, torch.Tensor | float] = field(default_factory=dict)


class Objective(ABC):
    """Base interface for objectives."""

    name: str

    @abstractmethod
    def compute(self, *args, **kwargs) -> LossOutput:
        """Compute loss and metrics for a batch."""
