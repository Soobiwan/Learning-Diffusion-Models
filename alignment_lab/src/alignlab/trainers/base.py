"""Base trainer utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import AdamW

try:
    from accelerate import Accelerator
except ImportError:  # pragma: no cover - fallback for minimal environments
    class Accelerator:  # type: ignore[override]
        def __init__(self, mixed_precision: str = "no") -> None:
            self.mixed_precision = mixed_precision
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def backward(self, loss: torch.Tensor) -> None:
            loss.backward()

        def clip_grad_norm_(self, parameters, max_norm: float) -> torch.Tensor:
            return torch.nn.utils.clip_grad_norm_(list(parameters), max_norm)

from ..common.logging import get_logger
from ..common.utils import tensor_dict_to_float


class BaseTrainer:
    """A lightweight accelerator-aware trainer base class."""

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1.0e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        mixed_precision: str = "no",
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        mp_mode = mixed_precision if (mixed_precision != "fp16" or torch.cuda.is_available()) else "no"
        self.accelerator = Accelerator(mixed_precision=mp_mode)
        self.model = model.to(self.accelerator.device)
        self.trainable_parameters: list[torch.nn.Parameter] = list(self.model.parameters())
        self.optimizer = optimizer or AdamW(self.trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        self.logger = get_logger(self.__class__.__name__)
        self.step = 0

    @property
    def device(self) -> torch.device:
        """Current accelerator device."""
        return self.accelerator.device

    def move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensor values in a batch to the trainer device."""
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _set_trainable_parameters(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Update the parameter list used for gradient clipping and optimization."""
        self.trainable_parameters = list(parameters)
        if not self.trainable_parameters:
            raise ValueError("Trainer received an empty parameter list.")

    def _backward_and_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.accelerator.backward(loss)
        if self.max_grad_norm is not None:
            self.accelerator.clip_grad_norm_(self.trainable_parameters, self.max_grad_norm)
        self.optimizer.step()
        self.step += 1

    def log_output(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Convert scalar metrics to plain floats."""
        return tensor_dict_to_float(metrics)
