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
        gradient_accumulation_steps: int = 1,
    ) -> None:
        mp_aliases = {
            "fp32": "no",
            "float32": "no",
            "float": "no",
            "none": "no",
        }
        normalized_mixed_precision = mp_aliases.get(str(mixed_precision).lower(), mixed_precision)
        mp_mode = normalized_mixed_precision if (normalized_mixed_precision != "fp16" or torch.cuda.is_available()) else "no"
        self.accelerator = Accelerator(mixed_precision=mp_mode)
        self.model = model.to(self.accelerator.device)
        self.trainable_parameters: list[torch.nn.Parameter] = list(self.model.parameters())
        self.optimizer = optimizer or AdamW(self.trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = max(int(gradient_accumulation_steps), 1)
        self.logger = get_logger(self.__class__.__name__)
        self.step = 0
        self.last_step_was_optimizer_step = False
        self.last_gradient_norm: float | None = None
        self._micro_steps_since_update = 0
        self.optimizer.zero_grad(set_to_none=True)

    @property
    def device(self) -> torch.device:
        """Current accelerator device."""
        return self.accelerator.device

    @property
    def has_pending_gradients(self) -> bool:
        """Whether gradients are currently accumulated but not stepped."""
        return self._micro_steps_since_update > 0

    @property
    def accumulation_progress(self) -> tuple[int, int]:
        """Human-readable accumulation progress for the most recent micro-step."""
        if self.last_step_was_optimizer_step:
            return self.gradient_accumulation_steps, self.gradient_accumulation_steps
        if self._micro_steps_since_update > 0:
            return self._micro_steps_since_update, self.gradient_accumulation_steps
        return 0, self.gradient_accumulation_steps

    @property
    def accumulation_status(self) -> str:
        """Formatted accumulation progress like '2/4'."""
        completed, total = self.accumulation_progress
        return f"{completed}/{total}"

    def move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensor values in a batch to the trainer device."""
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def prepare_auxiliary_module(self, module: Any) -> Any:
        """Place non-primary models on the trainer device when safe.

        Quantized bitsandbytes modules are already device-dispatched by
        transformers/accelerate and must not be moved with `.to(...)`.
        """
        if module is None:
            return None

        candidates = [module]
        for attribute in ("backbone", "model", "base_model", "policy_model"):
            child = getattr(module, attribute, None)
            if child is not None:
                candidates.append(child)

        manages_own_device = any(
            getattr(candidate, "is_loaded_in_4bit", False)
            or getattr(candidate, "is_loaded_in_8bit", False)
            or getattr(candidate, "hf_device_map", None) is not None
            for candidate in candidates
        )
        if manages_own_device:
            align_helper = getattr(module, "move_auxiliary_modules_to_backbone_device", None)
            if callable(align_helper):
                align_helper()
            return module
        return module.to(self.device)

    def _set_trainable_parameters(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Update the parameter list used for gradient clipping and optimization."""
        self.trainable_parameters = list(parameters)
        if not self.trainable_parameters:
            raise ValueError("Trainer received an empty parameter list.")

    def _optimizer_step(self) -> None:
        grad_norm = None
        if self.max_grad_norm is not None:
            grad_norm = self.accelerator.clip_grad_norm_(self.trainable_parameters, self.max_grad_norm)
        if grad_norm is not None:
            self.last_gradient_norm = float(grad_norm.detach().cpu().item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self._micro_steps_since_update = 0
        self.last_step_was_optimizer_step = True

    def _backward_and_step(self, loss: torch.Tensor) -> bool:
        self.last_step_was_optimizer_step = False
        scaled_loss = loss / float(self.gradient_accumulation_steps)
        self.accelerator.backward(scaled_loss)
        self._micro_steps_since_update += 1
        if self._micro_steps_since_update >= self.gradient_accumulation_steps:
            self._optimizer_step()
        return self.last_step_was_optimizer_step

    def flush(self) -> bool:
        """Apply any partially accumulated gradients."""
        self.last_step_was_optimizer_step = False
        if self._micro_steps_since_update <= 0:
            return False
        self._optimizer_step()
        return True

    def log_output(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Convert scalar metrics to plain floats."""
        return tensor_dict_to_float(metrics)
