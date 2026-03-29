"""PEFT/LoRA helpers."""

from __future__ import annotations

from typing import Any

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:  # pragma: no cover - exercised only in minimal environments
    LoraConfig = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    prepare_model_for_kbit_training = None  # type: ignore[assignment]

from ..common.utils import count_parameters
from .specs import ModelSpec


def build_lora_config(spec: ModelSpec) -> LoraConfig:
    """Create a LoRA config from a `ModelSpec`."""
    if LoraConfig is None:
        raise ImportError("peft is required to build LoRA configs.")
    return LoraConfig(
        r=spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        bias="none",
        task_type=spec.task_type,
        target_modules=spec.lora_target_modules,
    )


def maybe_apply_lora(model: Any, spec: ModelSpec, is_quantized: bool = False) -> Any:
    """Apply LoRA to a model when enabled in the spec."""
    if not spec.use_lora:
        return model
    if get_peft_model is None:
        raise ImportError("peft is required when `use_lora=True`.")
    if is_quantized:
        if prepare_model_for_kbit_training is None:
            raise ImportError("peft is required for k-bit LoRA preparation.")
        model = prepare_model_for_kbit_training(model)
    return get_peft_model(model, build_lora_config(spec))


def trainable_parameter_summary(model: Any) -> dict[str, int]:
    """Return trainable and total parameter counts."""
    return {
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "total_parameters": count_parameters(model, trainable_only=False),
    }
