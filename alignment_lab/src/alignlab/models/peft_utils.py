"""PEFT/LoRA helpers."""

from __future__ import annotations

from typing import Any

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except ImportError:  # pragma: no cover - exercised only in minimal environments
    LoraConfig = None  # type: ignore[assignment]
    TaskType = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    prepare_model_for_kbit_training = None  # type: ignore[assignment]

from ..common.utils import count_parameters
from .specs import ModelSpec


def build_lora_config(spec: ModelSpec) -> LoraConfig:
    """Create a LoRA config from a `ModelSpec`."""
    if LoraConfig is None:
        raise ImportError("peft is required to build LoRA configs.")
    task_type = getattr(TaskType, spec.task_type.upper(), spec.task_type) if TaskType is not None else spec.task_type
    modules_to_save = ["score"] if task_type == getattr(TaskType, "SEQ_CLS", "SEQ_CLS") else None
    return LoraConfig(
        r=spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        bias="none",
        task_type=task_type,
        target_modules=spec.lora_target_modules,
        modules_to_save=modules_to_save,
    )


def enable_gradient_checkpointing(model: Any) -> Any:
    """Enable gradient checkpointing when the model supports it."""
    gradient_checkpointing_enable = getattr(model, "gradient_checkpointing_enable", None)
    if callable(gradient_checkpointing_enable):
        gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model


def enable_input_require_grads(model: Any) -> Any:
    """Ensure PEFT inputs require gradients when the model supports it."""
    enable_inputs = getattr(model, "enable_input_require_grads", None)
    if callable(enable_inputs):
        enable_inputs()
    return model


def prepare_model_for_peft_training(model: Any, is_quantized: bool = False) -> Any:
    """Prepare a model for PEFT training without re-creating adapters."""
    if is_quantized:
        if prepare_model_for_kbit_training is None:
            raise ImportError("peft is required for k-bit LoRA preparation.")
        model = prepare_model_for_kbit_training(model)
    return model


def maybe_apply_lora(model: Any, spec: ModelSpec, is_quantized: bool = False) -> Any:
    """Apply LoRA to a model when enabled in the spec."""
    if not spec.use_lora:
        return model
    if get_peft_model is None:
        raise ImportError("peft is required when `use_lora=True`.")
    model = prepare_model_for_peft_training(model, is_quantized=is_quantized)
    peft_model = get_peft_model(model, build_lora_config(spec))
    if spec.task_type.upper() == "SEQ_CLS":
        active_adapter = getattr(peft_model, "active_adapter", "default")
        if isinstance(active_adapter, str) and active_adapter in peft_model.peft_config:
            peft_model.peft_config[active_adapter].modules_to_save = ["score"]
    return peft_model


def trainable_parameter_summary(model: Any) -> dict[str, int]:
    """Return trainable and total parameter counts."""
    return {
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "total_parameters": count_parameters(model, trainable_only=False),
    }
