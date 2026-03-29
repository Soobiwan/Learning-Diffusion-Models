"""Centralized model loading functions."""

from __future__ import annotations

from typing import Any

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

from .peft_utils import maybe_apply_lora
from .specs import ModelSpec
from .tokenizer_utils import resolve_torch_dtype
from .value import CausalValueModel, _hidden_size_from_config


def make_quantization_config(mode: str | None) -> BitsAndBytesConfig | None:
    """Build a bitsandbytes quantization config when requested."""
    if mode is None:
        return None
    normalized = mode.lower()
    if normalized == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=resolve_torch_dtype("fp16"))
    if normalized == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization mode '{mode}'.")


def _base_model_kwargs(spec: ModelSpec) -> dict[str, Any]:
    quantization_config = make_quantization_config(spec.quantization)
    kwargs: dict[str, Any] = {
        "trust_remote_code": spec.trust_remote_code,
        "torch_dtype": resolve_torch_dtype(spec.dtype),
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    return kwargs


def load_policy_model(spec: ModelSpec) -> Any:
    """Load a policy causal LM, optionally with LoRA."""
    model = AutoModelForCausalLM.from_pretrained(spec.hf_path, **_base_model_kwargs(spec))
    model.config.use_cache = False
    return maybe_apply_lora(model, spec, is_quantized=spec.quantization is not None)


def load_reward_model(spec: ModelSpec) -> Any:
    """Load a scalar reward model using sequence classification."""
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.hf_path,
        num_labels=1,
        **_base_model_kwargs(spec),
    )
    if getattr(model.config, "pad_token_id", None) is None and getattr(model.config, "eos_token_id", None) is not None:
        model.config.pad_token_id = model.config.eos_token_id
    return maybe_apply_lora(model, spec, is_quantized=spec.quantization is not None)


def load_reference_model(spec: ModelSpec) -> Any:
    """Load a frozen reference policy."""
    model = AutoModelForCausalLM.from_pretrained(spec.hf_path, **_base_model_kwargs(spec))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_value_model(spec: ModelSpec) -> CausalValueModel:
    """Load a value model as a decoder-only backbone plus value head."""
    config = AutoConfig.from_pretrained(spec.hf_path, trust_remote_code=spec.trust_remote_code)
    backbone = AutoModelForCausalLM.from_pretrained(spec.hf_path, **_base_model_kwargs(spec))
    backbone.config.use_cache = False
    backbone = maybe_apply_lora(backbone, spec, is_quantized=spec.quantization is not None)
    return CausalValueModel(backbone=backbone, hidden_size=_hidden_size_from_config(config))
