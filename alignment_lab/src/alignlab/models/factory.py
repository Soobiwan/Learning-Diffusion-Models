"""Centralized model loading functions."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Any

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

try:
    from peft import PeftConfig, PeftModel
except ImportError:  # pragma: no cover - exercised only in minimal environments
    PeftConfig = None  # type: ignore[assignment]
    PeftModel = None  # type: ignore[assignment]

from .peft_utils import (
    enable_gradient_checkpointing,
    enable_input_require_grads,
    maybe_apply_lora,
    prepare_model_for_peft_training,
)
from .specs import ModelSpec
from .tokenizer_utils import normalize_model_config_special_ids, resolve_torch_dtype
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


def _peft_checkpoint_dir(path: str) -> Path | None:
    checkpoint_dir = Path(path)
    if checkpoint_dir.is_dir() and (checkpoint_dir / "adapter_config.json").exists():
        return checkpoint_dir
    return None


def _freeze_module(module: Any) -> Any:
    """Freeze a module for inference use."""
    if hasattr(module, "eval"):
        module.eval()
    if hasattr(module, "parameters"):
        for parameter in module.parameters():
            parameter.requires_grad = False
    return module


@contextmanager
def _suppress_peft_reward_head_warning() -> Any:
    """Hide the misleading fresh-score warning before adapter restore."""
    logger = logging.getLogger("transformers.modeling_utils")

    class _RewardHeadWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            if "newly initialized" in message and "score.weight" in message:
                return False
            if "You should probably TRAIN this model on a down-stream task" in message:
                return False
            return True

    warning_filter = _RewardHeadWarningFilter()
    logger.addFilter(warning_filter)
    try:
        yield
    finally:
        logger.removeFilter(warning_filter)


def _load_causal_lm_base(spec: ModelSpec) -> Any:
    model = AutoModelForCausalLM.from_pretrained(spec.hf_path, **_base_model_kwargs(spec))
    normalize_model_config_special_ids(model.config)
    model.config.use_cache = False
    return model


def _load_causal_lm_from_peft_checkpoint(spec: ModelSpec, trainable: bool) -> Any:
    peft_checkpoint_dir = _peft_checkpoint_dir(spec.hf_path)
    if peft_checkpoint_dir is None:
        raise ValueError(f"Expected a PEFT checkpoint directory at '{spec.hf_path}'.")
    if PeftConfig is None or PeftModel is None:
        raise ImportError("peft is required to load policy adapter checkpoints.")

    peft_config = PeftConfig.from_pretrained(str(peft_checkpoint_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(peft_config.base_model_name_or_path),
        **_base_model_kwargs(spec),
    )
    normalize_model_config_special_ids(model.config)
    model.config.use_cache = False
    if trainable:
        enable_gradient_checkpointing(model)
        model = prepare_model_for_peft_training(model, is_quantized=spec.quantization is not None)
    model = PeftModel.from_pretrained(model, str(peft_checkpoint_dir), is_trainable=trainable)
    normalize_model_config_special_ids(model.config)
    if trainable:
        enable_input_require_grads(model)
        return model
    return _freeze_module(model)


def load_policy_model(spec: ModelSpec) -> Any:
    """Load a policy causal LM, optionally with LoRA."""
    if _peft_checkpoint_dir(spec.hf_path) is not None:
        return _load_causal_lm_from_peft_checkpoint(spec, trainable=True)

    model = _load_causal_lm_base(spec)
    enable_gradient_checkpointing(model)
    model = maybe_apply_lora(model, spec, is_quantized=spec.quantization is not None)
    enable_input_require_grads(model)
    return model


def load_reward_model(spec: ModelSpec, tokenizer: Any | None = None) -> Any:
    """Load a scalar reward model using sequence classification."""
    peft_checkpoint_dir = _peft_checkpoint_dir(spec.hf_path)
    if peft_checkpoint_dir is not None:
        if PeftConfig is None or PeftModel is None:
            raise ImportError("peft is required to load reward-model adapter checkpoints.")
        peft_config = PeftConfig.from_pretrained(str(peft_checkpoint_dir))
        with _suppress_peft_reward_head_warning():
            model = AutoModelForSequenceClassification.from_pretrained(
                str(peft_config.base_model_name_or_path),
                num_labels=1,
                **_base_model_kwargs(spec),
            )
        normalize_model_config_special_ids(model.config, tokenizer=tokenizer)
        model = PeftModel.from_pretrained(model, str(peft_checkpoint_dir), is_trainable=False)
        normalize_model_config_special_ids(model.config, tokenizer=tokenizer)
        return _freeze_module(model)

    model = AutoModelForSequenceClassification.from_pretrained(
        spec.hf_path,
        num_labels=1,
        **_base_model_kwargs(spec),
    )
    normalize_model_config_special_ids(model.config, tokenizer=tokenizer)
    enable_gradient_checkpointing(model)
    model = maybe_apply_lora(model, spec, is_quantized=spec.quantization is not None)
    enable_input_require_grads(model)
    return model


def load_reference_model(spec: ModelSpec) -> Any:
    """Load a frozen reference policy."""
    if _peft_checkpoint_dir(spec.hf_path) is not None:
        return _load_causal_lm_from_peft_checkpoint(spec, trainable=False)
    return _freeze_module(_load_causal_lm_base(spec))


def load_value_model(spec: ModelSpec) -> CausalValueModel:
    """Load a value model as a decoder-only backbone plus value head."""
    config = AutoConfig.from_pretrained(spec.hf_path, trust_remote_code=spec.trust_remote_code)
    normalize_model_config_special_ids(config)
    backbone = AutoModelForCausalLM.from_pretrained(spec.hf_path, **_base_model_kwargs(spec))
    normalize_model_config_special_ids(backbone.config)
    enable_gradient_checkpointing(backbone)
    backbone = maybe_apply_lora(backbone, spec, is_quantized=spec.quantization is not None)
    enable_input_require_grads(backbone)
    return CausalValueModel(backbone=backbone, hidden_size=_hidden_size_from_config(config))
