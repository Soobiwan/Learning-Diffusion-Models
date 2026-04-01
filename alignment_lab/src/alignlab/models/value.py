"""Value model wrapper for PPO-style training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerBase

from .peft_utils import enable_gradient_checkpointing, enable_input_require_grads, maybe_apply_lora
from .specs import ModelSpec
from .tokenizer_utils import load_tokenizer, normalize_model_config_special_ids, resolve_torch_dtype


def _hidden_size_from_config(config: Any) -> int:
    for attribute in ("hidden_size", "n_embd", "d_model"):
        if hasattr(config, attribute):
            return int(getattr(config, attribute))
    raise AttributeError("Unable to infer hidden size from model config.")


def _quantization_config(mode: str | None) -> BitsAndBytesConfig | None:
    if mode is None:
        return None
    normalized = mode.lower()
    if normalized == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=resolve_torch_dtype("fp16"))
    if normalized == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization mode '{mode}'.")


class CausalValueModel(nn.Module):
    """A lightweight value head on top of a decoder-only backbone."""

    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.value_head.weight, std=0.01)
        nn.init.zeros_(self.value_head.bias)

    def move_auxiliary_modules_to_backbone_device(self) -> "CausalValueModel":
        """Align the trainable value head with a quantized backbone device."""
        device = next(self.backbone.parameters()).device
        self.value_head.to(device)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]
        return self.value_head(hidden_states).squeeze(-1)


@dataclass(slots=True)
class ValueModelBundle:
    """Convenience bundle for value training."""

    model: CausalValueModel
    tokenizer: PreTrainedTokenizerBase
    spec: ModelSpec


def load_value_bundle(spec: ModelSpec) -> ValueModelBundle:
    """Load a causal backbone with a scalar token-value head."""
    config = AutoConfig.from_pretrained(spec.hf_path, trust_remote_code=spec.trust_remote_code)
    normalize_model_config_special_ids(config)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": spec.trust_remote_code,
        "torch_dtype": resolve_torch_dtype(spec.dtype),
    }
    quantization_config = _quantization_config(spec.quantization)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    backbone = AutoModelForCausalLM.from_pretrained(
        spec.hf_path,
        **model_kwargs,
    )
    normalize_model_config_special_ids(backbone.config)
    enable_gradient_checkpointing(backbone)
    backbone = maybe_apply_lora(backbone, spec, is_quantized=spec.quantization is not None)
    enable_input_require_grads(backbone)
    bundle_model = CausalValueModel(backbone=backbone, hidden_size=_hidden_size_from_config(config))
    return ValueModelBundle(model=bundle_model, tokenizer=load_tokenizer(spec), spec=spec)
