"""Reward model bundle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from transformers import PreTrainedTokenizerBase

from .factory import load_reward_model
from .specs import ModelSpec
from .tokenizer_utils import load_tokenizer


@dataclass(slots=True)
class RewardModelBundle:
    """Bundled reward model components."""

    model: object
    tokenizer: PreTrainedTokenizerBase
    spec: ModelSpec


def freeze_module(module: Any) -> Any:
    """Freeze a module for inference use."""
    if hasattr(module, "eval"):
        module.eval()
    if hasattr(module, "parameters"):
        for parameter in module.parameters():
            parameter.requires_grad = False
    return module


def last_non_pad_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the last non-pad token index for each sequence."""
    return attention_mask.long().sum(dim=-1).clamp_min(1) - 1


def load_reward_bundle(spec: ModelSpec, freeze: bool = False) -> RewardModelBundle:
    """Load reward model plus tokenizer."""
    model = load_reward_model(spec)
    if freeze:
        model = freeze_module(model)
    return RewardModelBundle(
        model=model,
        tokenizer=load_tokenizer(spec),
        spec=spec,
    )
