"""Tokenizer helpers and model-family quirks."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .specs import ModelSpec


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a config dtype string into a torch dtype."""
    normalized = dtype_name.lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise KeyError(f"Unsupported dtype '{dtype_name}'.")
    return mapping[normalized]


def normalize_special_token_id(token_id: Any) -> int | None:
    """Convert tokenizer/config special token ids into a scalar int when possible."""
    if token_id is None:
        return None
    if isinstance(token_id, (list, tuple)):
        if not token_id:
            return None
        token_id = token_id[0]
    return int(token_id)


def normalize_model_config_special_ids(config: Any, tokenizer: PreTrainedTokenizerBase | None = None) -> Any:
    """Ensure model config pad/eos token ids are scalar integers."""
    eos_token_id = normalize_special_token_id(getattr(config, "eos_token_id", None))
    pad_token_id = normalize_special_token_id(getattr(config, "pad_token_id", None))

    tokenizer_eos_token_id = None
    tokenizer_pad_token_id = None
    if tokenizer is not None:
        tokenizer_eos_token_id = normalize_special_token_id(getattr(tokenizer, "eos_token_id", None))
        tokenizer_pad_token_id = normalize_special_token_id(getattr(tokenizer, "pad_token_id", None))

    if eos_token_id is None:
        eos_token_id = tokenizer_eos_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer_pad_token_id if tokenizer_pad_token_id is not None else eos_token_id

    if eos_token_id is not None:
        config.eos_token_id = eos_token_id
    if pad_token_id is not None:
        config.pad_token_id = pad_token_id
    return config


def normalize_tokenizer_special_ids(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Ensure tokenizer pad/eos token ids are scalar integers and internally consistent."""
    eos_token_id = normalize_special_token_id(getattr(tokenizer, "eos_token_id", None))
    pad_token_id = normalize_special_token_id(getattr(tokenizer, "pad_token_id", None))

    if eos_token_id is not None:
        tokenizer.eos_token_id = eos_token_id
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = eos_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is not None:
        tokenizer.pad_token_id = pad_token_id
    return tokenizer


def load_tokenizer(spec: ModelSpec) -> PreTrainedTokenizerBase:
    """Load and configure the tokenizer for a model spec."""
    tokenizer = AutoTokenizer.from_pretrained(
        spec.tokenizer_path or spec.hf_path,
        trust_remote_code=spec.trust_remote_code,
        padding_side=spec.padding_side,
    )
    tokenizer = normalize_tokenizer_special_ids(tokenizer)
    tokenizer.padding_side = spec.padding_side
    return tokenizer


@contextmanager
def temporary_padding_side(
    tokenizer: PreTrainedTokenizerBase,
    padding_side: str,
) -> Iterator[PreTrainedTokenizerBase]:
    """Temporarily switch tokenizer padding side."""
    original = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        yield tokenizer
    finally:
        tokenizer.padding_side = original
