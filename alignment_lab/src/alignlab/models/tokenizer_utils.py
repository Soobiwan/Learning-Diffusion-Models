"""Tokenizer helpers and model-family quirks."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

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


def load_tokenizer(spec: ModelSpec) -> PreTrainedTokenizerBase:
    """Load and configure the tokenizer for a model spec."""
    tokenizer = AutoTokenizer.from_pretrained(
        spec.tokenizer_path or spec.hf_path,
        trust_remote_code=spec.trust_remote_code,
        padding_side=spec.padding_side,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
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
