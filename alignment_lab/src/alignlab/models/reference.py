"""Reference model helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from .factory import load_reference_model
from .specs import ModelSpec
from .tokenizer_utils import load_tokenizer


class AdapterDisabledReference(nn.Module):
    """Use a LoRA-equipped policy as a frozen reference by disabling adapters."""

    def __init__(self, policy_model: nn.Module) -> None:
        super().__init__()
        self.policy_model = policy_model

    @contextmanager
    def _adapter_context(self) -> Iterator[None]:
        disable_adapter = getattr(self.policy_model, "disable_adapter", None)
        if disable_adapter is None:
            yield
            return
        with disable_adapter():
            yield

    def forward(self, *args, **kwargs):  # type: ignore[override]
        with self._adapter_context():
            return self.policy_model(*args, **kwargs)


@dataclass(slots=True)
class ReferenceBundle:
    """Bundled frozen reference policy."""

    model: nn.Module
    tokenizer: PreTrainedTokenizerBase
    spec: ModelSpec


def build_reference_bundle(
    spec: ModelSpec | None = None,
    policy_model: nn.Module | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> ReferenceBundle:
    """Build a reference model either from a separate spec or from the policy model."""
    if policy_model is not None:
        if tokenizer is None or spec is None:
            raise ValueError("Passing `policy_model` requires both `tokenizer` and `spec`.")
        return ReferenceBundle(
            model=AdapterDisabledReference(policy_model),
            tokenizer=tokenizer,
            spec=spec,
        )
    if spec is None:
        raise ValueError("A model spec is required when loading a standalone reference model.")
    return ReferenceBundle(model=load_reference_model(spec), tokenizer=load_tokenizer(spec), spec=spec)
