"""Policy model bundle helpers."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from .factory import load_policy_model
from .peft_utils import trainable_parameter_summary
from .specs import ModelSpec
from .tokenizer_utils import load_tokenizer


@dataclass(slots=True)
class PolicyBundle:
    """Bundled policy components."""

    model: object
    tokenizer: PreTrainedTokenizerBase
    spec: ModelSpec
    parameter_summary: dict[str, int]


def load_policy_bundle(spec: ModelSpec) -> PolicyBundle:
    """Load policy model plus tokenizer."""
    model = load_policy_model(spec)
    tokenizer = load_tokenizer(spec)
    return PolicyBundle(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        parameter_summary=trainable_parameter_summary(model),
    )
