"""Reward model bundle helpers."""

from __future__ import annotations

from dataclasses import dataclass

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


def load_reward_bundle(spec: ModelSpec) -> RewardModelBundle:
    """Load reward model plus tokenizer."""
    return RewardModelBundle(
        model=load_reward_model(spec),
        tokenizer=load_tokenizer(spec),
        spec=spec,
    )
