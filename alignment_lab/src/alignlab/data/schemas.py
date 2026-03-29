"""Canonical internal example schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PreferenceExample:
    """A pairwise preference example with a shared prompt."""

    prompt: str
    chosen: str
    rejected: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SFTExample:
    """A prompt-response example for supervised fine-tuning."""

    prompt: str
    response: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VerifiableExample:
    """A prompt with a verifiable target answer for RLVR-style training."""

    prompt: str
    gold_answer: str
    meta: dict[str, Any] = field(default_factory=dict)
