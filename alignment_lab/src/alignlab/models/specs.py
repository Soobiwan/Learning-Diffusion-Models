"""Structured model specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelSpec:
    """Configuration for loading a model and its tokenizer."""

    hf_path: str
    family: str
    tokenizer_path: str | None = None
    lora_target_modules: list[str] = field(default_factory=list)
    padding_side: str = "right"
    dtype: str = "fp16"
    trust_remote_code: bool = False
    task_type: str = "CAUSAL_LM"
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    quantization: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelSpec":
        """Create a spec from a config dictionary."""
        return cls(**payload)
