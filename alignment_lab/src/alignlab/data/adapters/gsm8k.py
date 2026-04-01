"""Adapter and answer extraction utilities for GSM8K."""

from __future__ import annotations

import re
from typing import Any

from ..base import AdapterRegistry, DatasetAdapter
from ..schemas import VerifiableExample
from ...prompts.formatting import format_gsm8k_prompt

_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_ANSWER_PHRASE_PATTERN = re.compile(
    r"(?:the answer is|answer:|final answer:)\s*([-+]?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_HASH_PATTERN = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def normalize_numeric_answer(text: str) -> str:
    """Normalize numeric answers by removing commas and trimming whitespace."""
    return text.replace(",", "").strip()


def extract_numeric_answer(text: str) -> str | None:
    """Extract the most likely numeric answer from free-form GSM8K text."""
    if not text:
        return None
    for pattern in (_HASH_PATTERN, _ANSWER_PHRASE_PATTERN):
        match = pattern.search(text)
        if match:
            return normalize_numeric_answer(match.group(1))

    matches = _NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return normalize_numeric_answer(matches[-1])


@AdapterRegistry.register
class GSM8KAdapter(DatasetAdapter):
    """Convert GSM8K rows into `VerifiableExample` objects."""

    name = "gsm8k"
    dataset_path = "gsm8k"

    def raw_to_canonical(self, raw_example: dict[str, Any]) -> VerifiableExample:
        answer_text = str(raw_example.get("answer", "")).strip()
        gold_answer = extract_numeric_answer(answer_text)
        if gold_answer is None:
            raise ValueError("Could not extract a numeric GSM8K answer from the raw row.")
        return VerifiableExample(
            prompt=format_gsm8k_prompt(str(raw_example["question"]).strip()),
            gold_answer=gold_answer,
            meta={"source": self.name, "answer_text": answer_text},
        )
