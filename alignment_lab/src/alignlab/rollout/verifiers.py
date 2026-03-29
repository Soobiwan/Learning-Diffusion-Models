"""Verifier utilities for RLVR."""

from __future__ import annotations

from dataclasses import dataclass

from ..data.adapters.gsm8k import extract_numeric_answer


@dataclass(slots=True)
class GSM8KAnswerVerifier:
    """Verify GSM8K completions by comparing extracted numeric answers."""

    reward_correct: float = 1.0
    reward_incorrect: float = 0.0

    def verify(self, response: str, gold_answer: str) -> float:
        extracted = extract_numeric_answer(response)
        if extracted is None:
            return self.reward_incorrect
        return self.reward_correct if extracted == gold_answer else self.reward_incorrect
