"""Evaluation helpers for GSM8K verifiable rewards."""

from __future__ import annotations

from typing import Sequence

from ..rollout.verifiers import GSM8KAnswerVerifier


def gsm8k_pass_at_1(
    responses: Sequence[str],
    gold_answers: Sequence[str],
    verifier: GSM8KAnswerVerifier | None = None,
) -> float:
    """Compute pass@1 for one sampled answer per prompt."""
    verifier = verifier or GSM8KAnswerVerifier()
    correct = [
        verifier.verify(response=response, gold_answer=gold_answer) > 0.0
        for response, gold_answer in zip(responses, gold_answers)
    ]
    if not correct:
        return 0.0
    return sum(correct) / len(correct)
