"""PA2-specific reporting and verification helpers."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

from ..data.schemas import PreferenceExample, SFTExample, VerifiableExample
from ..rollout.verifiers import GSM8KAnswerVerifier

CanonicalPreviewExample = PreferenceExample | SFTExample | VerifiableExample


def preview_canonical_examples(
    examples: Sequence[CanonicalPreviewExample],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Convert a few canonical examples into plain dictionaries for audit output."""
    rows: list[dict[str, Any]] = []
    for example in list(examples)[:limit]:
        rows.append(asdict(example))
    return rows


def load_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    """Load a JSON artifact when it exists."""
    candidate = Path(path)
    if not candidate.exists():
        return None
    with candidate.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in '{candidate}', found {type(payload)!r}.")
    return payload


def summarize_resource_payload(
    experiment_name: str,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize a resource-summary payload into a stable row for comparison tables."""
    payload = payload or {}
    return {
        "experiment_name": experiment_name,
        "peak_vram_gb": payload.get("peak_vram_gb"),
        "step_time_seconds": payload.get("step_time_seconds"),
        "total_time_seconds": payload.get("total_time_seconds"),
        "num_recorded_steps": payload.get("num_recorded_steps"),
    }


def verify_gsm8k_answer_extractor(
    examples: Sequence[VerifiableExample],
    verifier: GSM8KAnswerVerifier | None = None,
    gold_limit: int = 20,
    wrong_limit: int = 20,
) -> dict[str, Any]:
    """Run the PA2 precheck harness for GSM8K answer extraction."""
    verifier = verifier or GSM8KAnswerVerifier()
    selected = list(examples)[:gold_limit]
    gold_checks: list[dict[str, Any]] = []
    wrong_checks: list[dict[str, Any]] = []

    for example in selected:
        answer_text = str(example.meta.get("answer_text", f"#### {example.gold_answer}"))
        gold_checks.append(
            {
                "prompt": example.prompt,
                "response": answer_text,
                "gold_answer": example.gold_answer,
                "correct": verifier.verify(response=answer_text, gold_answer=example.gold_answer) > 0.0,
                "has_valid_answer": verifier.has_valid_answer(answer_text),
            }
        )

    for index, example in enumerate(selected[:wrong_limit], start=1):
        wrong_response = f"The answer is {987654321 + index}."
        wrong_checks.append(
            {
                "prompt": example.prompt,
                "response": wrong_response,
                "gold_answer": example.gold_answer,
                "correct": verifier.verify(response=wrong_response, gold_answer=example.gold_answer) > 0.0,
                "has_valid_answer": verifier.has_valid_answer(wrong_response),
            }
        )

    gold_correct = sum(1 for row in gold_checks if row["correct"])
    wrong_correct = sum(1 for row in wrong_checks if row["correct"])
    return {
        "gold_checks": len(gold_checks),
        "gold_correct": gold_correct,
        "gold_accuracy": (gold_correct / len(gold_checks)) if gold_checks else 0.0,
        "wrong_checks": len(wrong_checks),
        "wrong_correct": wrong_correct,
        "wrong_accuracy": (wrong_correct / len(wrong_checks)) if wrong_checks else 0.0,
        "samples": {
            "gold": gold_checks[:5],
            "wrong": wrong_checks[:5],
        },
    }
