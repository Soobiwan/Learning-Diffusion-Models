"""Sample generation reporting."""

from __future__ import annotations

from typing import Sequence


def build_generation_table(
    prompts: Sequence[str],
    responses: Sequence[str],
    rewards: Sequence[float] | None = None,
) -> list[dict[str, str | float]]:
    """Build a compact table of generations for reports."""
    rows: list[dict[str, str | float]] = []
    for idx, (prompt, response) in enumerate(zip(prompts, responses)):
        row: dict[str, str | float] = {"prompt": prompt, "response": response}
        if rewards is not None:
            row["reward"] = rewards[idx]
        rows.append(row)
    return rows
