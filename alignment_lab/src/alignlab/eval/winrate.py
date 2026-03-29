"""Win-rate helpers."""

from __future__ import annotations

from typing import Sequence

from .metrics import accuracy


def win_rate(wins: Sequence[bool]) -> float:
    """Compute a win rate over pairwise judgements."""
    return accuracy(wins)
