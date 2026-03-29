"""Win-rate helpers."""

from __future__ import annotations

from typing import Sequence

import torch
from .metrics import accuracy


def win_rate(wins: Sequence[bool]) -> float:
    """Compute a win rate over pairwise judgements."""
    return accuracy(wins)


def win_rate_from_scores(candidate_scores: Sequence[float], baseline_scores: Sequence[float]) -> float:
    """Compute win rate directly from scalar candidate and baseline scores."""
    candidate = torch.as_tensor(candidate_scores, dtype=torch.float32)
    baseline = torch.as_tensor(baseline_scores, dtype=torch.float32)
    return float((candidate > baseline).float().mean().item())
