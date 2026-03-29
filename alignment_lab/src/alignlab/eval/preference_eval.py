"""Preference optimization evaluation helpers."""

from __future__ import annotations

import torch


def preference_accuracy(logits: torch.Tensor) -> float:
    """Compute binary preference accuracy from signed margins."""
    return float((logits > 0).float().mean().item())


def preference_accuracy_from_logprobs(
    chosen_logprobs: torch.Tensor,
    rejected_logprobs: torch.Tensor,
) -> float:
    """Compute preference accuracy directly from chosen/rejected sequence scores."""
    return float((chosen_logprobs > rejected_logprobs).float().mean().item())
