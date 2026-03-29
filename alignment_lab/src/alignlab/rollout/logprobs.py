"""Sequence and token log-probability utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather token log-probabilities under shifted causal LM logits."""
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:].clone()
    mask = shifted_labels.ne(ignore_index)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    token_logprobs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask.to(token_logprobs.dtype)
    return token_logprobs, mask


def sequence_logprobs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce token log-probabilities to per-sequence scores."""
    token_logprobs, mask = gather_token_logprobs(logits=logits, labels=labels)
    sequence_logprobs = token_logprobs.sum(dim=-1)
    if average:
        counts = mask.sum(dim=-1).clamp_min(1)
        sequence_logprobs = sequence_logprobs / counts
    return sequence_logprobs, mask
