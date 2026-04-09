"""KL helpers for reference shaping."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..common.utils import masked_mean


def per_token_kl(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Approximate sampled-token KL with log-prob differences."""
    kl = policy_logprobs - reference_logprobs
    return kl * mask.to(kl.dtype)


def mean_kl(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute mean masked sampled-token KL."""
    return masked_mean(per_token_kl(policy_logprobs, reference_logprobs, mask), mask)


def full_vocab_token_kl_from_logits(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact per-token KL over the full vocabulary under a causal mask."""
    shifted_policy_logits = policy_logits[:, :-1, :]
    shifted_reference_logits = reference_logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(ignore_index)
    policy_logprobs = F.log_softmax(shifted_policy_logits, dim=-1)
    reference_logprobs = F.log_softmax(shifted_reference_logits, dim=-1)
    token_kl = (policy_logprobs.exp() * (policy_logprobs - reference_logprobs)).sum(dim=-1)
    return token_kl * mask.to(token_kl.dtype), mask


def mean_full_vocab_kl_from_logits(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute mean masked exact KL over the full vocabulary."""
    token_kl, mask = full_vocab_token_kl_from_logits(
        policy_logits=policy_logits,
        reference_logits=reference_logits,
        labels=labels,
        ignore_index=ignore_index,
    )
    return masked_mean(token_kl, mask)
