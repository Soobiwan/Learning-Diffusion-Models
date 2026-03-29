"""KL helpers for reference shaping."""

from __future__ import annotations

import torch

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
