"""KL evaluation utilities."""

from __future__ import annotations

import torch

from ..rollout.kl import mean_full_vocab_kl_from_logits, mean_kl


def estimate_policy_reference_kl(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Estimate average sampled-token KL."""
    return float(mean_kl(policy_logprobs, reference_logprobs, mask).item())


def estimate_policy_reference_full_vocab_kl(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Estimate exact average token KL from policy and reference logits."""
    return float(mean_full_vocab_kl_from_logits(policy_logits, reference_logits, labels).item())
