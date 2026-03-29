"""KL evaluation utilities."""

from __future__ import annotations

import torch

from ..rollout.kl import mean_kl


def estimate_policy_reference_kl(
    policy_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Estimate average sampled-token KL."""
    return float(mean_kl(policy_logprobs, reference_logprobs, mask).item())
