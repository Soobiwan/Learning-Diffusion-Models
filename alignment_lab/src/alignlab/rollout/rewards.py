"""Reward function abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch


class RewardFunction(ABC):
    """Abstract interface for scalar sequence rewards."""

    @abstractmethod
    def score_batch(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        targets: Sequence[str] | None = None,
        meta: Sequence[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        """Return one scalar reward per prompt-response pair."""

    def format_compliance_batch(self, responses: Sequence[str]) -> torch.Tensor | None:
        """Optionally measure response format compliance."""
        return None


class LearnedRewardFunction(RewardFunction):
    """Reward function backed by a sequence classification model."""

    def __init__(self, model: Any, tokenizer: Any, max_length: int = 384) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    @torch.no_grad()
    def score_batch(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        targets: Sequence[str] | None = None,
        meta: Sequence[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        texts = [f"{prompt}\n{response}" for prompt, response in zip(prompts, responses)]
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = self.model(**encoded).logits.squeeze(-1)
        return logits.detach().cpu()


class VerifiableRewardFunction(RewardFunction):
    """Reward function backed by deterministic answer verification."""

    def __init__(self, verifier: Any) -> None:
        self.verifier = verifier

    def score_batch(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        targets: Sequence[str] | None = None,
        meta: Sequence[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        if targets is None:
            raise ValueError("Verifiable rewards require target answers.")
        rewards = [
            float(self.verifier.verify(response=response, gold_answer=target))
            for response, target in zip(responses, targets)
        ]
        return torch.tensor(rewards, dtype=torch.float32)

    def format_compliance_batch(self, responses: Sequence[str]) -> torch.Tensor | None:
        values = [1.0 if self.verifier.has_valid_answer(response) else 0.0 for response in responses]
        return torch.tensor(values, dtype=torch.float32)
