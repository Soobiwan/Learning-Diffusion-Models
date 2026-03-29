"""Rollout generation and prompt expansion."""

from __future__ import annotations

from typing import Any

import torch

from ..models.generation import generate_batched


def repeat_prompt_batch(batch: dict[str, Any], repeats: int) -> dict[str, Any]:
    """Repeat prompt-only batches for grouped rollouts."""
    expanded = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            expanded[key] = value.repeat_interleave(repeats, dim=0)
        elif isinstance(value, list):
            expanded[key] = [item for item in value for _ in range(repeats)]
        else:
            expanded[key] = value
    return expanded


def generate_rollout_batch(
    model: Any,
    tokenizer: Any,
    prompt_batch: dict[str, Any],
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    """Generate continuations and construct labels/masks for RL updates."""
    generation = generate_batched(
        model=model,
        tokenizer=tokenizer,
        input_ids=prompt_batch["input_ids"],
        attention_mask=prompt_batch["attention_mask"],
        generation_config=generation_config,
    )
    sequences = generation["sequences"]
    pad_token_id = getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", 0))
    attention_mask = (sequences != pad_token_id).long()
    prompt_lengths = prompt_batch["attention_mask"].sum(dim=-1).tolist()

    labels = sequences.clone()
    response_mask = torch.zeros_like(sequences)
    for row_idx, prompt_length in enumerate(prompt_lengths):
        labels[row_idx, :prompt_length] = -100
        response_mask[row_idx, prompt_length:] = 1
    labels = labels.masked_fill(attention_mask == 0, -100)

    return {
        "input_ids": sequences,
        "attention_mask": attention_mask,
        "labels": labels,
        "response_mask": response_mask,
        "responses": generation["responses"],
        "prompt_lengths": prompt_lengths,
    }
