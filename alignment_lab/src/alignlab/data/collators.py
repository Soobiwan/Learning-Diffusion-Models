"""Collators for SFT, RM, pairwise, and rollout batches."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import torch

from .schemas import PreferenceExample, SFTExample, VerifiableExample


def _get_special_id(tokenizer: Any, preferred: str, fallback: str) -> int:
    token_id = getattr(tokenizer, preferred, None)
    if token_id is None:
        token_id = getattr(tokenizer, fallback, None)
    if token_id is None:
        raise ValueError(f"Tokenizer is missing both '{preferred}' and '{fallback}'.")
    return int(token_id)


def _truncate_prompt_response(
    prompt_ids: list[int],
    response_ids: list[int],
    max_length: int,
) -> tuple[list[int], list[int]]:
    if len(prompt_ids) + len(response_ids) <= max_length:
        return prompt_ids, response_ids

    overflow = len(prompt_ids) + len(response_ids) - max_length
    if overflow > 0 and prompt_ids:
        prompt_ids = prompt_ids[min(overflow, len(prompt_ids)) :]
    if len(prompt_ids) + len(response_ids) > max_length:
        response_ids = response_ids[: max_length - len(prompt_ids)]
    return prompt_ids, response_ids


def build_prompt_response_features(
    tokenizer: Any,
    prompt: str,
    response: str,
    max_length: int,
    add_eos: bool = True,
) -> dict[str, list[int]]:
    """Tokenize prompt and response separately so prompt tokens can be masked."""
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    response_ids = list(tokenizer.encode(response, add_special_tokens=False))

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if add_eos and eos_token_id is not None and (not response_ids or response_ids[-1] != eos_token_id):
        response_ids.append(int(eos_token_id))

    prompt_ids, response_ids = _truncate_prompt_response(prompt_ids, response_ids, max_length)
    input_ids = prompt_ids + response_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + response_ids
    response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "response_mask": response_mask,
        "prompt_length": len(prompt_ids),
        "response_length": len(response_ids),
    }


def _pad_sequences(sequences: Iterable[list[int]], pad_value: int, left_pad: bool = False) -> torch.Tensor:
    rows = list(sequences)
    max_len = max(len(row) for row in rows)
    padded: list[list[int]] = []
    for row in rows:
        pad = [pad_value] * (max_len - len(row))
        padded.append(pad + row if left_pad else row + pad)
    return torch.tensor(padded, dtype=torch.long)


def _use_left_padding(tokenizer: Any) -> bool:
    return str(getattr(tokenizer, "padding_side", "right")).lower() == "left"


class SFTCollator:
    """Collate `SFTExample` rows with response-only labels."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[SFTExample]) -> dict[str, torch.Tensor]:
        features = [
            build_prompt_response_features(
                tokenizer=self.tokenizer,
                prompt=example.prompt,
                response=example.response,
                max_length=self.max_length,
            )
            for example in examples
        ]
        pad_token_id = _get_special_id(self.tokenizer, "pad_token_id", "eos_token_id")
        left_pad = _use_left_padding(self.tokenizer)
        return {
            "input_ids": _pad_sequences((item["input_ids"] for item in features), pad_token_id, left_pad=left_pad),
            "attention_mask": _pad_sequences((item["attention_mask"] for item in features), 0, left_pad=left_pad),
            "labels": _pad_sequences((item["labels"] for item in features), -100, left_pad=left_pad),
            "response_mask": _pad_sequences((item["response_mask"] for item in features), 0, left_pad=left_pad),
        }


class RewardModelCollator:
    """Collate `PreferenceExample` rows for reward model ranking."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[PreferenceExample]) -> dict[str, torch.Tensor]:
        chosen_features = [
            build_prompt_response_features(self.tokenizer, item.prompt, item.chosen, self.max_length)
            for item in examples
        ]
        rejected_features = [
            build_prompt_response_features(self.tokenizer, item.prompt, item.rejected, self.max_length)
            for item in examples
        ]
        pad_token_id = _get_special_id(self.tokenizer, "pad_token_id", "eos_token_id")
        left_pad = _use_left_padding(self.tokenizer)
        return {
            "chosen_input_ids": _pad_sequences(
                (item["input_ids"] for item in chosen_features), pad_token_id, left_pad=left_pad
            ),
            "chosen_attention_mask": _pad_sequences(
                (item["attention_mask"] for item in chosen_features), 0, left_pad=left_pad
            ),
            "rejected_input_ids": _pad_sequences(
                (item["input_ids"] for item in rejected_features), pad_token_id, left_pad=left_pad
            ),
            "rejected_attention_mask": _pad_sequences(
                (item["attention_mask"] for item in rejected_features), 0, left_pad=left_pad
            ),
        }


class PreferenceCollator:
    """Collate `PreferenceExample` rows with prompt-masked labels for pairwise objectives."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[PreferenceExample]) -> dict[str, torch.Tensor]:
        chosen_features = [
            build_prompt_response_features(self.tokenizer, item.prompt, item.chosen, self.max_length)
            for item in examples
        ]
        rejected_features = [
            build_prompt_response_features(self.tokenizer, item.prompt, item.rejected, self.max_length)
            for item in examples
        ]
        pad_token_id = _get_special_id(self.tokenizer, "pad_token_id", "eos_token_id")
        left_pad = _use_left_padding(self.tokenizer)
        batch = {
            "chosen_input_ids": _pad_sequences(
                (item["input_ids"] for item in chosen_features), pad_token_id, left_pad=left_pad
            ),
            "chosen_attention_mask": _pad_sequences(
                (item["attention_mask"] for item in chosen_features), 0, left_pad=left_pad
            ),
            "chosen_labels": _pad_sequences((item["labels"] for item in chosen_features), -100, left_pad=left_pad),
            "chosen_response_mask": _pad_sequences(
                (item["response_mask"] for item in chosen_features), 0, left_pad=left_pad
            ),
            "rejected_input_ids": _pad_sequences(
                (item["input_ids"] for item in rejected_features), pad_token_id, left_pad=left_pad
            ),
            "rejected_attention_mask": _pad_sequences(
                (item["attention_mask"] for item in rejected_features), 0, left_pad=left_pad
            ),
            "rejected_labels": _pad_sequences(
                (item["labels"] for item in rejected_features), -100, left_pad=left_pad
            ),
            "rejected_response_mask": _pad_sequences(
                (item["response_mask"] for item in rejected_features), 0, left_pad=left_pad
            ),
        }
        return batch


class PromptOnlyCollator:
    """Collate prompts for online rollout generation or evaluation."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[PreferenceExample | SFTExample | VerifiableExample]) -> dict[str, Any]:
        prompts = [example.prompt for example in examples]
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch: dict[str, Any] = dict(encoded)
        batch["raw_examples"] = [asdict(example) for example in examples]
        return batch
