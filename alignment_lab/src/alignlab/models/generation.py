"""Shared text generation utilities."""

from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def generate_batched(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    """Run batched generation and decode responses."""
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(generation_config.get("max_new_tokens", 64)),
        do_sample=bool(generation_config.get("do_sample", True)),
        temperature=float(generation_config.get("temperature", 0.8)),
        top_p=float(generation_config.get("top_p", 0.95)),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    prompt_length = input_ids.shape[1]
    response_ids = generated[:, prompt_length:]
    response_ids_cpu = response_ids.detach().cpu()
    return {
        "sequences": generated,
        "response_ids": response_ids,
        "responses": tokenizer.batch_decode(response_ids_cpu, skip_special_tokens=True),
    }
