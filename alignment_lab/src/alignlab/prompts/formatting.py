"""Centralize prompt formatting quirks by model family."""

from __future__ import annotations


def format_prompt(family: str, prompt: str) -> str:
    """Apply family-specific prompt normalization."""
    normalized = prompt.strip()
    if family.lower() == "smollm":
        return normalized
    return normalized


def format_prompt_response(family: str, prompt: str, response: str) -> str:
    """Join a prompt and response for plain decoder-only training."""
    prompt_text = format_prompt(family, prompt)
    response_text = response.strip()
    if prompt_text.endswith((" ", "\n")):
        return f"{prompt_text}{response_text}"
    return f"{prompt_text}\n{response_text}"
