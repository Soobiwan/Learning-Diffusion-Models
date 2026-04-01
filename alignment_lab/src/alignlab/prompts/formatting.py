"""Centralize prompt formatting quirks by model family."""

from __future__ import annotations


def format_gsm8k_prompt(question: str) -> str:
    """Format a GSM8K problem as the PA2 RLVR prompt template."""
    return (
        "Solve the following math problem step by step.\n"
        "At the end, write your final answer as a single number.\n\n"
        f"Problem: {question.strip()}\n"
        "Solution:"
    )


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
