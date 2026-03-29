"""Adapter for Anthropic HH-RLHF preference data."""

from __future__ import annotations

from typing import Any

from ..base import AdapterRegistry, DatasetAdapter
from ..schemas import PreferenceExample

_ASSISTANT_MARKERS = ("\n\nAssistant:", "\n\nassistant:", "Assistant:", "assistant:")


def _split_prompt_response(transcript: str) -> tuple[str, str]:
    marker_positions = [(transcript.rfind(marker), marker) for marker in _ASSISTANT_MARKERS]
    idx, marker = max(marker_positions, key=lambda item: item[0])
    if idx < 0:
        raise ValueError("HH-RLHF transcript does not contain an Assistant marker.")
    prompt = transcript[: idx + len(marker)].strip()
    response = transcript[idx + len(marker) :].strip()
    return prompt, response


def _shared_prompt(chosen_prompt: str, rejected_prompt: str) -> str:
    if chosen_prompt == rejected_prompt:
        return chosen_prompt

    common_length = 0
    for left_char, right_char in zip(chosen_prompt, rejected_prompt):
        if left_char != right_char:
            break
        common_length += 1

    prefix = chosen_prompt[:common_length]
    valid_cutoffs = [prefix.rfind(marker) + len(marker) for marker in _ASSISTANT_MARKERS]
    cutoff = max(valid_cutoffs)
    if cutoff <= 0:
        raise ValueError("Unable to recover shared HH-RLHF prompt.")
    return prefix[:cutoff].strip()


@AdapterRegistry.register
class HHRLHFAdapter(DatasetAdapter):
    """Convert HH-RLHF rows into `PreferenceExample` objects."""

    name = "hh_rlhf"
    dataset_path = "Anthropic/hh-rlhf"

    def raw_to_canonical(self, raw_example: dict[str, Any]) -> PreferenceExample:
        chosen_prompt, chosen_response = _split_prompt_response(str(raw_example["chosen"]))
        rejected_prompt, rejected_response = _split_prompt_response(str(raw_example["rejected"]))
        prompt = _shared_prompt(chosen_prompt, rejected_prompt)
        return PreferenceExample(
            prompt=prompt,
            chosen=chosen_response,
            rejected=rejected_response,
            meta={
                "source": self.name,
                "split_hint": raw_example.get("split"),
            },
        )
