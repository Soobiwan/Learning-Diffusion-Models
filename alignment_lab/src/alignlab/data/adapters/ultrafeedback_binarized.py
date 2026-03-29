"""Adapter skeleton for UltraFeedback binarized preferences."""

from __future__ import annotations

from typing import Any

from ..base import AdapterRegistry, DatasetAdapter
from ..schemas import PreferenceExample


@AdapterRegistry.register
class UltraFeedbackBinarizedAdapter(DatasetAdapter):
    """Map common UltraFeedback-style fields into canonical preferences.

    TODO: verify the exact raw field names against the chosen dataset release
    before the first full training run.
    """

    name = "ultrafeedback_binarized"
    dataset_path = "HuggingFaceH4/ultrafeedback_binarized"

    def raw_to_canonical(self, raw_example: dict[str, Any]) -> PreferenceExample:
        prompt = raw_example.get("prompt") or raw_example.get("instruction") or raw_example.get("question")
        chosen = raw_example.get("chosen") or raw_example.get("response_chosen")
        rejected = raw_example.get("rejected") or raw_example.get("response_rejected")
        if not all((prompt, chosen, rejected)):
            raise KeyError(
                "UltraFeedback adapter field mapping is incomplete. "
                "TODO: inspect the downloaded dataset schema and update the adapter."
            )
        return PreferenceExample(
            prompt=str(prompt).strip(),
            chosen=str(chosen).strip(),
            rejected=str(rejected).strip(),
            meta={"source": self.name},
        )
