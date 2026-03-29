"""Adapter skeleton for Orca-style DPO pairs."""

from __future__ import annotations

from typing import Any

from ..base import AdapterRegistry, DatasetAdapter
from ..schemas import PreferenceExample


@AdapterRegistry.register
class OrcaDPOPairsAdapter(DatasetAdapter):
    """Map common Orca DPO pair fields into canonical preferences.

    TODO: verify the exact dataset id and raw field names before first full run.
    """

    name = "orca_dpo_pairs"
    dataset_path = "Intel/orca_dpo_pairs"

    def raw_to_canonical(self, raw_example: dict[str, Any]) -> PreferenceExample:
        prompt = raw_example.get("prompt") or raw_example.get("question") or raw_example.get("instruction")
        chosen = raw_example.get("chosen") or raw_example.get("response_a")
        rejected = raw_example.get("rejected") or raw_example.get("response_b")
        if not all((prompt, chosen, rejected)):
            raise KeyError(
                "Orca DPO pair adapter field mapping is incomplete. "
                "TODO: inspect the downloaded dataset schema and update the adapter."
            )
        return PreferenceExample(
            prompt=str(prompt).strip(),
            chosen=str(chosen).strip(),
            rejected=str(rejected).strip(),
            meta={"source": self.name},
        )
