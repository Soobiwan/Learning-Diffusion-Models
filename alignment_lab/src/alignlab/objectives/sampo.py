"""SamPO placeholder objective."""

from __future__ import annotations

from .base import LossOutput, Objective


class SamPOObjective(Objective):
    """Placeholder for future SamPO integration."""

    name = "sampo"

    def compute(self, *args, **kwargs) -> LossOutput:  # type: ignore[override]
        raise NotImplementedError(
            "TODO: implement SamPO loss. Reuse the pairwise trainer with this objective."
        )
