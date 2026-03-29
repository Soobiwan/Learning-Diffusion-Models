"""SimPO placeholder objective."""

from __future__ import annotations

from .base import LossOutput, Objective


class SimPOObjective(Objective):
    """Placeholder for future SimPO integration."""

    name = "simpo"

    def compute(self, *args, **kwargs) -> LossOutput:  # type: ignore[override]
        raise NotImplementedError(
            "TODO: implement SimPO loss. The pairwise trainer interface is already in place."
        )
