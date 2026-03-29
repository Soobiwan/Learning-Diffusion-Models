"""RLVR reuses the GRPO-style objective with verifiable rewards."""

from __future__ import annotations

from .grpo import GRPOObjective


class RLVRObjective(GRPOObjective):
    """Alias GRPO objective semantics for verifiable rewards."""

    name = "rlvr"
