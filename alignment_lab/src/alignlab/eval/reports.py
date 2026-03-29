"""Lightweight report stubs."""

from __future__ import annotations

from typing import Any


def resource_timing_stub() -> dict[str, Any]:
    """Placeholder for resource and timing tables."""
    return {
        "gpu_memory_gb": None,
        "tokens_per_second": None,
        "step_time_seconds": None,
        "notes": "TODO: wire profiler/resource logging during longer benchmark runs.",
    }
