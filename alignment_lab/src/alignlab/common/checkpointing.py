"""Checkpoint and artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def experiment_checkpoint_dir(config: dict[str, Any]) -> Path:
    """Return the canonical checkpoint directory for an experiment."""
    output_dir = Path(config.get("output_dir", "artifacts/checkpoints"))
    return output_dir / str(config["experiment_name"])


def final_checkpoint_dir(config: dict[str, Any]) -> Path:
    """Return the final checkpoint path."""
    return experiment_checkpoint_dir(config) / "final"


def save_pretrained_artifact(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a pretrained-style artifact plus a small metadata manifest."""
    save_dir = final_checkpoint_dir(config)
    save_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_dir)
    else:
        raise TypeError("Model does not implement save_pretrained().")
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_dir)
    metadata = {
        "experiment_name": config["experiment_name"],
        "checkpoint_dir": str(save_dir),
        "source_config": config.get("config_path"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    with (experiment_checkpoint_dir(config) / "checkpoint_info.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return save_dir
