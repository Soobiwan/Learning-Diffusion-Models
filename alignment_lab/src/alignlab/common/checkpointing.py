"""Checkpoint and artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any


def experiment_checkpoint_dir(config: dict[str, Any]) -> Path:
    """Return the canonical checkpoint directory for an experiment."""
    output_dir = Path(config.get("output_dir", "artifacts/checkpoints"))
    return output_dir / str(config["experiment_name"])


def checkpoint_variant_dir(config: dict[str, Any], artifact_name: str = "final") -> Path:
    """Return a named checkpoint variant path for an experiment."""
    return experiment_checkpoint_dir(config) / artifact_name


def final_checkpoint_dir(config: dict[str, Any]) -> Path:
    """Return the final checkpoint path."""
    return checkpoint_variant_dir(config, "final")


def checkpoint_metadata(
    config: dict[str, Any],
    checkpoint_dir: Path,
    artifact_name: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build checkpoint metadata shared by saved and promoted artifacts."""
    metadata = {
        "experiment_name": config["experiment_name"],
        "checkpoint_dir": str(checkpoint_dir),
        "source_config": config.get("config_path"),
        "artifact_name": artifact_name,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def write_checkpoint_metadata(config: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Persist the experiment-level checkpoint manifest."""
    manifest_path = experiment_checkpoint_dir(config) / "checkpoint_info.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def save_pretrained_artifact(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    artifact_name: str = "final",
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a pretrained-style artifact plus a small metadata manifest."""
    save_dir = checkpoint_variant_dir(config, artifact_name)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(save_dir)
    else:
        raise TypeError("Model does not implement save_pretrained().")
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_dir)
    metadata = checkpoint_metadata(config, save_dir, artifact_name, extra_metadata=extra_metadata)
    write_checkpoint_metadata(config, metadata)
    return save_dir


def promote_checkpoint_variant(
    config: dict[str, Any],
    source_name: str,
    target_name: str = "final",
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Copy a named checkpoint variant into another variant slot."""
    source_dir = checkpoint_variant_dir(config, source_name)
    if not source_dir.exists():
        raise FileNotFoundError(f"Checkpoint variant '{source_name}' was not found at '{source_dir}'.")
    target_dir = checkpoint_variant_dir(config, target_name)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    metadata = checkpoint_metadata(
        config,
        target_dir,
        target_name,
        extra_metadata={"promoted_from": str(source_dir), **(extra_metadata or {})},
    )
    write_checkpoint_metadata(config, metadata)
    return target_dir
