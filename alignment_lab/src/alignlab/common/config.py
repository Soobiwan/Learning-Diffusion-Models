"""Configuration helpers for experiment-driven runs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at {path}, found {type(data)!r}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings and return a new dictionary."""
    merged: dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_named_config(config_root: Path, section: str, value: str) -> Path:
    candidate = Path(value)
    if candidate.suffix == ".yaml":
        if candidate.is_absolute():
            return candidate
        return (config_root / candidate).resolve()
    return (config_root / section / f"{value}.yaml").resolve()


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load defaults plus model, data, and method sections for an experiment."""
    experiment_path = Path(path).resolve()
    config_root = experiment_path.parents[1]
    config = load_yaml(config_root / "defaults.yaml")
    experiment_cfg = load_yaml(experiment_path)

    for section in ("model", "data", "method", "reference_model", "reward_model"):
        value = experiment_cfg.get(section)
        if value is None:
            continue
        if section in {"reference_model", "reward_model"} and isinstance(value, str):
            resolved = _resolve_named_config(config_root, "model", value)
            config[section] = load_yaml(resolved)
            continue
        if isinstance(value, str):
            resolved = _resolve_named_config(config_root, section, value)
            config[section] = load_yaml(resolved)
        elif isinstance(value, Mapping):
            config[section] = deep_merge(config.get(section, {}), value)
        else:
            raise TypeError(f"Unsupported config value for section '{section}': {type(value)!r}")

    overrides = experiment_cfg.get("overrides", {})
    if overrides:
        config = deep_merge(config, overrides)

    config["experiment_name"] = experiment_cfg.get("name", experiment_path.stem)
    config["config_path"] = str(experiment_path)
    return config
