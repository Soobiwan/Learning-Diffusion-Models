"""Report writing and lightweight resource tracking helpers."""

from __future__ import annotations

import csv
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def experiment_log_path(config: dict[str, Any]) -> Path:
    log_dir = Path(config.get("log_dir", "artifacts/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{config['experiment_name']}.jsonl"


def experiment_table_path(config: dict[str, Any], stem: str, suffix: str = ".json") -> Path:
    return _ensure_parent(Path("artifacts/tables") / f"{config['experiment_name']}_{stem}{suffix}")


def experiment_sample_path(config: dict[str, Any], stem: str, suffix: str = ".json") -> Path:
    return _ensure_parent(Path("artifacts/samples") / f"{config['experiment_name']}_{stem}{suffix}")


def experiment_plot_path(config: dict[str, Any], stem: str, suffix: str = ".png") -> Path:
    return _ensure_parent(Path("artifacts/plots") / f"{config['experiment_name']}_{stem}{suffix}")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def write_json(path: Path, payload: Any) -> Path:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def write_csv_rows(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    _ensure_parent(path)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_generation_artifacts(
    config: dict[str, Any],
    stem: str,
    rows: Sequence[dict[str, Any]],
) -> dict[str, str]:
    json_path = write_json(experiment_sample_path(config, stem, ".json"), list(rows))
    csv_path = write_csv_rows(experiment_sample_path(config, stem, ".csv"), list(rows))
    return {"json": str(json_path), "csv": str(csv_path)}


def plot_histogram(path: Path, series: dict[str, Sequence[float]], bins: int = 20, title: str | None = None) -> Path:
    _ensure_parent(path)
    fig, axis = plt.subplots(figsize=(8, 5))
    for label, values in series.items():
        axis.hist(list(values), bins=bins, alpha=0.6, label=label)
    axis.set_xlabel("Score")
    axis.set_ylabel("Count")
    if title:
        axis.set_title(title)
    if len(series) > 1:
        axis.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_metric_curves(path: Path, rows: Sequence[dict[str, Any]], keys: Sequence[str], title: str | None = None) -> Path:
    _ensure_parent(path)
    if not rows:
        raise ValueError("Cannot plot metric curves without rows.")
    steps = [float(row.get("step", idx + 1)) for idx, row in enumerate(rows)]
    fig, axis = plt.subplots(figsize=(8, 5))
    for key in keys:
        values = [row.get(key) for row in rows]
        if any(value is not None for value in values):
            axis.plot(steps, values, label=key)
    axis.set_xlabel("Step")
    axis.set_ylabel("Value")
    if title:
        axis.set_title(title)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


class ResourceTracker:
    """Track wall time and peak CUDA memory across a training or evaluation run."""

    def __init__(self) -> None:
        self._run_start = time.perf_counter()
        self._step_durations: list[float] = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def measure_step(self) -> Iterator[None]:
        step_start = time.perf_counter()
        try:
            yield
        finally:
            self._step_durations.append(time.perf_counter() - step_start)

    def summary(self) -> dict[str, Any]:
        total_time = time.perf_counter() - self._run_start
        peak_memory_gb = None
        if torch.cuda.is_available():
            peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        mean_step = sum(self._step_durations) / len(self._step_durations) if self._step_durations else 0.0
        return {
            "peak_vram_gb": peak_memory_gb,
            "step_time_seconds": mean_step,
            "total_time_seconds": total_time,
            "num_recorded_steps": len(self._step_durations),
        }
