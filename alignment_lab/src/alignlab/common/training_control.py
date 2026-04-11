"""Training-control helpers for save-best and early stopping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class EarlyStoppingDecision:
    """Result of an early-stopping update."""

    enabled: bool
    metric: str
    value: float | None
    improved: bool
    should_save: bool
    should_stop: bool
    bad_evals: int
    best_value: float | None
    best_step: int | None
    reason: str | None = None


@dataclass(slots=True)
class EarlyStopper:
    """Track best metric value and trigger patience-based stopping."""

    enabled: bool
    metric: str
    mode: str
    min_delta: float
    min_delta_mode: str
    patience: int
    min_steps: int
    save_best: bool
    best_value: float | None = None
    best_step: int | None = None
    bad_evals: int = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EarlyStopper":
        payload = dict(config.get("evaluation", {}).get("early_stopping", {}))
        return cls(
            enabled=bool(payload.get("enabled", False)),
            metric=str(payload.get("metric", "")),
            mode=str(payload.get("mode", "max")).lower(),
            min_delta=float(payload.get("min_delta", 0.0)),
            min_delta_mode=str(payload.get("min_delta_mode", "absolute")).lower(),
            patience=int(payload.get("patience", 0)),
            min_steps=int(payload.get("min_steps", 0)),
            save_best=bool(payload.get("save_best", True)),
        )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable state summary."""
        return {
            "enabled": self.enabled,
            "metric": self.metric,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "min_delta_mode": self.min_delta_mode,
            "patience": self.patience,
            "min_steps": self.min_steps,
            "save_best": self.save_best,
            "best_value": self.best_value,
            "best_step": self.best_step,
            "bad_evals": self.bad_evals,
        }

    def _improved(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.min_delta_mode == "relative":
            if self.mode == "min":
                threshold = self.best_value * (1.0 - self.min_delta)
                return value < threshold
            threshold = self.best_value * (1.0 + self.min_delta)
            return value > threshold
        if self.mode == "min":
            return value < (self.best_value - self.min_delta)
        return value > (self.best_value + self.min_delta)

    def update(self, step: int, metrics: Mapping[str, Any]) -> EarlyStoppingDecision:
        """Consume an evaluation summary and decide whether to save/stop."""
        if not self.enabled or not self.metric:
            return EarlyStoppingDecision(
                enabled=False,
                metric=self.metric,
                value=None,
                improved=False,
                should_save=False,
                should_stop=False,
                bad_evals=self.bad_evals,
                best_value=self.best_value,
                best_step=self.best_step,
            )

        raw_value = metrics.get(self.metric)
        if raw_value is None:
            return EarlyStoppingDecision(
                enabled=True,
                metric=self.metric,
                value=None,
                improved=False,
                should_save=False,
                should_stop=False,
                bad_evals=self.bad_evals,
                best_value=self.best_value,
                best_step=self.best_step,
                reason=f"metric_missing:{self.metric}",
            )

        value = float(raw_value)
        improved = self._improved(value)
        if improved:
            self.best_value = value
            self.best_step = step
            self.bad_evals = 0
        elif step >= self.min_steps:
            self.bad_evals += 1

        should_stop = (
            step >= self.min_steps
            and self.patience > 0
            and self.bad_evals >= self.patience
        )
        return EarlyStoppingDecision(
            enabled=True,
            metric=self.metric,
            value=value,
            improved=improved,
            should_save=improved and self.save_best,
            should_stop=should_stop,
            bad_evals=self.bad_evals,
            best_value=self.best_value,
            best_step=self.best_step,
        )
