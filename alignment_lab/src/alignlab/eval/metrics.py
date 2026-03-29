"""Core evaluation metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def mean(values: Sequence[float]) -> float:
    """Return a simple arithmetic mean."""
    return float(np.mean(list(values))) if values else 0.0


def accuracy(predictions: Sequence[bool]) -> float:
    """Compute binary accuracy."""
    return mean([1.0 if prediction else 0.0 for prediction in predictions])
