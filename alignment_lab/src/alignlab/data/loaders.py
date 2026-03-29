"""Dataset loading and registry helpers."""

from __future__ import annotations

from typing import Any

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - depends on local environment
    load_dataset = None  # type: ignore[assignment]

from .base import AdapterRegistry, DatasetAdapter
from .schemas import PreferenceExample, SFTExample, VerifiableExample

# Import for registration side effects.
from . import adapters as _adapters  # noqa: F401


def get_adapter(name: str) -> DatasetAdapter:
    """Instantiate a dataset adapter by name."""
    return AdapterRegistry.create(name)


def list_adapters() -> list[str]:
    """Return registered adapter names."""
    return AdapterRegistry.names()


def load_canonical_dataset(
    adapter_name: str,
    path: str | None = None,
    split: str = "train",
    sample_limit: int | None = None,
    dataset_kwargs: dict[str, Any] | None = None,
) -> list[PreferenceExample | SFTExample | VerifiableExample]:
    """Load a Hugging Face dataset split and map it into canonical schemas."""
    adapter = get_adapter(adapter_name)
    dataset_kwargs = dataset_kwargs or {}
    dataset_path = path or adapter.dataset_path
    if dataset_path is None:
        raise ValueError(f"Adapter '{adapter_name}' does not define a default dataset path.")
    if load_dataset is None:
        raise ImportError("datasets is required to load Hugging Face datasets. Install it from requirements.txt.")

    dataset = load_dataset(dataset_path, split=split, **dataset_kwargs)
    if sample_limit is not None:
        dataset = dataset.select(range(min(sample_limit, len(dataset))))
    return [adapter.raw_to_canonical(dict(row)) for row in dataset]
