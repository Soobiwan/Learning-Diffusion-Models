"""Dataset adapter interfaces and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable

from .schemas import PreferenceExample, SFTExample, VerifiableExample

CanonicalExample = PreferenceExample | SFTExample | VerifiableExample


class DatasetAdapter(ABC):
    """Base interface for converting raw dataset rows into canonical schemas."""

    name: ClassVar[str]
    dataset_path: ClassVar[str | None] = None

    @abstractmethod
    def raw_to_canonical(self, raw_example: dict[str, Any]) -> CanonicalExample:
        """Convert one raw dataset row to an internal canonical schema."""

    def raw_to_sft(self, raw_example: dict[str, Any]) -> SFTExample:
        """Optionally expose a dataset as SFT data."""
        canonical = self.raw_to_canonical(raw_example)
        if isinstance(canonical, SFTExample):
            return canonical
        if isinstance(canonical, PreferenceExample):
            return SFTExample(
                prompt=canonical.prompt,
                response=canonical.chosen,
                meta={**canonical.meta, "source_schema": "preference"},
            )
        raise TypeError(f"Adapter '{self.name}' cannot convert {type(canonical).__name__} to SFT.")

    def map_dataset(self, raw_examples: Iterable[dict[str, Any]]) -> list[CanonicalExample]:
        """Materialize a sequence of raw rows into canonical dataclasses."""
        return [self.raw_to_canonical(example) for example in raw_examples]


class AdapterRegistry:
    """Global registry of dataset adapters."""

    _registry: dict[str, type[DatasetAdapter]] = {}

    @classmethod
    def register(cls, adapter_cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
        """Register an adapter class by name."""
        cls._registry[adapter_cls.name] = adapter_cls
        return adapter_cls

    @classmethod
    def create(cls, name: str) -> DatasetAdapter:
        """Instantiate an adapter from the registry."""
        if name not in cls._registry:
            raise KeyError(f"Unknown dataset adapter '{name}'. Known adapters: {sorted(cls._registry)}")
        return cls._registry[name]()

    @classmethod
    def names(cls) -> list[str]:
        """List registered adapter names."""
        return sorted(cls._registry)
