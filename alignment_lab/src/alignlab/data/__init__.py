"""Data abstractions and canonical schemas."""

from .loaders import AdapterRegistry, get_adapter, list_adapters, load_canonical_dataset
from .schemas import PreferenceExample, SFTExample, VerifiableExample

__all__ = [
    "AdapterRegistry",
    "PreferenceExample",
    "SFTExample",
    "VerifiableExample",
    "get_adapter",
    "list_adapters",
    "load_canonical_dataset",
]
