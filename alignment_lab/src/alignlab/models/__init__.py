"""Model subpackage.

Keep package-level imports lightweight so pure-PyTorch utilities can be imported
without forcing heavyweight optional dependencies during test collection.
"""

from .specs import ModelSpec

__all__ = ["ModelSpec"]
