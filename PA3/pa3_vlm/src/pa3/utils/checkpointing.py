from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(path: str, **state: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)

