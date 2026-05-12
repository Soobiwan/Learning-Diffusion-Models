import ast
import json
from pathlib import Path
from typing import Any, Dict


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip("\"'")


def load_config(path: str) -> Dict[str, Any]:
    """Small YAML subset reader to avoid depending on non-assignment packages."""
    cfg: Dict[str, Any] = {}
    stack = [(0, cfg)]
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack and indent < stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            node: Dict[str, Any] = {}
            parent[key] = node
            stack.append((indent + 2, node))
        else:
            parent[key] = _parse_scalar(value)
    return cfg


def ensure_dirs(*paths: str) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

