"""Minimal import sanity script."""

from __future__ import annotations

import importlib
import importlib.util


MODULES = [
    "alignlab",
    "alignlab.common.config",
    "alignlab.data.loaders",
    "alignlab.objectives.dpo",
    "alignlab.rollout.gae",
    "alignlab.eval.gsm8k_eval",
    "alignlab.eval.pa2_tools",
]


def main() -> None:
    optional_dependencies = {"transformers", "accelerate", "datasets"}
    missing = sorted(
        dependency for dependency in optional_dependencies if importlib.util.find_spec(dependency) is None
    )
    for module_name in MODULES:
        importlib.import_module(module_name)
    if missing:
        print(f"Import check passed for lightweight modules. Missing optional dependencies: {', '.join(missing)}")
        return
    for module_name in ["alignlab.models.factory", "alignlab.trainers.online_rl_trainer", "alignlab.cli.compare_pa2"]:
        importlib.import_module(module_name)
    print("Import check passed.")


if __name__ == "__main__":
    main()
