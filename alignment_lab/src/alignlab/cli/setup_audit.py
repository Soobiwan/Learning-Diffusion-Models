"""PA2 setup-audit command for data parsing and model-loading checks."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

from ._shared import (
    build_argument_parser,
    load_training_examples,
    model_spec_from_config,
    resolve_config,
    summarize_config,
)
from ..common.utils import count_parameters
from ..eval.pa2_tools import preview_canonical_examples
from ..eval.reports import experiment_table_path, write_json


def _cuda_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "allocated_gb": None,
            "reserved_gb": None,
            "peak_allocated_gb": None,
            "total_vram_gb": None,
            "fits_available_vram": None,
        }
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return {
        "cuda_available": True,
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "peak_allocated_gb": peak,
        "total_vram_gb": total,
        "fits_available_vram": peak <= total,
    }


def _tokenizer_summary(tokenizer: Any) -> dict[str, Any]:
    return {
        "padding_side": getattr(tokenizer, "padding_side", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
    }


def _model_summary(model: Any) -> dict[str, Any]:
    device = None
    try:
        device = str(next(model.parameters()).device)
    except StopIteration:
        device = "unknown"
    except AttributeError:
        device = "unknown"
    return {
        "device": device,
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "total_parameters": count_parameters(model, trainable_only=False),
    }


def _load_policy_section(config: dict[str, Any]) -> dict[str, Any]:
    from ..models.policy import load_policy_bundle

    bundle = load_policy_bundle(model_spec_from_config(config))
    summary = {
        "kind": "policy",
        "tokenizer": _tokenizer_summary(bundle.tokenizer),
        "model": _model_summary(bundle.model),
        "parameter_summary": bundle.parameter_summary,
        "cuda": _cuda_summary(),
    }
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _load_reference_section(config: dict[str, Any]) -> dict[str, Any] | None:
    if "reference_model" not in config:
        return None
    from ..models.reference import build_reference_bundle

    bundle = build_reference_bundle(spec=model_spec_from_config(config, "reference_model"))
    summary = {
        "kind": "reference",
        "tokenizer": _tokenizer_summary(bundle.tokenizer),
        "model": _model_summary(bundle.model),
        "cuda": _cuda_summary(),
    }
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _load_reward_section(config: dict[str, Any]) -> dict[str, Any] | None:
    spec_key = "reward_model" if "reward_model" in config else ("model" if config["method"]["name"] == "reward_model" else None)
    if spec_key is None:
        return None
    from ..models.reward import load_reward_bundle

    bundle = load_reward_bundle(model_spec_from_config(config, spec_key), freeze=(spec_key == "reward_model"))
    summary = {
        "kind": "reward_model",
        "tokenizer": _tokenizer_summary(bundle.tokenizer),
        "model": _model_summary(bundle.model),
        "cuda": _cuda_summary(),
    }
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _load_value_section(config: dict[str, Any]) -> dict[str, Any] | None:
    if "value_model" not in config:
        return None
    from ..models.value import load_value_bundle

    bundle = load_value_bundle(model_spec_from_config(config, "value_model"))
    summary = {
        "kind": "value_model",
        "tokenizer": _tokenizer_summary(bundle.tokenizer),
        "model": _model_summary(bundle.model),
        "cuda": _cuda_summary(),
    }
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = build_argument_parser("Run the PA2 setup audit.")
    args = parser.parse_args()
    config = resolve_config(args.config, sample_limit=args.sample_limit, max_steps=args.max_steps)
    print(summarize_config(config))
    if args.dry_run:
        return

    examples = load_training_examples(config)
    preview = preview_canonical_examples(examples, limit=3)
    method_name = str(config.get("method", {}).get("name", "")).lower()
    primary_model_spec = model_spec_from_config(config) if "model" in config else None
    sections = []
    if primary_model_spec is not None and primary_model_spec.task_type.upper() == "CAUSAL_LM" and method_name != "reward_model":
        sections.append(_load_policy_section(config))
    reward_section = _load_reward_section(config)
    if reward_section is not None:
        sections.append(reward_section)
    reference_section = _load_reference_section(config)
    if reference_section is not None:
        sections.append(reference_section)
    value_section = _load_value_section(config)
    if value_section is not None:
        sections.append(value_section)
    payload = {
        "experiment_name": config["experiment_name"],
        "config_path": config["config_path"],
        "preview_examples": preview,
        "sections": sections,
    }
    output_path = write_json(experiment_table_path(config, "setup_audit"), payload)
    print(json.dumps(payload, indent=2))
    print(f"setup_audit_json={output_path}")


if __name__ == "__main__":
    main()
