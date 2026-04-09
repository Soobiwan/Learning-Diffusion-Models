"""Shared CLI helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from ..common.config import load_experiment_config
from ..data.collators import PreferenceCollator, PromptOnlyCollator, RewardModelCollator, SFTCollator
from ..data.loaders import load_canonical_dataset
from ..data.schemas import PreferenceExample, SFTExample, VerifiableExample
from ..models.specs import ModelSpec
from ..objectives.dpo import DPOObjective
from ..objectives.grpo import GRPOObjective
from ..objectives.ppo import PPOObjective
from ..objectives.rlvr import RLVRObjective
from ..objectives.sampo import SamPOObjective
from ..objectives.simpo import SimPOObjective


def build_argument_parser(description: str) -> argparse.ArgumentParser:
    """Create a compact shared CLI parser."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and exit.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional dataset cap.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional training step cap.")
    return parser


def resolve_config(config_path: str, sample_limit: int | None = None, max_steps: int | None = None) -> dict[str, Any]:
    """Load an experiment config and optionally override dataset/train limits."""
    config = load_experiment_config(config_path)
    if sample_limit is not None:
        config.setdefault("data", {})
        config["data"]["sample_limit"] = sample_limit
        evaluation = config.setdefault("evaluation", {})
        for key in ("num_eval_prompts", "num_eval_pairs", "sample_table_size"):
            current = evaluation.get(key)
            if current is None:
                evaluation[key] = sample_limit
            else:
                evaluation[key] = min(int(current), sample_limit)
    if max_steps is not None:
        config.setdefault("training", {})
        config["training"]["max_steps"] = max_steps
        config.setdefault("method", {})
        config["method"]["max_steps"] = max_steps
    return config


def summarize_config(config: dict[str, Any]) -> str:
    """Return a compact human-readable summary."""
    model = config.get("model", {}).get("hf_path", "<unspecified>")
    method = config.get("method", {}).get("name", "<unspecified>")
    adapter = config.get("data", {}).get("adapter", "<unspecified>")
    return f"experiment={config['experiment_name']} method={method} dataset={adapter} model={model}"


def configured_max_steps(config: dict[str, Any]) -> int | None:
    """Resolve an optional max-step override from method or training config."""
    value = config["method"].get("max_steps", config["training"].get("max_steps"))
    if value is None:
        return None
    return int(value)


def configured_num_epochs(config: dict[str, Any]) -> int:
    """Resolve the offline training epoch count."""
    return int(config["training"].get("num_epochs", 1))


def configured_gradient_accumulation_steps(config: dict[str, Any]) -> int:
    """Resolve gradient accumulation from method or global config."""
    value = config["method"].get("gradient_accumulation_steps", config.get("gradient_accumulation_steps", 1))
    return int(value)


def evaluation_every_steps(config: dict[str, Any]) -> int:
    """Resolve periodic evaluation cadence."""
    return int(config.get("evaluation", {}).get("eval_every_steps", 0))


def load_training_examples(config: dict[str, Any]) -> list[Any]:
    """Load canonical examples through the adapter registry."""
    return load_dataset_examples(config, split=config["data"].get("split", "train"))


def load_dataset_examples(
    config: dict[str, Any],
    split: str,
    sample_limit: int | None = None,
) -> list[Any]:
    """Load canonical examples for an arbitrary dataset split."""
    data_cfg = config["data"]
    dataset_kwargs = {}
    if "name" in data_cfg:
        dataset_kwargs["name"] = data_cfg["name"]
    return load_canonical_dataset(
        adapter_name=data_cfg["adapter"],
        path=data_cfg.get("path"),
        split=split,
        sample_limit=data_cfg.get("sample_limit") if sample_limit is None else sample_limit,
        dataset_kwargs=dataset_kwargs,
    )


def load_eval_examples(config: dict[str, Any]) -> list[Any]:
    """Load canonical evaluation examples."""
    return load_dataset_examples(config, split=config["data"].get("eval_split", "test"))


def make_dataloader(examples: list[Any], collator: Any, batch_size: int) -> DataLoader:
    """Wrap examples in a simple DataLoader."""
    return DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=collator)


def build_pairwise_objective(method_name: str, method_cfg: dict[str, Any]) -> Any:
    """Select the pairwise objective implementation."""
    name = method_name.lower()
    if name == "dpo":
        return DPOObjective(
            beta=float(method_cfg.get("beta", 0.1)),
            label_smoothing=float(method_cfg.get("label_smoothing", 0.0)),
        )
    if name == "simpo":
        return SimPOObjective()
    if name == "sampo":
        return SamPOObjective()
    raise KeyError(f"Unsupported pairwise objective '{method_name}'.")


def build_online_objective(method_name: str, method_cfg: dict[str, Any]) -> Any:
    """Select an online RL objective."""
    name = method_name.lower()
    if name == "ppo":
        return PPOObjective(
            clip_range=float(method_cfg.get("clip_range", 0.2)),
            value_clip_range=float(method_cfg.get("value_clip_range", 0.2)),
            value_loss_coef=float(method_cfg.get("value_loss_coef", 0.5)),
            entropy_coef=float(method_cfg.get("entropy_coef", 0.0)),
        )
    if name == "grpo":
        return GRPOObjective(
            beta_kl=float(method_cfg.get("beta_kl", 0.02)),
            clip_range=float(method_cfg.get("clip_range", 0.2)),
        )
    if name == "rlvr":
        return RLVRObjective(
            beta_kl=float(method_cfg.get("beta_kl", 0.01)),
            clip_range=float(method_cfg.get("clip_range", 0.2)),
        )
    raise KeyError(f"Unsupported online RL objective '{method_name}'.")


def build_sft_collator(tokenizer: Any, config: dict[str, Any]) -> SFTCollator:
    """Create the SFT collator from config."""
    max_length = int(config["method"].get("max_sequence_length", config["tokenization"]["max_sequence_length"]))
    return SFTCollator(tokenizer=tokenizer, max_length=max_length)


def build_rm_collator(tokenizer: Any, config: dict[str, Any]) -> RewardModelCollator:
    """Create the RM collator from config."""
    max_length = int(config["method"].get("max_sequence_length", config["tokenization"]["max_sequence_length"]))
    return RewardModelCollator(tokenizer=tokenizer, max_length=max_length)


def build_pairwise_collator(tokenizer: Any, config: dict[str, Any]) -> PreferenceCollator:
    """Create the pairwise collator from config."""
    max_length = int(config["method"].get("max_sequence_length", config["tokenization"]["max_sequence_length"]))
    return PreferenceCollator(tokenizer=tokenizer, max_length=max_length)


def build_prompt_collator(tokenizer: Any, config: dict[str, Any]) -> PromptOnlyCollator:
    """Create the prompt-only collator from config."""
    max_length = int(config["method"].get("max_prompt_length", config["tokenization"]["max_prompt_length"]))
    return PromptOnlyCollator(tokenizer=tokenizer, max_length=max_length)


def model_spec_from_config(config: dict[str, Any], key: str = "model") -> ModelSpec:
    """Build a `ModelSpec` from a config section."""
    return ModelSpec.from_dict(config[key])


def preference_examples(examples: list[Any]) -> list[PreferenceExample]:
    """Filter canonical examples to pairwise preferences."""
    filtered = [example for example in examples if isinstance(example, PreferenceExample)]
    if len(filtered) != len(examples):
        raise TypeError("Expected only PreferenceExample rows for this CLI.")
    return filtered


def sft_examples(examples: list[Any]) -> list[SFTExample]:
    """Convert preference or direct SFT examples into `SFTExample` rows."""
    converted: list[SFTExample] = []
    for example in examples:
        if isinstance(example, SFTExample):
            converted.append(example)
        elif isinstance(example, PreferenceExample):
            converted.append(
                SFTExample(
                    prompt=example.prompt,
                    response=example.chosen,
                    meta={**example.meta, "source_schema": "preference"},
                )
            )
        else:
            raise TypeError("SFT CLI supports SFTExample or PreferenceExample rows.")
    return converted


def verifiable_examples(examples: list[Any]) -> list[VerifiableExample]:
    """Filter canonical examples to verifiable RLVR rows."""
    filtered = [example for example in examples if isinstance(example, VerifiableExample)]
    if len(filtered) != len(examples):
        raise TypeError("Expected only VerifiableExample rows for this CLI.")
    return filtered


def require_checkpoint(path: str, message: str) -> None:
    """Raise a helpful error when a local checkpoint path is missing."""
    checkpoint_path = Path(path)
    if checkpoint_path.is_absolute() or (checkpoint_path.parts and checkpoint_path.parts[0] == "artifacts"):
        if not checkpoint_path.exists():
            raise FileNotFoundError(message)
