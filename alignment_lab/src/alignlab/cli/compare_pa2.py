"""Cross-method PA2 comparison and DPO beta ablation reporting."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

from ..common.checkpointing import final_checkpoint_dir
from ..eval.pa2_tools import load_json_if_exists, summarize_resource_payload
from ..eval.pipeline import evaluate_hh_policy, evaluate_rlvr_policy
from ..eval.reports import (
    experiment_plot_path,
    experiment_sample_path,
    experiment_table_path,
    plot_scalar_sweep,
    write_csv_rows,
    write_generation_artifacts,
    write_json,
)
from ..rollout.verifiers import GSM8KAnswerVerifier


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _config_path(name: str) -> Path:
    return _project_root() / "configs" / "experiment" / f"{name}.yaml"


def _checkpoint_spec(config: dict[str, object], key: str, checkpoint_dir: Path):
    from ._shared import model_spec_from_config

    spec = model_spec_from_config(config, key)
    return replace(spec, hf_path=str(checkpoint_dir), tokenizer_path=str(checkpoint_dir))


def _comparison_config(name: str) -> dict[str, Any]:
    return {
        "experiment_name": name,
        "evaluation": {
            "histogram_bins": 20,
        },
    }


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    candidate = Path(path)
    with candidate.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON list in '{candidate}', found {type(payload)!r}.")
    return payload


def _combine_hh_sample_rows(sample_payloads: dict[str, Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not sample_payloads:
        return []
    ordered_methods = list(sample_payloads)
    row_count = min(len(rows) for rows in sample_payloads.values())
    combined: list[dict[str, Any]] = []
    for idx in range(row_count):
        prompt = sample_payloads[ordered_methods[0]][idx]["prompt"]
        row: dict[str, Any] = {"prompt": prompt}
        for method_name, rows in sample_payloads.items():
            row[f"{method_name}_response"] = rows[idx]["candidate_response"]
            row[f"{method_name}_score"] = rows[idx]["candidate_reward"]
        combined.append(row)
    return combined


def _combine_rlvr_sample_rows(sample_payloads: dict[str, Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not sample_payloads:
        return []
    ordered_methods = list(sample_payloads)
    row_count = min(len(rows) for rows in sample_payloads.values())
    combined: list[dict[str, Any]] = []
    for idx in range(row_count):
        prompt = sample_payloads[ordered_methods[0]][idx]["prompt"]
        row: dict[str, Any] = {"prompt": prompt, "gold_answer": sample_payloads[ordered_methods[0]][idx]["gold_answer"]}
        for method_name, rows in sample_payloads.items():
            row[f"{method_name}_response"] = rows[idx]["response"]
            row[f"{method_name}_correct"] = rows[idx]["correct"]
            row[f"{method_name}_has_final_answer"] = rows[idx]["has_final_answer"]
        combined.append(row)
    return combined


def _write_combined_outputs(
    config_name: str,
    summary_rows: Sequence[dict[str, Any]],
    sample_rows: Sequence[dict[str, Any]],
) -> dict[str, str]:
    config = _comparison_config(config_name)
    summary_json = write_json(experiment_table_path(config, "summary"), list(summary_rows))
    summary_csv = write_csv_rows(experiment_table_path(config, "summary", ".csv"), list(summary_rows))
    sample_paths = write_generation_artifacts(config, "samples", list(sample_rows))
    return {
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        **sample_paths,
    }


def _evaluate_hh_suite(num_prompts: int, sample_limit: int) -> dict[str, Any]:
    from ._shared import load_eval_examples, model_spec_from_config, preference_examples, require_checkpoint, resolve_config
    from ..models.policy import load_policy_bundle
    from ..models.reference import build_reference_bundle
    from ..models.reward import load_reward_bundle
    from ..rollout.rewards import LearnedRewardFunction

    method_names = ["pa2_sft_hh_rlhf", "pa2_dpo_hh_rlhf", "pa2_ppo_hh_rlhf", "pa2_grpo_hh_rlhf"]
    configs = {name: resolve_config(str(_config_path(name))) for name in method_names}
    baseline_config = configs["pa2_sft_hh_rlhf"]
    shared_eval_examples = preference_examples(load_eval_examples(configs["pa2_dpo_hh_rlhf"]))[:num_prompts]

    baseline_checkpoint = final_checkpoint_dir(baseline_config)
    require_checkpoint(str(baseline_checkpoint), f"Missing checkpoint '{baseline_checkpoint}'.")
    baseline_bundle = load_policy_bundle(_checkpoint_spec(baseline_config, "model", baseline_checkpoint))

    reference_bundle = build_reference_bundle(spec=model_spec_from_config(configs["pa2_dpo_hh_rlhf"], "reference_model"))
    reward_bundle = load_reward_bundle(model_spec_from_config(configs["pa2_dpo_hh_rlhf"], "reward_model"), freeze=True)
    reward_function = LearnedRewardFunction(
        model=reward_bundle.model,
        tokenizer=reward_bundle.tokenizer,
        max_length=int(
            configs["pa2_dpo_hh_rlhf"]["method"].get(
                "max_sequence_length",
                configs["pa2_dpo_hh_rlhf"]["tokenization"]["max_sequence_length"],
            )
        ),
    )

    summary_rows: list[dict[str, Any]] = []
    sample_payloads: dict[str, list[dict[str, Any]]] = {}
    for method_name, method_config in configs.items():
        checkpoint_dir = final_checkpoint_dir(method_config)
        require_checkpoint(str(checkpoint_dir), f"Missing checkpoint '{checkpoint_dir}'.")
        candidate_bundle = baseline_bundle if method_name == "pa2_sft_hh_rlhf" else load_policy_bundle(
            _checkpoint_spec(method_config, "model", checkpoint_dir)
        )
        eval_config = deepcopy(method_config)
        eval_config["experiment_name"] = f"{method_name}_compare_eval"
        eval_config.setdefault("evaluation", {})
        eval_config["evaluation"]["num_eval_prompts"] = num_prompts
        eval_config["evaluation"]["sample_table_size"] = sample_limit
        summary = evaluate_hh_policy(
            eval_config,
            candidate_model=candidate_bundle.model,
            candidate_tokenizer=candidate_bundle.tokenizer,
            reference_model=reference_bundle.model,
            reward_function=reward_function,
            prompt_examples=shared_eval_examples,
            pair_examples=shared_eval_examples,
            baseline_model=baseline_bundle.model,
            baseline_tokenizer=baseline_bundle.tokenizer,
            stem="comparison",
        )
        resource_path = _project_root() / "artifacts" / "tables" / f"{method_name}_resource_summary.json"
        summary_rows.append(
            {
                "method": method_name.replace("pa2_", "").replace("_hh_rlhf", "").upper(),
                "rm_score_mean": summary["rm_score_mean"],
                "rm_win_rate_vs_sft": summary["rm_win_rate_vs_sft"],
                "kl_from_reference": summary["kl_from_reference"],
                "preference_accuracy": summary.get("preference_accuracy"),
                "mean_response_length": summary.get("mean_response_length"),
                **summarize_resource_payload(method_name, load_json_if_exists(resource_path)),
            }
        )
        sample_payloads[method_name.replace("pa2_", "").replace("_hh_rlhf", "")] = _load_rows(summary["artifacts"]["json"])
        if candidate_bundle is not baseline_bundle:
            del candidate_bundle

    combined_samples = _combine_hh_sample_rows(sample_payloads)
    artifacts = _write_combined_outputs("pa2_method_comparison_hh", summary_rows, combined_samples)
    return {"rows": summary_rows, "artifacts": artifacts}


def _evaluate_rlvr_suite(num_prompts: int, sample_limit: int) -> dict[str, Any]:
    from ._shared import load_eval_examples, model_spec_from_config, require_checkpoint, resolve_config, verifiable_examples
    from ..models.policy import load_policy_bundle
    from ..models.reference import build_reference_bundle

    sft_config = resolve_config(str(_config_path("pa2_sft_hh_rlhf")))
    rlvr_config = resolve_config(str(_config_path("pa2_rlvr_gsm8k")))
    shared_eval_examples = verifiable_examples(load_eval_examples(rlvr_config))[:num_prompts]
    verifier = GSM8KAnswerVerifier()

    baseline_checkpoint = final_checkpoint_dir(sft_config)
    require_checkpoint(str(baseline_checkpoint), f"Missing checkpoint '{baseline_checkpoint}'.")
    baseline_bundle = load_policy_bundle(_checkpoint_spec(sft_config, "model", baseline_checkpoint))
    reference_bundle = build_reference_bundle(spec=model_spec_from_config(rlvr_config, "reference_model"))

    method_rows: list[dict[str, Any]] = []
    sample_payloads: dict[str, list[dict[str, Any]]] = {}
    for method_name, method_config in [("pa2_sft_hh_rlhf", sft_config), ("pa2_rlvr_gsm8k", rlvr_config)]:
        checkpoint_dir = final_checkpoint_dir(method_config if method_name == "pa2_rlvr_gsm8k" else sft_config)
        require_checkpoint(str(checkpoint_dir), f"Missing checkpoint '{checkpoint_dir}'.")
        candidate_bundle = baseline_bundle if method_name == "pa2_sft_hh_rlhf" else load_policy_bundle(
            _checkpoint_spec(method_config, "model", checkpoint_dir)
        )
        eval_config = deepcopy(method_config)
        eval_config["experiment_name"] = f"{method_name}_compare_eval"
        eval_config.setdefault("evaluation", {})
        eval_config["evaluation"]["num_eval_prompts"] = num_prompts
        eval_config["evaluation"]["sample_table_size"] = sample_limit
        summary = evaluate_rlvr_policy(
            eval_config,
            candidate_model=candidate_bundle.model,
            candidate_tokenizer=candidate_bundle.tokenizer,
            reference_model=reference_bundle.model,
            examples=shared_eval_examples,
            verifier=verifier,
            stem="comparison",
        )
        resource_path = _project_root() / "artifacts" / "tables" / f"{method_name}_resource_summary.json"
        method_rows.append(
            {
                "method": "RLVR" if method_name == "pa2_rlvr_gsm8k" else "SFT",
                "pass_at_1": summary["pass_at_1"],
                "format_compliance_rate": summary["format_compliance_rate"],
                "mean_response_length": summary["mean_response_length"],
                "kl_from_reference": summary["kl_from_reference"],
                **summarize_resource_payload(method_name, load_json_if_exists(resource_path)),
            }
        )
        key = "rlvr" if method_name == "pa2_rlvr_gsm8k" else "sft"
        sample_payloads[key] = _load_rows(summary["artifacts"]["json"])
        if candidate_bundle is not baseline_bundle:
            del candidate_bundle

    combined_samples = _combine_rlvr_sample_rows(sample_payloads)
    artifacts = _write_combined_outputs("pa2_method_comparison_rlvr", method_rows, combined_samples)
    return {"rows": method_rows, "artifacts": artifacts}


def _evaluate_dpo_beta_ablation(num_prompts: int) -> dict[str, Any]:
    from ._shared import load_eval_examples, model_spec_from_config, preference_examples, require_checkpoint, resolve_config
    from ..models.policy import load_policy_bundle
    from ..models.reference import build_reference_bundle
    from ..models.reward import load_reward_bundle
    from ..rollout.rewards import LearnedRewardFunction

    dpo_configs = sorted(_project_root().glob("configs/experiment/pa2_dpo_beta_*.yaml"))
    if not dpo_configs:
        return {"rows": [], "artifacts": {}}

    sft_config = resolve_config(str(_config_path("pa2_sft_hh_rlhf")))
    base_config = resolve_config(str(_config_path("pa2_dpo_hh_rlhf")))
    shared_eval_examples = preference_examples(load_eval_examples(base_config))[:num_prompts]

    baseline_checkpoint = final_checkpoint_dir(sft_config)
    require_checkpoint(str(baseline_checkpoint), f"Missing checkpoint '{baseline_checkpoint}'.")
    baseline_bundle = load_policy_bundle(_checkpoint_spec(sft_config, "model", baseline_checkpoint))
    reference_bundle = build_reference_bundle(spec=model_spec_from_config(base_config, "reference_model"))
    reward_bundle = load_reward_bundle(model_spec_from_config(base_config, "reward_model"), freeze=True)
    reward_function = LearnedRewardFunction(
        model=reward_bundle.model,
        tokenizer=reward_bundle.tokenizer,
        max_length=int(base_config["method"].get("max_sequence_length", base_config["tokenization"]["max_sequence_length"])),
    )

    rows: list[dict[str, Any]] = []
    for config_path in dpo_configs:
        config = resolve_config(str(config_path))
        checkpoint_dir = final_checkpoint_dir(config)
        if not checkpoint_dir.exists():
            continue
        candidate_bundle = load_policy_bundle(_checkpoint_spec(config, "model", checkpoint_dir))
        eval_config = deepcopy(config)
        eval_config["experiment_name"] = f"{config['experiment_name']}_ablation_eval"
        eval_config.setdefault("evaluation", {})
        eval_config["evaluation"]["num_eval_prompts"] = num_prompts
        eval_config["evaluation"]["num_eval_pairs"] = num_prompts
        summary = evaluate_hh_policy(
            eval_config,
            candidate_model=candidate_bundle.model,
            candidate_tokenizer=candidate_bundle.tokenizer,
            reference_model=reference_bundle.model,
            reward_function=reward_function,
            prompt_examples=shared_eval_examples,
            pair_examples=shared_eval_examples,
            baseline_model=baseline_bundle.model,
            baseline_tokenizer=baseline_bundle.tokenizer,
            stem="ablation",
        )
        resource_path = _project_root() / "artifacts" / "tables" / f"{config['experiment_name']}_resource_summary.json"
        rows.append(
            {
                "beta": float(config["method"]["beta"]),
                "experiment_name": config["experiment_name"],
                "rm_score_mean": summary["rm_score_mean"],
                "rm_win_rate_vs_sft": summary["rm_win_rate_vs_sft"],
                "preference_accuracy": summary.get("preference_accuracy"),
                "kl_from_reference": summary["kl_from_reference"],
                **summarize_resource_payload(config["experiment_name"], load_json_if_exists(resource_path)),
            }
        )
        del candidate_bundle

    compare_config = _comparison_config("pa2_dpo_beta_ablation")
    summary_json = write_json(experiment_table_path(compare_config, "summary"), rows)
    summary_csv = write_csv_rows(experiment_table_path(compare_config, "summary", ".csv"), rows)
    plot_path = plot_scalar_sweep(
        experiment_plot_path(compare_config, "beta_sweep"),
        rows,
        x_key="beta",
        metric_keys=["rm_score_mean", "preference_accuracy", "kl_from_reference"],
        title="PA2 DPO Beta Sweep",
        x_label="beta",
    ) if rows else None
    return {
        "rows": rows,
        "artifacts": {
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
            "plot_png": str(plot_path) if plot_path is not None else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PA2 comparison tables and ablation reports.")
    parser.add_argument("--num-prompts", type=int, default=200, help="Shared evaluation prompt count per suite.")
    parser.add_argument("--sample-limit", type=int, default=5, help="Number of prompts in side-by-side sample tables.")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip the DPO beta ablation aggregation.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the PA2 config paths and exit.")
    args = parser.parse_args()

    config_names = [
        "pa2_sft_hh_rlhf",
        "pa2_rm_hh_rlhf",
        "pa2_dpo_hh_rlhf",
        "pa2_ppo_hh_rlhf",
        "pa2_grpo_hh_rlhf",
        "pa2_rlvr_gsm8k",
    ]
    print("PA2 comparison configs:")
    for name in config_names:
        print(f"- {name}: {_config_path(name)}")
    if args.dry_run:
        return

    hh_summary = _evaluate_hh_suite(num_prompts=args.num_prompts, sample_limit=args.sample_limit)
    print(f"hh_comparison={hh_summary['artifacts']}")
    rlvr_summary = _evaluate_rlvr_suite(num_prompts=args.num_prompts, sample_limit=args.sample_limit)
    print(f"rlvr_comparison={rlvr_summary['artifacts']}")
    if not args.skip_ablation:
        ablation_summary = _evaluate_dpo_beta_ablation(num_prompts=args.num_prompts)
        print(f"dpo_beta_ablation={ablation_summary['artifacts']}")


if __name__ == "__main__":
    main()
