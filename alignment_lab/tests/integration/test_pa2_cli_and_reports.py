from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

from alignlab.cli import compare_pa2, setup_audit, train_online, train_sft
from alignlab.common.config import load_experiment_config
from alignlab.data.schemas import PreferenceExample, VerifiableExample
from tests.helpers import DummyCausalLM, DummyTokenizer


def _write_json_file(path: str | Path, payload: object) -> Path:
    output_path = Path(path)
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return output_path


def test_setup_audit_main_writes_preview(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "setup_audit.json"
    monkeypatch.setattr(
        setup_audit,
        "resolve_config",
        lambda *args, **kwargs: {
            "experiment_name": "pa2_setup_audit_test",
            "config_path": "unused",
            "method": {"name": "ppo"},
            "model": {"hf_path": "dummy-model", "family": "smollm"},
            "data": {"adapter": "hh_rlhf"},
        },
    )
    monkeypatch.setattr(
        setup_audit,
        "load_training_examples",
        lambda config: [PreferenceExample(prompt="p", chosen="c", rejected="r")],
    )
    monkeypatch.setattr(
        setup_audit,
        "_load_policy_section",
        lambda config: {"kind": "policy", "tokenizer": {"padding_side": "left"}, "model": {"total_parameters": 1}},
    )
    monkeypatch.setattr(setup_audit, "_load_reference_section", lambda config: None)
    monkeypatch.setattr(setup_audit, "_load_reward_section", lambda config: None)
    monkeypatch.setattr(setup_audit, "_load_value_section", lambda config: None)
    monkeypatch.setattr(setup_audit, "experiment_table_path", lambda config, stem: output_path)
    monkeypatch.setattr(sys, "argv", ["setup_audit", "--config", "unused"])

    setup_audit.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["preview_examples"][0]["prompt"] == "p"
    assert payload["sections"][0]["kind"] == "policy"


def test_compare_pa2_combines_sample_rows_and_writes_outputs(monkeypatch, tmp_path: Path) -> None:
    hh_rows = compare_pa2._combine_hh_sample_rows(
        {
            "sft": [{"prompt": "p", "candidate_response": "s", "candidate_reward": 1.0}],
            "dpo": [{"prompt": "p", "candidate_response": "d", "candidate_reward": 2.0}],
        }
    )
    rlvr_rows = compare_pa2._combine_rlvr_sample_rows(
        {
            "sft": [{"prompt": "p", "response": "s", "correct": False, "has_final_answer": True, "gold_answer": "42"}],
            "rlvr": [{"prompt": "p", "response": "r", "correct": True, "has_final_answer": True, "gold_answer": "42"}],
        }
    )
    assert hh_rows[0]["dpo_score"] == 2.0
    assert rlvr_rows[0]["rlvr_correct"] is True

    base_dir = tmp_path / "artifacts"
    monkeypatch.setattr(
        compare_pa2,
        "experiment_table_path",
        lambda config, stem, suffix=".json": base_dir / f"{config['experiment_name']}_{stem}{suffix}",
    )
    outputs = compare_pa2._write_combined_outputs("pa2_compare_test", [{"method": "SFT"}], [{"prompt": "p"}])
    assert Path(outputs["summary_json"]).exists()
    assert Path(outputs["json"]).exists()


def test_train_sft_logs_heldout_perplexity_and_samples(monkeypatch, tmp_path: Path) -> None:
    tokenizer = DummyTokenizer()
    policy_bundle = SimpleNamespace(
        model=DummyCausalLM(vocab_size=tokenizer.vocab_size + 8),
        tokenizer=tokenizer,
        spec=SimpleNamespace(),
        parameter_summary={"trainable_parameters": 1, "total_parameters": 1},
    )
    log_path = tmp_path / "sft.jsonl"
    plot_path = tmp_path / "plot.png"
    table_path = tmp_path / "table.json"
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment_name": "pa2_sft_test",
        "config_path": "unused",
        "method": {
            "name": "sft",
            "train_batch_size": 1,
            "learning_rate": 1.0e-3,
            "max_sequence_length": 16,
        },
        "training": {"train_batch_size": 1, "eval_batch_size": 1, "learning_rate": 1.0e-3, "weight_decay": 0.0, "num_epochs": 1},
        "tokenization": {"max_sequence_length": 16, "max_prompt_length": 16},
        "generation": {"max_new_tokens": 1, "do_sample": False, "temperature": 1.0, "top_p": 1.0},
        "evaluation": {"num_eval_prompts": 1, "sample_table_size": 1, "sft_eval_every_steps": 1, "sft_sample_every_steps": 1},
        "max_grad_norm": 1.0,
        "dtype": "fp32",
        "gradient_accumulation_steps": 1,
        "model": {"hf_path": "dummy-policy", "family": "smollm", "dtype": "fp32"},
        "data": {"adapter": "hh_rlhf"},
        "reward_model": {"hf_path": str(tmp_path / "missing_reward"), "family": "llama"},
    }
    examples = [PreferenceExample(prompt="Human: hello Assistant:", chosen="good", rejected="bad")]

    monkeypatch.setattr(train_sft, "resolve_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(train_sft, "load_training_examples", lambda cfg: examples)
    monkeypatch.setattr(train_sft, "load_eval_examples", lambda cfg: examples)
    monkeypatch.setattr(train_sft, "experiment_log_path", lambda cfg: log_path)
    monkeypatch.setattr(train_sft, "experiment_plot_path", lambda cfg, stem: plot_path)
    monkeypatch.setattr(train_sft, "experiment_table_path", lambda cfg, stem: table_path)
    monkeypatch.setattr(train_sft, "save_pretrained_artifact", lambda *args, **kwargs: tmp_path / "checkpoint")
    monkeypatch.setattr(train_sft, "write_json", _write_json_file)
    monkeypatch.setattr(train_sft, "write_generation_artifacts", lambda cfg, stem, rows: {
        "json": str(sample_dir / f"{stem}.json"),
        "csv": str(sample_dir / f"{stem}.csv"),
    })
    monkeypatch.setattr(train_sft, "plot_metric_curves", lambda *args, **kwargs: plot_path)
    monkeypatch.setattr("alignlab.models.policy.load_policy_bundle", lambda spec: policy_bundle)
    monkeypatch.setattr(sys, "argv", ["train_sft", "--config", "unused"])

    train_sft.main()

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(record["event"] == "heldout_perplexity" for record in records)
    assert any(record["event"] == "sample_generation" for record in records)


def test_train_sft_early_stopping_saves_best_checkpoint(monkeypatch, tmp_path: Path) -> None:
    tokenizer = DummyTokenizer()
    policy_bundle = SimpleNamespace(
        model=DummyCausalLM(vocab_size=tokenizer.vocab_size + 8),
        tokenizer=tokenizer,
        spec=SimpleNamespace(),
        parameter_summary={"trainable_parameters": 1, "total_parameters": 1},
    )
    log_path = tmp_path / "sft_early_stop.jsonl"
    checkpoint_dir = tmp_path / "checkpoint_variants"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved_variants: list[str] = []
    perplexity_rows = iter(
        [
            {"heldout_loss": 2.0, "heldout_perplexity": 10.0, "num_eval_tokens": 8},
            {"heldout_loss": 2.1, "heldout_perplexity": 10.1, "num_eval_tokens": 8},
        ]
    )

    config = {
        "experiment_name": "pa2_sft_early_stop_test",
        "config_path": "unused",
        "method": {
            "name": "sft",
            "train_batch_size": 1,
            "learning_rate": 1.0e-3,
            "max_sequence_length": 16,
            "max_steps": 4,
        },
        "training": {"train_batch_size": 1, "eval_batch_size": 1, "learning_rate": 1.0e-3, "weight_decay": 0.0, "num_epochs": 1},
        "tokenization": {"max_sequence_length": 16, "max_prompt_length": 16},
        "generation": {"max_new_tokens": 1, "do_sample": False, "temperature": 1.0, "top_p": 1.0},
        "evaluation": {
            "num_eval_prompts": 1,
            "sample_table_size": 1,
            "sft_eval_every_steps": 1,
            "sft_sample_every_steps": 0,
            "early_stopping": {
                "enabled": True,
                "metric": "heldout_perplexity",
                "mode": "min",
                "min_delta": 0.0,
                "min_delta_mode": "absolute",
                "patience": 1,
                "min_steps": 1,
                "save_best": True,
            },
        },
        "max_grad_norm": 1.0,
        "dtype": "fp32",
        "gradient_accumulation_steps": 1,
        "model": {"hf_path": "dummy-policy", "family": "smollm", "dtype": "fp32"},
        "data": {"adapter": "hh_rlhf"},
        "reward_model": {"hf_path": str(tmp_path / "missing_reward"), "family": "llama"},
    }
    examples = [
        PreferenceExample(prompt="Human: hello Assistant:", chosen="good", rejected="bad"),
        PreferenceExample(prompt="Human: hello Assistant:", chosen="good", rejected="bad"),
    ]

    monkeypatch.setattr(train_sft, "resolve_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(train_sft, "load_training_examples", lambda cfg: examples)
    monkeypatch.setattr(train_sft, "load_eval_examples", lambda cfg: examples)
    monkeypatch.setattr(train_sft, "evaluate_sft_perplexity", lambda *args, **kwargs: next(perplexity_rows))
    monkeypatch.setattr(train_sft, "experiment_log_path", lambda cfg: log_path)
    monkeypatch.setattr(train_sft, "experiment_plot_path", lambda cfg, stem: tmp_path / "plot.png")
    monkeypatch.setattr(train_sft, "experiment_table_path", lambda cfg, stem: tmp_path / f"{stem}.json")
    monkeypatch.setattr(train_sft, "write_json", _write_json_file)
    monkeypatch.setattr(train_sft, "plot_metric_curves", lambda *args, **kwargs: tmp_path / "plot.png")
    monkeypatch.setattr(train_sft, "checkpoint_variant_dir", lambda cfg, artifact_name="final": checkpoint_dir / artifact_name)
    monkeypatch.setattr(
        train_sft,
        "save_pretrained_artifact",
        lambda *args, artifact_name="final", **kwargs: saved_variants.append(artifact_name) or (checkpoint_dir / artifact_name).mkdir(parents=True, exist_ok=True) or checkpoint_dir / artifact_name,
    )
    monkeypatch.setattr(
        train_sft,
        "promote_checkpoint_variant",
        lambda *args, **kwargs: (checkpoint_dir / "final").mkdir(parents=True, exist_ok=True) or checkpoint_dir / "final",
    )
    monkeypatch.setattr("alignlab.models.policy.load_policy_bundle", lambda spec: policy_bundle)
    monkeypatch.setattr(sys, "argv", ["train_sft", "--config", "unused"])

    train_sft.main()

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(record["event"] == "best_checkpoint" for record in records)
    assert any(record["event"] == "early_stopping" for record in records)
    assert "best" in saved_variants


def test_train_online_logs_extractor_precheck(monkeypatch, tmp_path: Path) -> None:
    tokenizer = DummyTokenizer()
    policy_bundle = SimpleNamespace(model=DummyCausalLM(vocab_size=tokenizer.vocab_size + 8), tokenizer=tokenizer, spec=SimpleNamespace())
    reference_bundle = SimpleNamespace(model=DummyCausalLM(vocab_size=tokenizer.vocab_size + 8), tokenizer=tokenizer, spec=SimpleNamespace())
    log_path = tmp_path / "online.jsonl"
    table_dir = tmp_path / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "policy").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ref").mkdir(parents=True, exist_ok=True)

    config = {
        "experiment_name": "pa2_rlvr_test",
        "config_path": "unused",
        "method": {
            "name": "rlvr",
            "rollout_batch_size": 1,
            "group_size": 1,
            "update_minibatch_size": 1,
            "epochs_per_rollout": 1,
            "learning_rate": 1.0e-3,
            "max_steps": 1,
            "gradient_accumulation_steps": 1,
            "clip_range": 0.2,
            "beta_kl": 0.01,
            "max_prompt_length": 8,
            "max_sequence_length": 16,
        },
        "training": {"train_batch_size": 1, "eval_batch_size": 1, "learning_rate": 1.0e-3, "weight_decay": 0.0, "max_steps": 1},
        "tokenization": {"max_prompt_length": 8, "max_sequence_length": 16},
        "generation": {"max_new_tokens": 1, "do_sample": False, "temperature": 1.0, "top_p": 1.0, "group_size": 1},
        "evaluation": {"num_eval_prompts": 1, "sample_table_size": 1, "extractor_precheck_limit": 1, "eval_every_steps": 0},
        "memory": {"cpu_rollout_cache": False},
        "max_grad_norm": 1.0,
        "dtype": "fp32",
        "model": {"hf_path": str(tmp_path / "policy"), "tokenizer_path": str(tmp_path / "policy"), "family": "smollm", "dtype": "fp32"},
        "data": {"adapter": "gsm8k"},
        "reference_model": {"hf_path": str(tmp_path / "ref"), "tokenizer_path": str(tmp_path / "ref"), "family": "smollm"},
    }
    examples = [VerifiableExample(prompt="math ?", gold_answer="42", meta={"answer_text": "#### 42"})]

    monkeypatch.setattr(train_online, "resolve_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(train_online, "load_training_examples", lambda cfg: examples)
    monkeypatch.setattr(train_online, "load_eval_examples", lambda cfg: examples)
    monkeypatch.setattr(train_online, "experiment_log_path", lambda cfg: log_path)
    monkeypatch.setattr(train_online, "experiment_plot_path", lambda cfg, stem: tmp_path / "plot.png")
    monkeypatch.setattr(train_online, "experiment_table_path", lambda cfg, stem: table_dir / f"{stem}.json")
    monkeypatch.setattr(train_online, "save_pretrained_artifact", lambda *args, **kwargs: tmp_path / "checkpoint")
    monkeypatch.setattr(train_online, "plot_metric_curves", lambda *args, **kwargs: tmp_path / "plot.png")
    monkeypatch.setattr(train_online, "write_json", _write_json_file)
    monkeypatch.setattr(train_online, "evaluate_rlvr_policy", lambda *args, **kwargs: {
        "pass_at_1": 0.0,
        "format_compliance_rate": 0.0,
        "mean_response_length": 0.0,
        "kl_from_reference": 0.0,
        "num_eval_prompts": 1,
        "artifacts": {},
    })
    monkeypatch.setattr("alignlab.models.policy.load_policy_bundle", lambda spec: policy_bundle)
    monkeypatch.setattr("alignlab.models.reference.build_reference_bundle", lambda spec=None, policy_model=None, tokenizer=None: reference_bundle)
    monkeypatch.setattr(sys, "argv", ["train_online", "--config", "unused"])

    train_online.main()

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(record["event"] == "extractor_precheck" for record in records)


def test_pa2_configs_reuse_archived_rm_and_subset_caps() -> None:
    sft_config = load_experiment_config("configs/experiment/pa2_sft_hh_rlhf.yaml")
    dpo_config = load_experiment_config("configs/experiment/pa2_dpo_hh_rlhf.yaml")
    ppo_config = load_experiment_config("configs/experiment/pa2_ppo_hh_rlhf.yaml")
    grpo_config = load_experiment_config("configs/experiment/pa2_grpo_hh_rlhf.yaml")
    rlvr_config = load_experiment_config("configs/experiment/pa2_rlvr_gsm8k.yaml")

    assert sft_config["reward_model"]["hf_path"] == "artifacts/checkpoints/rm_hh_rlhf/final"
    assert sft_config["data"]["sample_limit"] == 20000
    assert dpo_config["data"]["sample_limit"] == 20000
    assert ppo_config["data"]["sample_limit"] == 20000
    assert grpo_config["data"]["sample_limit"] == 20000
    assert rlvr_config["data"]["sample_limit"] == 2000
    assert sft_config["evaluation"]["early_stopping"]["metric"] == "heldout_perplexity"
    assert dpo_config["evaluation"]["early_stopping"]["metric"] == "preference_accuracy"
    assert ppo_config["evaluation"]["early_stopping"]["metric"] == "rm_win_rate_vs_sft"
    assert grpo_config["evaluation"]["early_stopping"]["metric"] == "rm_win_rate_vs_sft"
    assert rlvr_config["evaluation"]["early_stopping"]["metric"] == "pass_at_1"
