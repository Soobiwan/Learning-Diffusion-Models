"""CLI for PPO/GRPO/RLVR experiments."""

from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

from ._shared import (
    build_argument_parser,
    build_online_objective,
    build_prompt_collator,
    evaluation_every_steps,
    load_training_examples,
    load_eval_examples,
    make_dataloader,
    model_spec_from_config,
    preference_examples,
    require_checkpoint,
    resolve_config,
    summarize_config,
    verifiable_examples,
)
from ..common.checkpointing import save_pretrained_artifact
from ..eval.pipeline import evaluate_hh_policy, evaluate_rlvr_policy
from ..eval.pa2_tools import verify_gsm8k_answer_extractor
from ..eval.reports import (
    ResourceTracker,
    append_jsonl,
    experiment_log_path,
    experiment_plot_path,
    experiment_table_path,
    plot_metric_curves,
    write_json,
)


def main() -> None:
    parser = build_argument_parser("Train or dry-run an online RL alignment experiment.")
    args = parser.parse_args()
    config = resolve_config(args.config, sample_limit=args.sample_limit, max_steps=args.max_steps)
    tqdm.write(summarize_config(config))
    if args.dry_run:
        return
    log_path = experiment_log_path(config)

    from ..models.policy import load_policy_bundle
    from ..models.reference import build_reference_bundle
    from ..models.reward import load_reward_bundle
    from ..models.value import load_value_bundle
    from ..rollout.rewards import LearnedRewardFunction, VerifiableRewardFunction
    from ..rollout.verifiers import GSM8KAnswerVerifier
    from ..trainers.online_rl_trainer import OnlineRLTrainer

    policy_spec = model_spec_from_config(config)
    require_checkpoint(policy_spec.hf_path, f"Policy checkpoint '{policy_spec.hf_path}' was not found. Run train_sft first.")
    policy_bundle = load_policy_bundle(policy_spec)
    reference_bundle = None
    if "reference_model" in config:
        reference_spec = model_spec_from_config(config, "reference_model")
        require_checkpoint(
            reference_spec.hf_path,
            f"Reference checkpoint '{reference_spec.hf_path}' was not found. Run train_sft first.",
        )
        reference_bundle = build_reference_bundle(spec=reference_spec)

    value_bundle = None
    if config["method"]["name"].lower() == "ppo":
        value_bundle = load_value_bundle(
            model_spec_from_config(config, "value_model") if "value_model" in config else model_spec_from_config(config)
        )

    if config["method"]["name"].lower() == "rlvr":
        reward_function = VerifiableRewardFunction(GSM8KAnswerVerifier())
    else:
        if "reward_model" not in config:
            raise ValueError("Online RL with learned rewards requires a `reward_model` config section.")
        reward_spec = model_spec_from_config(config, "reward_model")
        reward_path = Path(reward_spec.hf_path)
        require_checkpoint(
            str(reward_path),
            f"Reward-model checkpoint '{reward_spec.hf_path}' was not found. Run train_rm first.",
        )
        reward_bundle = load_reward_bundle(reward_spec, freeze=True)
        reward_function = LearnedRewardFunction(
            model=reward_bundle.model,
            tokenizer=reward_bundle.tokenizer,
            max_length=int(config["method"].get("max_sequence_length", config["tokenization"]["max_sequence_length"])),
        )

    examples = load_training_examples(config)
    eval_examples = load_eval_examples(config)
    dataloader = make_dataloader(
        examples=examples,
        collator=build_prompt_collator(policy_bundle.tokenizer, config),
        batch_size=int(config["method"].get("rollout_batch_size", config["training"]["train_batch_size"])),
    )
    trainer = OnlineRLTrainer(
        model=policy_bundle.model,
        objective=build_online_objective(config["method"]["name"], config["method"]),
        reward_function=reward_function,
        tokenizer=policy_bundle.tokenizer,
        reference_model=reference_bundle.model if reference_bundle is not None else None,
        value_model=value_bundle.model if value_bundle is not None else None,
        generation_config=config["generation"],
        gamma=float(config["method"].get("gamma", 0.99)),
        gae_lambda=float(config["method"].get("gae_lambda", 0.95)),
        kl_coef=float(config["method"].get("kl_coef", config["method"].get("beta_kl", 0.02))),
        group_size=int(config["method"].get("group_size", config["generation"].get("group_size", 1))),
        update_minibatch_size=int(config["method"].get("update_minibatch_size", 1)),
        epochs_per_rollout=int(config["method"].get("epochs_per_rollout", 1)),
        cpu_rollout_cache=bool(config["memory"].get("cpu_rollout_cache", True)),
        learning_rate=float(config["method"].get("learning_rate", config["training"]["learning_rate"])),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        mixed_precision=str(config.get("dtype", config["model"].get("dtype", "no"))),
        gradient_accumulation_steps=int(config["method"].get("gradient_accumulation_steps", config.get("gradient_accumulation_steps", 1))),
    )
    max_steps = int(config["method"].get("max_steps", config["training"]["max_steps"]))
    eval_every = evaluation_every_steps(config)
    tracker = ResourceTracker()
    train_rows: list[dict[str, float | int | str]] = []
    verifier = GSM8KAnswerVerifier()

    if config["method"]["name"].lower() == "rlvr":
        extractor_summary = verify_gsm8k_answer_extractor(
            verifiable_examples(examples),
            verifier=verifier,
            gold_limit=int(config.get("evaluation", {}).get("extractor_precheck_limit", 20)),
            wrong_limit=int(config.get("evaluation", {}).get("extractor_precheck_limit", 20)),
        )
        write_json(experiment_table_path(config, "extractor_precheck"), extractor_summary)
        append_jsonl(log_path, {"event": "extractor_precheck", **extractor_summary})
        tqdm.write(f"extractor_precheck={extractor_summary}")

    total_rollouts = min(len(dataloader), max_steps)
    with tqdm(
        total=total_rollouts,
        desc=f"{config['method']['name']} rollouts",
        dynamic_ncols=True,
    ) as progress:
        for step, batch in enumerate(dataloader, start=1):
            with tracker.measure_step():
                metrics = trainer.train_batch(batch)
            record: dict[str, float | int | str] = {"event": "train", "step": step, **metrics}
            append_jsonl(log_path, record)
            train_rows.append(record)
            progress.update(1)
            progress.set_postfix(
                step=step,
                loss=f"{metrics['loss']:.4f}",
                mean_reward=f"{metrics.get('mean_reward', 0.0):.3f}",
                kl=f"{metrics.get('rollout_mean_kl', 0.0):.4f}",
            )
            tqdm.write(f"step={step} metrics={metrics}")
            if eval_every > 0 and step % eval_every == 0:
                tqdm.write(f"running periodic evaluation at step {step}...")
                if config["method"]["name"].lower() == "rlvr":
                    evaluation_summary = evaluate_rlvr_policy(
                        config,
                        candidate_model=policy_bundle.model,
                        candidate_tokenizer=policy_bundle.tokenizer,
                        reference_model=reference_bundle.model if reference_bundle is not None else policy_bundle.model,
                        examples=verifiable_examples(eval_examples),
                        verifier=verifier,
                        stem=f"rlvr_eval_step_{step:05d}",
                    )
                else:
                    evaluation_summary = evaluate_hh_policy(
                        config,
                        candidate_model=policy_bundle.model,
                        candidate_tokenizer=policy_bundle.tokenizer,
                        reference_model=reference_bundle.model if reference_bundle is not None else policy_bundle.model,
                        reward_function=reward_function,
                        prompt_examples=preference_examples(eval_examples),
                        pair_examples=None,
                        baseline_model=reference_bundle.model if reference_bundle is not None else policy_bundle.model,
                        baseline_tokenizer=reference_bundle.tokenizer if reference_bundle is not None else policy_bundle.tokenizer,
                        stem=f"{config['method']['name']}_eval_step_{step:05d}",
                    )
                append_jsonl(log_path, {"event": "evaluation", "step": step, **evaluation_summary})
                tqdm.write(f"eval_step={step} metrics={evaluation_summary}")
            if step >= max_steps:
                break
    tqdm.write("saving checkpoint...")
    save_dir = save_pretrained_artifact(
        policy_bundle.model,
        policy_bundle.tokenizer,
        config,
        extra_metadata={"task": config["method"]["name"]},
    )
    tqdm.write(f"saved_checkpoint={save_dir}")

    tqdm.write("running final evaluation...")
    if config["method"]["name"].lower() == "rlvr":
        final_evaluation = evaluate_rlvr_policy(
            config,
            candidate_model=policy_bundle.model,
            candidate_tokenizer=policy_bundle.tokenizer,
            reference_model=reference_bundle.model if reference_bundle is not None else policy_bundle.model,
            examples=verifiable_examples(eval_examples),
            verifier=verifier,
            stem="rlvr_final_eval",
        )
    else:
        final_evaluation = evaluate_hh_policy(
            config,
            candidate_model=policy_bundle.model,
            candidate_tokenizer=policy_bundle.tokenizer,
            reference_model=reference_bundle.model if reference_bundle is not None else policy_bundle.model,
            reward_function=reward_function,
            prompt_examples=preference_examples(eval_examples),
            pair_examples=None,
            baseline_model=reference_bundle.model if reference_bundle is not None else policy_bundle.model,
            baseline_tokenizer=reference_bundle.tokenizer if reference_bundle is not None else policy_bundle.tokenizer,
            stem=f"{config['method']['name']}_final_eval",
        )
    append_jsonl(log_path, {"event": "evaluation", "stage": "final", **final_evaluation})
    tqdm.write(f"final_eval={final_evaluation}")
    if train_rows:
        plot_metric_curves(
            experiment_plot_path(config, "training_curve"),
            train_rows,
            keys=[
                key
                for key in ("loss", "mean_reward", "rollout_mean_kl", "degenerate_fraction", "format_compliance_rate")
                if key in train_rows[0]
            ],
            title=f"{config['method']['name'].upper()} Training Metrics",
        )
    resource_summary = tracker.summary()
    resource_summary["final_evaluation"] = final_evaluation
    write_json(experiment_table_path(config, "resource_summary"), resource_summary)


if __name__ == "__main__":
    main()
