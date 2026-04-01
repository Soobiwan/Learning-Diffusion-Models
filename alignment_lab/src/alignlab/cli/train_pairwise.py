"""CLI for pairwise preference optimization."""

from __future__ import annotations

from tqdm.auto import tqdm

from ._shared import (
    build_argument_parser,
    build_pairwise_collator,
    build_pairwise_objective,
    configured_gradient_accumulation_steps,
    configured_max_steps,
    configured_num_epochs,
    evaluation_every_steps,
    load_eval_examples,
    load_training_examples,
    make_dataloader,
    model_spec_from_config,
    preference_examples,
    require_checkpoint,
    resolve_config,
    summarize_config,
)
from ..common.checkpointing import save_pretrained_artifact
from ..eval.pipeline import evaluate_hh_policy
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
    parser = build_argument_parser("Train or dry-run a pairwise alignment experiment.")
    args = parser.parse_args()
    config = resolve_config(args.config, sample_limit=args.sample_limit, max_steps=args.max_steps)
    tqdm.write(summarize_config(config))
    if args.dry_run:
        return

    from ..models.policy import load_policy_bundle
    from ..models.reference import build_reference_bundle
    from ..models.reward import load_reward_bundle
    from ..rollout.rewards import LearnedRewardFunction
    from ..trainers.pairwise_trainer import PairwiseTrainer

    policy_spec = model_spec_from_config(config)
    require_checkpoint(policy_spec.hf_path, f"Policy checkpoint '{policy_spec.hf_path}' was not found. Run train_sft first.")
    reference_spec = model_spec_from_config(config, "reference_model") if "reference_model" in config else policy_spec
    require_checkpoint(
        reference_spec.hf_path,
        f"Reference checkpoint '{reference_spec.hf_path}' was not found. Run train_sft first.",
    )
    if "reward_model" not in config:
        raise ValueError("DPO training requires a `reward_model` section for PA2 evaluation.")
    reward_spec = model_spec_from_config(config, "reward_model")
    require_checkpoint(
        reward_spec.hf_path,
        f"Reward-model checkpoint '{reward_spec.hf_path}' was not found. Run train_rm first.",
    )

    policy_bundle = load_policy_bundle(model_spec_from_config(config))
    reference_bundle = build_reference_bundle(
        spec=model_spec_from_config(config, "reference_model")
        if "reference_model" in config
        else policy_bundle.spec,
        policy_model=None if "reference_model" in config else policy_bundle.model,
        tokenizer=None if "reference_model" in config else policy_bundle.tokenizer,
    )
    examples = preference_examples(load_training_examples(config))
    dataloader = make_dataloader(
        examples=examples,
        collator=build_pairwise_collator(policy_bundle.tokenizer, config),
        batch_size=int(config["method"].get("train_batch_size", config["training"]["train_batch_size"])),
    )
    reward_bundle = load_reward_bundle(reward_spec, freeze=True)
    reward_function = LearnedRewardFunction(
        model=reward_bundle.model,
        tokenizer=reward_bundle.tokenizer,
        max_length=int(config["method"].get("max_sequence_length", config["tokenization"]["max_sequence_length"])),
    )
    eval_examples = preference_examples(load_eval_examples(config))
    trainer = PairwiseTrainer(
        model=policy_bundle.model,
        objective=build_pairwise_objective(config["method"]["name"], config["method"]),
        reference_model=reference_bundle.model,
        learning_rate=float(config["method"].get("learning_rate", config["training"]["learning_rate"])),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        mixed_precision=str(config.get("dtype", config["model"].get("dtype", "no"))),
        gradient_accumulation_steps=configured_gradient_accumulation_steps(config),
    )
    max_steps = configured_max_steps(config)
    num_epochs = configured_num_epochs(config)
    eval_every = evaluation_every_steps(config)
    log_path = experiment_log_path(config)
    tracker = ResourceTracker()
    train_rows: list[dict[str, float | int | str]] = []
    last_metrics: dict[str, float] | None = None

    stop_training = False
    for epoch in range(1, num_epochs + 1):
        with tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"{config['method']['name']} epoch {epoch}/{num_epochs}",
            dynamic_ncols=True,
        ) as progress:
            for batch in progress:
                with tracker.measure_step():
                    metrics = trainer.train_batch(batch)
                last_metrics = metrics
                progress.set_postfix(
                    optimizer_step=trainer.step,
                    loss=f"{metrics['loss']:.4f}",
                    preference_accuracy=f"{metrics['preference_accuracy']:.3f}",
                )
                if not trainer.last_step_was_optimizer_step:
                    continue
                record: dict[str, float | int | str] = {"event": "train", "epoch": epoch, "step": trainer.step, **metrics}
                append_jsonl(log_path, record)
                train_rows.append(record)
                tqdm.write(f"step={trainer.step} metrics={metrics}")
                if eval_every > 0 and trainer.step % eval_every == 0:
                    tqdm.write(f"running periodic evaluation at step {trainer.step}...")
                    evaluation_summary = evaluate_hh_policy(
                        config,
                        candidate_model=policy_bundle.model,
                        candidate_tokenizer=policy_bundle.tokenizer,
                        reference_model=reference_bundle.model,
                        reward_function=reward_function,
                        prompt_examples=eval_examples,
                        pair_examples=eval_examples,
                        baseline_model=reference_bundle.model,
                        baseline_tokenizer=reference_bundle.tokenizer,
                        stem=f"dpo_eval_step_{trainer.step:05d}",
                    )
                    append_jsonl(log_path, {"event": "evaluation", "step": trainer.step, **evaluation_summary})
                    tqdm.write(f"eval_step={trainer.step} metrics={evaluation_summary}")
                if max_steps is not None and trainer.step >= max_steps:
                    stop_training = True
                    break
        if stop_training:
            break

    if trainer.flush() and last_metrics is not None:
        record = {"event": "train", "epoch": num_epochs, "step": trainer.step, **last_metrics}
        append_jsonl(log_path, record)
        train_rows.append(record)
        tqdm.write(f"step={trainer.step} metrics={last_metrics}")
    tqdm.write("saving checkpoint...")
    save_dir = save_pretrained_artifact(
        policy_bundle.model,
        policy_bundle.tokenizer,
        config,
        extra_metadata={"task": config["method"]["name"]},
    )
    tqdm.write(f"saved_checkpoint={save_dir}")
    tqdm.write("running final evaluation...")
    evaluation_summary = evaluate_hh_policy(
        config,
        candidate_model=policy_bundle.model,
        candidate_tokenizer=policy_bundle.tokenizer,
        reference_model=reference_bundle.model,
        reward_function=reward_function,
        prompt_examples=eval_examples,
        pair_examples=eval_examples,
        baseline_model=reference_bundle.model,
        baseline_tokenizer=reference_bundle.tokenizer,
        stem="dpo_final_eval",
    )
    append_jsonl(log_path, {"event": "evaluation", "stage": "final", **evaluation_summary})
    tqdm.write(f"final_eval={evaluation_summary}")
    if train_rows:
        plot_metric_curves(
            experiment_plot_path(config, "training_curve"),
            train_rows,
            keys=[key for key in ("loss", "preference_accuracy", "z_margin") if key in train_rows[0]],
            title="DPO Training Metrics",
        )
    resource_summary = tracker.summary()
    resource_summary["final_evaluation"] = evaluation_summary
    write_json(experiment_table_path(config, "resource_summary"), resource_summary)


if __name__ == "__main__":
    main()
