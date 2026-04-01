"""CLI for reward model training."""

from __future__ import annotations

from tqdm.auto import tqdm

from ._shared import (
    build_argument_parser,
    build_rm_collator,
    configured_gradient_accumulation_steps,
    configured_max_steps,
    configured_num_epochs,
    load_eval_examples,
    load_training_examples,
    make_dataloader,
    model_spec_from_config,
    preference_examples,
    resolve_config,
    summarize_config,
)
from ..common.checkpointing import save_pretrained_artifact
from ..eval.pipeline import evaluate_reward_model
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
    parser = build_argument_parser("Train or dry-run a reward model experiment.")
    args = parser.parse_args()
    config = resolve_config(args.config, sample_limit=args.sample_limit, max_steps=args.max_steps)
    tqdm.write(summarize_config(config))
    if args.dry_run:
        return

    from ..models.reward import load_reward_bundle
    from ..trainers.rm_trainer import RewardModelTrainer

    bundle = load_reward_bundle(model_spec_from_config(config))
    examples = preference_examples(load_training_examples(config))
    dataloader = make_dataloader(
        examples=examples,
        collator=build_rm_collator(bundle.tokenizer, config),
        batch_size=int(config["method"].get("train_batch_size", config["training"]["train_batch_size"])),
    )
    trainer = RewardModelTrainer(
        model=bundle.model,
        beta=float(config["method"].get("ranking_beta", 1.0)),
        regularization=float(config["method"].get("reward_regularization", 0.0)),
        learning_rate=float(config["method"].get("learning_rate", config["training"]["learning_rate"])),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        mixed_precision=str(config.get("dtype", config["model"].get("dtype", "no"))),
        gradient_accumulation_steps=configured_gradient_accumulation_steps(config),
    )
    max_steps = configured_max_steps(config)
    num_epochs = configured_num_epochs(config)
    eval_examples = preference_examples(load_eval_examples(config))
    log_path = experiment_log_path(config)
    tracker = ResourceTracker()
    train_rows: list[dict[str, float | int | str]] = []
    last_metrics: dict[str, float] | None = None

    stop_training = False
    for epoch in range(1, num_epochs + 1):
        with tqdm(
            total=len(dataloader),
            desc=f"rm epoch {epoch}/{num_epochs}",
            dynamic_ncols=True,
        ) as progress:
            for batch in dataloader:
                with tracker.measure_step():
                    metrics = trainer.train_batch(batch)
                last_metrics = metrics
                postfix: dict[str, str | int] = {
                    "optimizer_step": trainer.step,
                    "loss": f"{metrics['loss']:.4f}",
                    "reward_accuracy": f"{metrics['reward_accuracy']:.3f}",
                }
                if trainer.gradient_accumulation_steps > 1:
                    postfix["accum"] = trainer.accumulation_status
                progress.update(1)
                progress.set_postfix(postfix)
                if not trainer.last_step_was_optimizer_step:
                    continue
                record: dict[str, float | int | str] = {"event": "train", "epoch": epoch, "step": trainer.step, **metrics}
                append_jsonl(log_path, record)
                train_rows.append(record)
                tqdm.write(f"step={trainer.step} metrics={metrics}")
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
    save_dir = save_pretrained_artifact(bundle.model, bundle.tokenizer, config, extra_metadata={"task": "reward_model"})
    tqdm.write(f"saved_checkpoint={save_dir}")
    tqdm.write("running final reward-model evaluation...")
    evaluation_summary = evaluate_reward_model(
        config,
        reward_model=bundle.model,
        tokenizer=bundle.tokenizer,
        examples=eval_examples,
        batch_size=int(config["training"].get("eval_batch_size", 1)),
        max_length=int(config["method"].get("max_sequence_length", config["tokenization"]["max_sequence_length"])),
        stem="rm_final_eval",
    )
    append_jsonl(log_path, {"event": "evaluation", "stage": "final", **evaluation_summary})
    tqdm.write(f"final_eval={evaluation_summary}")
    if train_rows:
        plot_metric_curves(
            experiment_plot_path(config, "training_curve"),
            train_rows,
            keys=[key for key in ("loss", "reward_accuracy", "reward_margin") if key in train_rows[0]],
            title="Reward Model Training Metrics",
        )
    resource_summary = tracker.summary()
    resource_summary["final_evaluation"] = evaluation_summary
    write_json(experiment_table_path(config, "resource_summary"), resource_summary)


if __name__ == "__main__":
    main()
