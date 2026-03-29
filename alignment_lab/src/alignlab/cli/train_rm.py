"""CLI for reward model training."""

from __future__ import annotations

from ._shared import (
    build_argument_parser,
    build_rm_collator,
    load_training_examples,
    make_dataloader,
    model_spec_from_config,
    preference_examples,
    resolve_config,
    summarize_config,
)
from ..common.checkpointing import save_pretrained_artifact


def main() -> None:
    parser = build_argument_parser("Train or dry-run a reward model experiment.")
    args = parser.parse_args()
    config = resolve_config(args.config, sample_limit=args.sample_limit, max_steps=args.max_steps)
    print(summarize_config(config))
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
    )
    max_steps = int(config["method"].get("max_steps", config["training"]["max_steps"]))
    for step, batch in enumerate(dataloader, start=1):
        metrics = trainer.train_batch(batch)
        print(f"step={step} metrics={metrics}")
        if step >= max_steps:
            break
    save_dir = save_pretrained_artifact(bundle.model, bundle.tokenizer, config, extra_metadata={"task": "reward_model"})
    print(f"saved_checkpoint={save_dir}")


if __name__ == "__main__":
    main()
