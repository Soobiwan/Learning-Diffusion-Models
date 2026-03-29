"""CLI for supervised fine-tuning."""

from __future__ import annotations

from ._shared import (
    build_argument_parser,
    build_sft_collator,
    load_training_examples,
    make_dataloader,
    model_spec_from_config,
    resolve_config,
    sft_examples,
    summarize_config,
)
from ..common.checkpointing import save_pretrained_artifact


def main() -> None:
    parser = build_argument_parser("Train or dry-run an SFT experiment.")
    args = parser.parse_args()
    config = resolve_config(args.config, sample_limit=args.sample_limit, max_steps=args.max_steps)
    print(summarize_config(config))
    if args.dry_run:
        return

    from ..models.policy import load_policy_bundle
    from ..trainers.sft_trainer import SFTTrainer

    bundle = load_policy_bundle(model_spec_from_config(config))
    examples = sft_examples(load_training_examples(config))
    dataloader = make_dataloader(
        examples=examples,
        collator=build_sft_collator(bundle.tokenizer, config),
        batch_size=int(config["method"].get("train_batch_size", config["training"]["train_batch_size"])),
    )
    trainer = SFTTrainer(
        model=bundle.model,
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
    save_dir = save_pretrained_artifact(bundle.model, bundle.tokenizer, config, extra_metadata={"task": "sft"})
    print(f"saved_checkpoint={save_dir}")


if __name__ == "__main__":
    main()
