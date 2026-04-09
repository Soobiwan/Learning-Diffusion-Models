"""CLI for supervised fine-tuning."""

from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

from ._shared import (
    build_argument_parser,
    build_prompt_collator,
    build_sft_collator,
    configured_gradient_accumulation_steps,
    configured_max_steps,
    configured_num_epochs,
    load_eval_examples,
    load_training_examples,
    make_dataloader,
    model_spec_from_config,
    preference_examples,
    resolve_config,
    sft_examples,
    summarize_config,
)
from ..common.checkpointing import save_pretrained_artifact
from ..eval.pipeline import evaluate_hh_policy, evaluate_sft_perplexity
from ..eval.reports import (
    ResourceTracker,
    append_jsonl,
    experiment_log_path,
    experiment_plot_path,
    experiment_table_path,
    plot_metric_curves,
    write_generation_artifacts,
    write_json,
)


def main() -> None:
    parser = build_argument_parser("Train or dry-run an SFT experiment.")
    args = parser.parse_args()
    config = resolve_config(args.config, sample_limit=args.sample_limit, max_steps=args.max_steps)
    tqdm.write(summarize_config(config))
    if args.dry_run:
        return

    from ..models.policy import load_policy_bundle
    from ..models.reward import load_reward_bundle
    from ..rollout.rewards import LearnedRewardFunction
    from ..trainers.sft_trainer import SFTTrainer

    bundle = load_policy_bundle(model_spec_from_config(config))
    examples = sft_examples(load_training_examples(config))
    eval_pref_examples = preference_examples(load_eval_examples(config))
    eval_limit = int(config.get("evaluation", {}).get("num_eval_prompts", 200))
    heldout_sft_examples = sft_examples(eval_pref_examples[:eval_limit])
    sample_prompt_examples = eval_pref_examples[: int(config.get("evaluation", {}).get("sample_table_size", 5))]
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
        gradient_accumulation_steps=configured_gradient_accumulation_steps(config),
    )
    max_steps = configured_max_steps(config)
    num_epochs = configured_num_epochs(config)
    sft_eval_every = int(config.get("evaluation", {}).get("sft_eval_every_steps", 100))
    sft_sample_every = int(config.get("evaluation", {}).get("sft_sample_every_steps", sft_eval_every))
    log_path = experiment_log_path(config)
    tracker = ResourceTracker()
    train_rows: list[dict[str, float | int | str]] = []
    last_metrics: dict[str, float] | None = None
    prompt_collator = build_prompt_collator(bundle.tokenizer, config)
    greedy_generation_config = dict(config["generation"])
    greedy_generation_config["do_sample"] = False
    greedy_generation_config["temperature"] = 1.0
    greedy_generation_config["top_p"] = 1.0

    stop_training = False
    for epoch in range(1, num_epochs + 1):
        with tqdm(
            total=len(dataloader),
            desc=f"sft epoch {epoch}/{num_epochs}",
            dynamic_ncols=True,
        ) as progress:
            for batch in dataloader:
                with tracker.measure_step():
                    metrics = trainer.train_batch(batch)
                last_metrics = metrics
                postfix: dict[str, str | int] = {
                    "optimizer_step": trainer.step,
                    "loss": f"{metrics['loss']:.4f}",
                    "perplexity": f"{metrics['perplexity']:.2f}",
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
                if sft_eval_every > 0 and trainer.step % sft_eval_every == 0:
                    perplexity_summary = evaluate_sft_perplexity(
                        model=bundle.model,
                        tokenizer=bundle.tokenizer,
                        examples=heldout_sft_examples,
                        batch_size=int(config["training"].get("eval_batch_size", 1)),
                        max_length=int(config["method"].get("max_sequence_length", config["tokenization"]["max_sequence_length"])),
                    )
                    append_jsonl(log_path, {"event": "heldout_perplexity", "step": trainer.step, **perplexity_summary})
                    tqdm.write(f"heldout_step={trainer.step} metrics={perplexity_summary}")
                if sft_sample_every > 0 and trainer.step % sft_sample_every == 0 and sample_prompt_examples:
                    sample_batch = prompt_collator(sample_prompt_examples)
                    responses = trainer.sample_generations(
                        tokenizer=bundle.tokenizer,
                        prompt_batch=sample_batch,
                        generation_config=greedy_generation_config,
                    )
                    sample_rows = [
                        {"prompt": example.prompt, "response": response}
                        for example, response in zip(sample_prompt_examples, responses)
                    ]
                    sample_paths = write_generation_artifacts(
                        config,
                        f"sft_samples_step_{trainer.step:05d}",
                        sample_rows,
                    )
                    append_jsonl(log_path, {"event": "sample_generation", "step": trainer.step, "artifacts": sample_paths})
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
    save_dir = save_pretrained_artifact(bundle.model, bundle.tokenizer, config, extra_metadata={"task": "sft"})
    tqdm.write(f"saved_checkpoint={save_dir}")

    reward_summary = None
    if "reward_model" in config:
        reward_spec = model_spec_from_config(config, "reward_model")
        reward_path = Path(reward_spec.hf_path)
        if reward_path.exists():
            tqdm.write("running final reward-model evaluation...")
            reward_bundle = load_reward_bundle(reward_spec, freeze=True)
            reward_function = LearnedRewardFunction(
                model=reward_bundle.model,
                tokenizer=reward_bundle.tokenizer,
                max_length=int(config["method"].get("max_sequence_length", config["tokenization"]["max_sequence_length"])),
            )
            reward_summary = evaluate_hh_policy(
                config,
                candidate_model=bundle.model,
                candidate_tokenizer=bundle.tokenizer,
                reference_model=bundle.model,
                reward_function=reward_function,
                prompt_examples=eval_pref_examples,
                pair_examples=None,
                baseline_model=bundle.model,
                baseline_tokenizer=bundle.tokenizer,
                stem="sft_final_eval",
            )
            append_jsonl(log_path, {"event": "evaluation", "stage": "final", **reward_summary})
            tqdm.write(f"final_eval={reward_summary}")

    if train_rows:
        plot_metric_curves(
            experiment_plot_path(config, "training_curve"),
            train_rows,
            keys=[key for key in ("loss", "perplexity") if key in train_rows[0]],
            title="SFT Training Metrics",
        )
    resource_summary = tracker.summary()
    if reward_summary is not None:
        resource_summary["final_evaluation"] = reward_summary
    write_json(experiment_table_path(config, "resource_summary"), resource_summary)


if __name__ == "__main__":
    main()
