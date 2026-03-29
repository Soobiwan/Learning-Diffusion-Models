"""CLI for PPO/GRPO/RLVR experiments."""

from __future__ import annotations

from pathlib import Path

from ._shared import (
    build_argument_parser,
    build_online_objective,
    build_prompt_collator,
    load_training_examples,
    make_dataloader,
    model_spec_from_config,
    resolve_config,
    summarize_config,
)


def main() -> None:
    parser = build_argument_parser("Train or dry-run an online RL alignment experiment.")
    args = parser.parse_args()
    config = resolve_config(args.config, sample_limit=args.sample_limit, max_steps=args.max_steps)
    print(summarize_config(config))
    if args.dry_run:
        return

    from ..models.policy import load_policy_bundle
    from ..models.reference import build_reference_bundle
    from ..models.reward import load_reward_bundle
    from ..models.value import load_value_bundle
    from ..rollout.rewards import LearnedRewardFunction, VerifiableRewardFunction
    from ..rollout.verifiers import GSM8KAnswerVerifier
    from ..trainers.online_rl_trainer import OnlineRLTrainer

    policy_bundle = load_policy_bundle(model_spec_from_config(config))
    reference_bundle = None
    if "reference_model" in config:
        reference_bundle = build_reference_bundle(spec=model_spec_from_config(config, "reference_model"))

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
        if not reward_path.is_absolute() and reward_path.parts and reward_path.parts[0] == "artifacts" and not reward_path.exists():
            raise FileNotFoundError(
                f"Reward-model checkpoint '{reward_spec.hf_path}' was not found. Run train_rm first."
            )
        reward_bundle = load_reward_bundle(reward_spec, freeze=True)
        reward_function = LearnedRewardFunction(
            model=reward_bundle.model,
            tokenizer=reward_bundle.tokenizer,
            max_length=int(config["method"].get("max_sequence_length", config["tokenization"]["max_sequence_length"])),
        )

    examples = load_training_examples(config)
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
    )
    max_steps = int(config["method"].get("max_steps", config["training"]["max_steps"]))
    for step, batch in enumerate(dataloader, start=1):
        metrics = trainer.train_batch(batch)
        print(f"step={step} metrics={metrics}")
        if step >= max_steps:
            break


if __name__ == "__main__":
    main()
