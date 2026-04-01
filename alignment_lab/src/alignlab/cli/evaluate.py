"""CLI entrypoint for evaluation runs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ._shared import (
    build_argument_parser,
    load_eval_examples,
    model_spec_from_config,
    preference_examples,
    require_checkpoint,
    resolve_config,
    summarize_config,
    verifiable_examples,
)
from ..common.checkpointing import final_checkpoint_dir
from ..eval.pipeline import evaluate_hh_policy, evaluate_reward_model, evaluate_rlvr_policy


def _checkpoint_spec(config: dict[str, object], key: str, checkpoint_dir: Path):
    spec = model_spec_from_config(config, key)
    return replace(spec, hf_path=str(checkpoint_dir), tokenizer_path=str(checkpoint_dir))


def main() -> None:
    parser = build_argument_parser("Run evaluation for a trained PA2 experiment.")
    args = parser.parse_args()
    config = resolve_config(args.config, sample_limit=args.sample_limit, max_steps=args.max_steps)
    print(summarize_config(config))
    if args.dry_run:
        return

    method_name = str(config["method"]["name"]).lower()
    checkpoint_dir = final_checkpoint_dir(config)
    require_checkpoint(
        str(checkpoint_dir),
        f"Trained checkpoint '{checkpoint_dir}' was not found. Run the training command for this experiment first.",
    )

    if method_name == "reward_model":
        from ..models.reward import load_reward_bundle

        bundle = load_reward_bundle(_checkpoint_spec(config, "model", checkpoint_dir), freeze=True)
        examples = preference_examples(load_eval_examples(config))
        summary = evaluate_reward_model(
            config,
            reward_model=bundle.model,
            tokenizer=bundle.tokenizer,
            examples=examples,
            batch_size=int(config["training"].get("eval_batch_size", 1)),
            max_length=int(config["method"].get("max_sequence_length", config["tokenization"]["max_sequence_length"])),
        )
        print(summary)
        return

    if method_name == "rlvr":
        from ..models.policy import load_policy_bundle
        from ..models.reference import build_reference_bundle
        from ..rollout.verifiers import GSM8KAnswerVerifier

        policy_bundle = load_policy_bundle(_checkpoint_spec(config, "model", checkpoint_dir))
        reference_bundle = build_reference_bundle(spec=model_spec_from_config(config, "reference_model"))
        examples = verifiable_examples(load_eval_examples(config))
        summary = evaluate_rlvr_policy(
            config,
            candidate_model=policy_bundle.model,
            candidate_tokenizer=policy_bundle.tokenizer,
            reference_model=reference_bundle.model,
            examples=examples,
            verifier=GSM8KAnswerVerifier(),
        )
        print(summary)
        return

    from ..models.policy import load_policy_bundle
    from ..models.reference import build_reference_bundle
    from ..models.reward import load_reward_bundle
    from ..rollout.rewards import LearnedRewardFunction

    if "reward_model" not in config:
        raise ValueError("HH-RLHF policy evaluation requires a `reward_model` section in the experiment config.")
    reward_spec = model_spec_from_config(config, "reward_model")
    require_checkpoint(
        reward_spec.hf_path,
        f"Reward-model checkpoint '{reward_spec.hf_path}' was not found. Run train_rm first.",
    )

    policy_bundle = load_policy_bundle(_checkpoint_spec(config, "model", checkpoint_dir))
    reference_bundle = build_reference_bundle(spec=model_spec_from_config(config, "reference_model")) if "reference_model" in config else None
    reward_bundle = load_reward_bundle(reward_spec, freeze=True)
    reward_function = LearnedRewardFunction(
        model=reward_bundle.model,
        tokenizer=reward_bundle.tokenizer,
        max_length=int(config["method"].get("max_sequence_length", config["tokenization"]["max_sequence_length"])),
    )
    examples = preference_examples(load_eval_examples(config))
    summary = evaluate_hh_policy(
        config,
        candidate_model=policy_bundle.model,
        candidate_tokenizer=policy_bundle.tokenizer,
        reference_model=reference_bundle.model if reference_bundle is not None else policy_bundle.model,
        reward_function=reward_function,
        prompt_examples=examples,
        pair_examples=examples if method_name == "dpo" else None,
        baseline_model=reference_bundle.model if reference_bundle is not None else policy_bundle.model,
        baseline_tokenizer=reference_bundle.tokenizer if reference_bundle is not None else policy_bundle.tokenizer,
    )
    print(summary)


if __name__ == "__main__":
    main()
