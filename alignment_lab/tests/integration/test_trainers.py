from __future__ import annotations

import torch

from alignlab.data.collators import PreferenceCollator, PromptOnlyCollator, RewardModelCollator, SFTCollator
from alignlab.data.schemas import PreferenceExample, SFTExample, VerifiableExample
from alignlab.objectives.dpo import DPOObjective
from alignlab.objectives.grpo import GRPOObjective
from alignlab.objectives.ppo import PPOObjective
from alignlab.objectives.rlvr import RLVRObjective
from alignlab.rollout.rewards import VerifiableRewardFunction
from alignlab.rollout.verifiers import GSM8KAnswerVerifier
from alignlab.trainers.online_rl_trainer import OnlineRLTrainer
from alignlab.trainers.pairwise_trainer import PairwiseTrainer
from alignlab.trainers.rm_trainer import RewardModelTrainer
from alignlab.trainers.sft_trainer import SFTTrainer
from tests.helpers import DummyCausalLM, DummyRewardModel, DummyValueModel, KeywordRewardFunction


def test_sft_one_batch(tokenizer) -> None:
    collator = SFTCollator(tokenizer=tokenizer, max_length=8)
    batch = collator([SFTExample(prompt="Human: hello Assistant:", response="good")])
    model = DummyCausalLM(vocab_size=tokenizer.vocab_size + 8)
    trainer = SFTTrainer(model=model, learning_rate=1.0e-3)
    metrics = trainer.train_batch(batch)
    assert "loss" in metrics
    assert metrics["loss"] >= 0.0


def test_rm_one_batch(tokenizer) -> None:
    collator = RewardModelCollator(tokenizer=tokenizer, max_length=8)
    batch = collator([PreferenceExample(prompt="Human: hello Assistant:", chosen="good", rejected="bad")])
    model = DummyRewardModel(vocab_size=tokenizer.vocab_size + 8)
    trainer = RewardModelTrainer(model=model, learning_rate=1.0e-3)
    metrics = trainer.train_batch(batch)
    assert "reward_accuracy" in metrics


def test_dpo_one_batch(tokenizer) -> None:
    collator = PreferenceCollator(tokenizer=tokenizer, max_length=8)
    batch = collator([PreferenceExample(prompt="Human: hello Assistant:", chosen="good", rejected="bad")])
    policy = DummyCausalLM(vocab_size=tokenizer.vocab_size + 8)
    reference = DummyCausalLM(vocab_size=tokenizer.vocab_size + 8)
    trainer = PairwiseTrainer(
        model=policy,
        objective=DPOObjective(beta=0.1),
        reference_model=reference,
        learning_rate=1.0e-3,
    )
    metrics = trainer.train_batch(batch)
    assert "z_margin" in metrics


def test_ppo_one_batch(tokenizer) -> None:
    prompts = [SFTExample(prompt="Human: hello Assistant:", response="good")]
    prompt_collator = PromptOnlyCollator(tokenizer=tokenizer, max_length=6)
    prompt_batch = prompt_collator(prompts)
    trainer = OnlineRLTrainer(
        model=DummyCausalLM(vocab_size=tokenizer.vocab_size + 8, generation_cycle=[6]),
        objective=PPOObjective(),
        reward_function=KeywordRewardFunction(),
        tokenizer=tokenizer,
        reference_model=DummyCausalLM(vocab_size=tokenizer.vocab_size + 8, generation_cycle=[7]),
        value_model=DummyValueModel(vocab_size=tokenizer.vocab_size + 8),
        generation_config={"max_new_tokens": 1, "do_sample": False},
        update_minibatch_size=1,
        epochs_per_rollout=1,
        group_size=1,
        learning_rate=1.0e-3,
        cpu_rollout_cache=False,
    )
    rollout = trainer.collect_rollouts(prompt_batch)
    cached_old_logprobs = rollout.old_logprobs.clone()
    metrics = trainer.update_from_rollout(rollout)
    assert "policy_loss" in metrics
    assert abs(metrics["ratio_start_mean"] - 1.0) < 1.0e-5
    assert torch.allclose(rollout.old_logprobs, cached_old_logprobs)


def test_grpo_one_batch(tokenizer) -> None:
    prompts = [PreferenceExample(prompt="Human: hello Assistant:", chosen="good", rejected="bad")]
    prompt_collator = PromptOnlyCollator(tokenizer=tokenizer, max_length=6)
    prompt_batch = prompt_collator(prompts)
    trainer = OnlineRLTrainer(
        model=DummyCausalLM(vocab_size=tokenizer.vocab_size + 8, generation_cycle=[6, 7]),
        objective=GRPOObjective(beta_kl=0.01),
        reward_function=KeywordRewardFunction(),
        tokenizer=tokenizer,
        reference_model=DummyCausalLM(vocab_size=tokenizer.vocab_size + 8, generation_cycle=[7, 7]),
        generation_config={"max_new_tokens": 1, "do_sample": False},
        update_minibatch_size=2,
        epochs_per_rollout=1,
        group_size=2,
        learning_rate=1.0e-3,
        cpu_rollout_cache=False,
    )
    metrics = trainer.train_batch(prompt_batch)
    assert "degenerate_fraction" in metrics


def test_rlvr_one_batch(tokenizer) -> None:
    prompts = [VerifiableExample(prompt="math ?", gold_answer="42")]
    prompt_collator = PromptOnlyCollator(tokenizer=tokenizer, max_length=4)
    prompt_batch = prompt_collator(prompts)
    trainer = OnlineRLTrainer(
        model=DummyCausalLM(vocab_size=tokenizer.vocab_size + 8, generation_cycle=[8]),
        objective=RLVRObjective(beta_kl=0.01),
        reward_function=VerifiableRewardFunction(GSM8KAnswerVerifier()),
        tokenizer=tokenizer,
        reference_model=DummyCausalLM(vocab_size=tokenizer.vocab_size + 8, generation_cycle=[7]),
        generation_config={"max_new_tokens": 1, "do_sample": False},
        update_minibatch_size=1,
        epochs_per_rollout=1,
        group_size=1,
        learning_rate=1.0e-3,
        cpu_rollout_cache=False,
    )
    metrics = trainer.train_batch(prompt_batch)
    assert "loss" in metrics
