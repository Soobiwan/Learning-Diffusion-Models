"""Shared trainer for PPO, GRPO, and RLVR."""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import AdamW

from ..objectives.base import Objective
from ..objectives.ppo import PPOObjective
from ..rollout.advantages import broadcast_sequence_advantages, group_relative_advantages
from ..rollout.buffers import RolloutBatch
from ..rollout.gae import compute_gae
from ..rollout.generation import generate_rollout_batch, repeat_prompt_batch
from ..rollout.kl import mean_kl, per_token_kl
from ..rollout.logprobs import gather_token_logprobs
from ..rollout.rewards import RewardFunction
from .base import BaseTrainer


class OnlineRLTrainer(BaseTrainer):
    """Shared online RL trainer for PPO, GRPO, and RLVR."""

    def __init__(
        self,
        model: torch.nn.Module,
        objective: Objective,
        reward_function: RewardFunction,
        tokenizer: Any,
        reference_model: torch.nn.Module | None = None,
        value_model: torch.nn.Module | None = None,
        generation_config: dict[str, Any] | None = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        kl_coef: float = 0.02,
        group_size: int = 1,
        update_minibatch_size: int = 1,
        epochs_per_rollout: int = 1,
        cpu_rollout_cache: bool = True,
        learning_rate: float = 1.0e-4,
        weight_decay: float = 0.0,
        **kwargs: Any,
    ) -> None:
        optimizer = AdamW(
            list(model.parameters()) + list(value_model.parameters()) if value_model is not None else model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            **kwargs,
        )
        if value_model is not None:
            self._set_trainable_parameters(list(self.model.parameters()) + list(value_model.parameters()))
        self.objective = objective
        self.reward_function = reward_function
        self.tokenizer = tokenizer
        self.reference_model = reference_model.to(self.device) if reference_model is not None else None
        self.value_model = value_model.to(self.device) if value_model is not None else None
        self.generation_config = generation_config or {}
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.kl_coef = kl_coef
        self.group_size = group_size
        self.update_minibatch_size = update_minibatch_size
        self.epochs_per_rollout = epochs_per_rollout
        self.cpu_rollout_cache = cpu_rollout_cache
        if self.reference_model is not None:
            self.reference_model.eval()

    def _extract_prompt_fields(
        self,
        prompt_batch: dict[str, Any],
    ) -> tuple[list[str], list[str] | None, list[dict[str, Any]]]:
        raw_examples = prompt_batch.get("raw_examples", [])
        prompts = [str(example["prompt"]) for example in raw_examples]
        targets = [str(example["gold_answer"]) for example in raw_examples if "gold_answer" in example]
        return prompts, (targets if len(targets) == len(prompts) else None), raw_examples

    def collect_rollouts(self, prompt_batch: dict[str, Any]) -> RolloutBatch:
        """Generate responses and cache old policy/reference statistics."""
        prompt_batch = self.move_batch_to_device(prompt_batch)
        prompts, targets, raw_examples = self._extract_prompt_fields(prompt_batch)
        if self.group_size > 1:
            prompt_batch = repeat_prompt_batch(prompt_batch, self.group_size)
            prompts = [prompt for prompt in prompts for _ in range(self.group_size)]
            if targets is not None:
                targets = [target for target in targets for _ in range(self.group_size)]
            raw_examples = [example for example in raw_examples for _ in range(self.group_size)]

        self.model.eval()
        with torch.no_grad():
            generated = generate_rollout_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt_batch=prompt_batch,
                generation_config=self.generation_config,
            )
            policy_outputs = self.model(
                input_ids=generated["input_ids"],
                attention_mask=generated["attention_mask"],
            )
            old_logprobs, token_mask = gather_token_logprobs(policy_outputs.logits, generated["labels"])
            if self.reference_model is not None:
                reference_outputs = self.reference_model(
                    input_ids=generated["input_ids"],
                    attention_mask=generated["attention_mask"],
                )
                ref_logprobs, _ = gather_token_logprobs(reference_outputs.logits, generated["labels"])
            else:
                ref_logprobs = torch.zeros_like(old_logprobs)

            if self.value_model is not None:
                values = self.value_model(
                    input_ids=generated["input_ids"],
                    attention_mask=generated["attention_mask"],
                )[:, 1:]
            else:
                values = torch.zeros_like(old_logprobs)

        rewards = self.reward_function.score_batch(
            prompts=prompts,
            responses=generated["responses"],
            targets=targets,
            meta=raw_examples,
        ).to(self.device, dtype=old_logprobs.dtype)

        token_rewards = torch.zeros_like(old_logprobs)
        dones = torch.zeros_like(old_logprobs)
        response_lengths = token_mask.sum(dim=-1)
        for row_idx, response_length in enumerate(response_lengths.tolist()):
            if response_length == 0:
                continue
            terminal_idx = int(response_length - 1)
            token_rewards[row_idx, terminal_idx] = rewards[row_idx]
            dones[row_idx, terminal_idx] = 1.0

        kl_values = per_token_kl(old_logprobs, ref_logprobs, token_mask)
        shaped_rewards = token_rewards - self.kl_coef * kl_values

        sequence_advantages = None
        advantages = None
        returns = None
        if isinstance(self.objective, PPOObjective):
            advantages, returns = compute_gae(
                rewards=shaped_rewards,
                values=values,
                dones=dones,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
        else:
            sequence_advantages, _ = group_relative_advantages(rewards, self.group_size)
            sequence_advantages = sequence_advantages.to(self.device)
            advantages = broadcast_sequence_advantages(sequence_advantages, token_mask)
            returns = advantages.clone()

        rollout = RolloutBatch(
            input_ids=generated["input_ids"],
            attention_mask=generated["attention_mask"],
            labels=generated["labels"],
            response_mask=token_mask,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            values=values,
            old_values=values.clone(),
            rewards=shaped_rewards,
            advantages=advantages,
            returns=returns,
            sequence_advantages=sequence_advantages,
            prompts=prompts,
            responses=generated["responses"],
            meta={"raw_examples": raw_examples, "mean_kl": mean_kl(old_logprobs, ref_logprobs, token_mask)},
        )
        return rollout.cpu() if self.cpu_rollout_cache else rollout

    def update_from_rollout(self, rollout: RolloutBatch) -> dict[str, float]:
        """Run one policy update over a cached rollout batch."""
        aggregate: dict[str, float] = {}
        first_ratio_mean: float | None = None
        for _ in range(self.epochs_per_rollout):
            for minibatch in rollout.iter_minibatches(self.update_minibatch_size):
                minibatch = minibatch.to(self.device)
                self.model.train()
                outputs = self.model(
                    input_ids=minibatch.input_ids,
                    attention_mask=minibatch.attention_mask,
                )
                new_logprobs, token_mask = gather_token_logprobs(outputs.logits, minibatch.labels)
                if first_ratio_mean is None:
                    ratios = torch.exp(new_logprobs - minibatch.old_logprobs)
                    first_ratio_mean = float(ratios[token_mask].mean().detach().cpu().item())
                if isinstance(self.objective, PPOObjective):
                    if self.value_model is None:
                        raise ValueError("PPO requires a value model.")
                    values = self.value_model(
                        input_ids=minibatch.input_ids,
                        attention_mask=minibatch.attention_mask,
                    )[:, 1:]
                    loss_output = self.objective.compute(
                        new_logprobs=new_logprobs,
                        old_logprobs=minibatch.old_logprobs,
                        advantages=minibatch.advantages,
                        values=values,
                        returns=minibatch.returns,
                        mask=token_mask,
                        old_values=minibatch.old_values,
                    )
                else:
                    kl_values = None
                    if minibatch.ref_logprobs is not None:
                        kl_values = per_token_kl(new_logprobs, minibatch.ref_logprobs, token_mask)
                    loss_output = self.objective.compute(
                        token_logprobs=new_logprobs,
                        advantages=minibatch.advantages,
                        mask=token_mask,
                        kl_values=kl_values,
                    )
                    if minibatch.sequence_advantages is not None:
                        degenerate = float((minibatch.sequence_advantages.abs() < 1.0e-6).float().mean().item())
                        loss_output.metrics["degenerate_fraction"] = degenerate

                self._backward_and_step(loss_output.loss)
                aggregate = self.log_output(loss_output.metrics)
        if first_ratio_mean is not None:
            aggregate["ratio_start_mean"] = first_ratio_mean
        if rollout.meta is not None and "mean_kl" in rollout.meta:
            aggregate["rollout_mean_kl"] = float(rollout.meta["mean_kl"].detach().cpu().item())
        return aggregate

    def train_batch(self, prompt_batch: dict[str, Any]) -> dict[str, float]:
        """Collect rollouts from prompts and run a single update cycle."""
        rollout = self.collect_rollouts(prompt_batch)
        return self.update_from_rollout(rollout)
