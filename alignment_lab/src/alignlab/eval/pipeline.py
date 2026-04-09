"""Shared evaluation pipelines for RM, HH alignment methods, and RLVR."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.collators import PreferenceCollator, RewardModelCollator, SFTCollator
from ..data.schemas import PreferenceExample, SFTExample, VerifiableExample
from ..models.generation import generate_batched
from ..rollout.kl import full_vocab_token_kl_from_logits, per_token_kl
from ..rollout.logprobs import gather_token_logprobs, sequence_logprobs_from_logits
from ..rollout.verifiers import GSM8KAnswerVerifier
from .generations import build_generation_table
from .gsm8k_eval import gsm8k_pass_at_1
from .preference_eval import preference_accuracy_from_logprobs
from .reports import (
    experiment_plot_path,
    experiment_sample_path,
    experiment_table_path,
    plot_histogram,
    write_csv_rows,
    write_generation_artifacts,
    write_json,
)
from .rm_eval import reward_model_win_rate_vs_sft


def _batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@contextmanager
def _preserve_training_mode(*modules: Any) -> Iterator[None]:
    states: list[tuple[Any, bool]] = []
    for module in modules:
        if module is not None and hasattr(module, "training") and hasattr(module, "eval"):
            states.append((module, bool(module.training)))
            module.eval()
    try:
        yield
    finally:
        for module, was_training in states:
            module.train(was_training)


def _greedy_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    generation = dict(config.get("generation", {}))
    generation["do_sample"] = False
    generation["temperature"] = 1.0
    generation["top_p"] = 1.0
    return generation


def _prompt_batch(tokenizer: Any, prompts: Sequence[str], max_prompt_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(prompts),
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


@torch.no_grad()
def _generate_with_labels(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    max_prompt_length: int,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    prompt_batch = _prompt_batch(tokenizer, prompts, max_prompt_length=max_prompt_length, device=device)
    generation = generate_batched(
        model=model,
        tokenizer=tokenizer,
        input_ids=prompt_batch["input_ids"],
        attention_mask=prompt_batch["attention_mask"],
        generation_config=generation_config,
    )
    sequences = generation["sequences"]
    pad_token_id = getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", 0))
    attention_mask = (sequences != pad_token_id).long()
    prompt_lengths = prompt_batch["attention_mask"].sum(dim=-1).tolist()
    labels = sequences.clone()
    response_mask = torch.zeros_like(sequences)
    for row_idx, prompt_length in enumerate(prompt_lengths):
        labels[row_idx, :prompt_length] = -100
        response_mask[row_idx, prompt_length:] = 1
    labels = labels.masked_fill(attention_mask == 0, -100)
    return {
        "input_ids": sequences,
        "attention_mask": attention_mask,
        "labels": labels,
        "response_mask": response_mask[:, 1:].to(torch.bool),
        "responses": generation["responses"],
    }


@torch.no_grad()
def _kl_on_generated_sequences(
    policy_model: Any,
    reference_model: Any,
    generated: dict[str, Any],
    kl_mode: str = "sampled",
) -> tuple[float, int]:
    if reference_model is None:
        return 0.0, 0
    policy_outputs = policy_model(
        input_ids=generated["input_ids"],
        attention_mask=generated["attention_mask"],
    )
    policy_logprobs, mask = gather_token_logprobs(policy_outputs.logits, generated["labels"])
    reference_outputs = reference_model(
        input_ids=generated["input_ids"],
        attention_mask=generated["attention_mask"],
    )
    if kl_mode == "full_vocab":
        kl_values, full_mask = full_vocab_token_kl_from_logits(
            policy_logits=policy_outputs.logits,
            reference_logits=reference_outputs.logits,
            labels=generated["labels"],
        )
        return float(kl_values.sum().detach().cpu().item()), int(full_mask.sum().detach().cpu().item())
    reference_logprobs, _ = gather_token_logprobs(reference_outputs.logits, generated["labels"])
    kl_values = per_token_kl(policy_logprobs, reference_logprobs, mask)
    return float(kl_values.sum().detach().cpu().item()), int(mask.sum().detach().cpu().item())


def _limit_examples(examples: Sequence[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return list(examples)
    return list(examples[:limit])


def evaluate_reward_model(
    config: dict[str, Any],
    reward_model: Any,
    tokenizer: Any,
    examples: Sequence[PreferenceExample],
    batch_size: int,
    max_length: int,
    stem: str = "rm_eval",
) -> dict[str, Any]:
    collator = RewardModelCollator(tokenizer=tokenizer, max_length=max_length)
    dataloader = DataLoader(list(examples), batch_size=batch_size, shuffle=False, collate_fn=collator)
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    with _preserve_training_mode(reward_model):
        for batch in dataloader:
            device = next(reward_model.parameters()).device
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            chosen = reward_model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            ).logits.squeeze(-1)
            rejected = reward_model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            ).logits.squeeze(-1)
            chosen_scores.extend(chosen.detach().cpu().tolist())
            rejected_scores.extend(rejected.detach().cpu().tolist())

    chosen_tensor = torch.tensor(chosen_scores, dtype=torch.float32)
    rejected_tensor = torch.tensor(rejected_scores, dtype=torch.float32)
    accuracy = float((chosen_tensor > rejected_tensor).float().mean().item()) if chosen_scores else 0.0
    summary = {
        "preference_accuracy": accuracy,
        "mean_chosen_score": float(chosen_tensor.mean().item()) if chosen_scores else 0.0,
        "mean_rejected_score": float(rejected_tensor.mean().item()) if rejected_scores else 0.0,
        "num_pairs": len(chosen_scores),
    }
    bins = int(config.get("evaluation", {}).get("histogram_bins", 20))
    histogram_path = plot_histogram(
        experiment_plot_path(config, stem),
        {"chosen": chosen_scores, "rejected": rejected_scores},
        bins=bins,
        title="Reward Model Score Distribution",
    )
    summary_path = write_json(experiment_table_path(config, stem), summary)
    csv_path = write_csv_rows(
        experiment_table_path(config, stem, ".csv"),
        [
            {"kind": "chosen", "score": score}
            for score in chosen_scores
        ]
        + [
            {"kind": "rejected", "score": score}
            for score in rejected_scores
        ],
    )
    summary["artifacts"] = {
        "summary_json": str(summary_path),
        "scores_csv": str(csv_path),
        "histogram_png": str(histogram_path),
    }
    return summary


def evaluate_preference_policy(
    candidate_model: Any,
    tokenizer: Any,
    examples: Sequence[PreferenceExample],
    batch_size: int,
    max_length: int,
) -> float:
    collator = PreferenceCollator(tokenizer=tokenizer, max_length=max_length)
    dataloader = DataLoader(list(examples), batch_size=batch_size, shuffle=False, collate_fn=collator)
    chosen_logprobs: list[torch.Tensor] = []
    rejected_logprobs: list[torch.Tensor] = []
    with _preserve_training_mode(candidate_model):
        for batch in dataloader:
            device = next(candidate_model.parameters()).device
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            chosen_outputs = candidate_model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            )
            rejected_outputs = candidate_model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            )
            chosen_scores, _ = sequence_logprobs_from_logits(chosen_outputs.logits, batch["chosen_labels"])
            rejected_scores, _ = sequence_logprobs_from_logits(rejected_outputs.logits, batch["rejected_labels"])
            chosen_logprobs.append(chosen_scores.detach().cpu())
            rejected_logprobs.append(rejected_scores.detach().cpu())
    if not chosen_logprobs:
        return 0.0
    return preference_accuracy_from_logprobs(torch.cat(chosen_logprobs), torch.cat(rejected_logprobs))


def evaluate_sft_perplexity(
    model: Any,
    tokenizer: Any,
    examples: Sequence[SFTExample],
    batch_size: int,
    max_length: int,
) -> dict[str, float]:
    """Compute held-out response-token perplexity for SFT checkpoints."""
    collator = SFTCollator(tokenizer=tokenizer, max_length=max_length)
    dataloader = DataLoader(list(examples), batch_size=batch_size, shuffle=False, collate_fn=collator)
    total_loss = 0.0
    total_tokens = 0
    with _preserve_training_mode(model):
        for batch in dataloader:
            device = next(model.parameters()).device
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            loss_sum = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            token_count = int(shift_labels.ne(-100).sum().detach().cpu().item())
            total_loss += float(loss_sum.detach().cpu().item())
            total_tokens += token_count
    if total_tokens <= 0:
        return {"heldout_loss": 0.0, "heldout_perplexity": 1.0, "num_eval_tokens": 0}
    mean_loss = total_loss / total_tokens
    return {
        "heldout_loss": mean_loss,
        "heldout_perplexity": float(torch.exp(torch.tensor(mean_loss)).item()),
        "num_eval_tokens": total_tokens,
    }


def evaluate_hh_policy(
    config: dict[str, Any],
    candidate_model: Any,
    candidate_tokenizer: Any,
    reference_model: Any,
    reward_function: Any,
    prompt_examples: Sequence[PreferenceExample],
    pair_examples: Sequence[PreferenceExample] | None = None,
    baseline_model: Any | None = None,
    baseline_tokenizer: Any | None = None,
    stem: str = "policy_eval",
) -> dict[str, Any]:
    baseline_model = baseline_model or reference_model
    baseline_tokenizer = baseline_tokenizer or candidate_tokenizer
    evaluation_cfg = config.get("evaluation", {})
    kl_mode = str(evaluation_cfg.get("kl_mode", "sampled")).lower()
    prompt_limit = int(evaluation_cfg.get("num_eval_prompts", 200))
    sample_limit = int(evaluation_cfg.get("sample_table_size", 5))
    batch_size = int(config["training"].get("eval_batch_size", 1))
    max_prompt_length = int(config["method"].get("max_prompt_length", config["tokenization"]["max_prompt_length"]))
    generation_config = _greedy_generation_config(config)

    selected_prompts = _limit_examples(list(prompt_examples), prompt_limit)
    candidate_scores: list[float] = []
    baseline_scores: list[float] = []
    sample_rows: list[dict[str, Any]] = []
    kl_total = 0.0
    kl_tokens = 0
    response_lengths: list[int] = []

    with ExitStack() as stack:
        stack.enter_context(_preserve_training_mode(candidate_model, reference_model, baseline_model))
        for batch_examples in _batched(selected_prompts, batch_size):
            prompts = [example.prompt for example in batch_examples]
            candidate_generated = _generate_with_labels(
                candidate_model,
                candidate_tokenizer,
                prompts,
                max_prompt_length=max_prompt_length,
                generation_config=generation_config,
            )
            baseline_generated = _generate_with_labels(
                baseline_model,
                baseline_tokenizer,
                prompts,
                max_prompt_length=max_prompt_length,
                generation_config=generation_config,
            )
            candidate_rewards = reward_function.score_batch(prompts, candidate_generated["responses"]).tolist()
            baseline_rewards = reward_function.score_batch(prompts, baseline_generated["responses"]).tolist()
            candidate_scores.extend(float(score) for score in candidate_rewards)
            baseline_scores.extend(float(score) for score in baseline_rewards)
            response_lengths.extend(len(response.split()) for response in candidate_generated["responses"])

            batch_kl, batch_kl_tokens = _kl_on_generated_sequences(
                candidate_model,
                reference_model,
                candidate_generated,
                kl_mode=kl_mode,
            )
            kl_total += batch_kl
            kl_tokens += batch_kl_tokens

            if len(sample_rows) < sample_limit:
                for idx, prompt in enumerate(prompts):
                    sample_rows.append(
                        {
                            "prompt": prompt,
                            "candidate_response": candidate_generated["responses"][idx],
                            "baseline_response": baseline_generated["responses"][idx],
                            "candidate_reward": candidate_rewards[idx],
                            "baseline_reward": baseline_rewards[idx],
                            "candidate_response_length": len(candidate_generated["responses"][idx].split()),
                            "baseline_response_length": len(baseline_generated["responses"][idx].split()),
                        }
                    )
                    if len(sample_rows) >= sample_limit:
                        break

    summary = {
        "rm_score_mean": sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0.0,
        "rm_win_rate_vs_sft": reward_model_win_rate_vs_sft(candidate_scores, baseline_scores) if candidate_scores else 0.0,
        "kl_from_reference": (kl_total / kl_tokens) if kl_tokens > 0 else 0.0,
        "kl_mode": kl_mode,
        "mean_response_length": (sum(response_lengths) / len(response_lengths)) if response_lengths else 0.0,
        "num_eval_prompts": len(selected_prompts),
    }
    if pair_examples is not None:
        pair_limit = int(evaluation_cfg.get("num_eval_pairs", 200))
        summary["preference_accuracy"] = evaluate_preference_policy(
            candidate_model,
            candidate_tokenizer,
            _limit_examples(list(pair_examples), pair_limit),
            batch_size=batch_size,
            max_length=int(config["method"].get("max_sequence_length", config["tokenization"]["max_sequence_length"])),
        )
        summary["num_eval_pairs"] = min(pair_limit, len(pair_examples))

    summary_path = write_json(experiment_table_path(config, stem), summary)
    sample_paths = write_generation_artifacts(config, stem, sample_rows)
    summary["artifacts"] = {
        "summary_json": str(summary_path),
        **sample_paths,
    }
    return summary


def evaluate_rlvr_policy(
    config: dict[str, Any],
    candidate_model: Any,
    candidate_tokenizer: Any,
    reference_model: Any,
    examples: Sequence[VerifiableExample],
    verifier: GSM8KAnswerVerifier | None = None,
    stem: str = "rlvr_eval",
) -> dict[str, Any]:
    verifier = verifier or GSM8KAnswerVerifier()
    evaluation_cfg = config.get("evaluation", {})
    kl_mode = str(evaluation_cfg.get("kl_mode", "sampled")).lower()
    prompt_limit = int(evaluation_cfg.get("num_eval_prompts", 200))
    sample_limit = int(evaluation_cfg.get("sample_table_size", 5))
    batch_size = int(config["training"].get("eval_batch_size", 1))
    max_prompt_length = int(config["method"].get("max_prompt_length", config["tokenization"]["max_prompt_length"]))
    generation_config = _greedy_generation_config(config)

    selected_examples = _limit_examples(list(examples), prompt_limit)
    responses: list[str] = []
    gold_answers: list[str] = []
    sample_rows: list[dict[str, Any]] = []
    kl_total = 0.0
    kl_tokens = 0

    with _preserve_training_mode(candidate_model, reference_model):
        for batch_examples in _batched(selected_examples, batch_size):
            prompts = [example.prompt for example in batch_examples]
            generated = _generate_with_labels(
                candidate_model,
                candidate_tokenizer,
                prompts,
                max_prompt_length=max_prompt_length,
                generation_config=generation_config,
            )
            responses.extend(generated["responses"])
            gold_answers.extend([example.gold_answer for example in batch_examples])

            batch_kl, batch_kl_tokens = _kl_on_generated_sequences(
                candidate_model,
                reference_model,
                generated,
                kl_mode=kl_mode,
            )
            kl_total += batch_kl
            kl_tokens += batch_kl_tokens

            if len(sample_rows) < sample_limit:
                for idx, example in enumerate(batch_examples):
                    response = generated["responses"][idx]
                    sample_rows.append(
                        {
                            "prompt": example.prompt,
                            "response": response,
                            "gold_answer": example.gold_answer,
                            "correct": verifier.verify(response=response, gold_answer=example.gold_answer) > 0.0,
                            "has_final_answer": verifier.has_valid_answer(response),
                        }
                    )
                    if len(sample_rows) >= sample_limit:
                        break

    format_compliance = [
        1.0 if verifier.has_valid_answer(response) else 0.0
        for response in responses
    ]
    response_lengths = [len(response.split()) for response in responses]
    summary = {
        "pass_at_1": gsm8k_pass_at_1(responses, gold_answers, verifier=verifier),
        "format_compliance_rate": sum(format_compliance) / len(format_compliance) if format_compliance else 0.0,
        "mean_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0.0,
        "kl_from_reference": (kl_total / kl_tokens) if kl_tokens > 0 else 0.0,
        "kl_mode": kl_mode,
        "num_eval_prompts": len(selected_examples),
    }
    summary_path = write_json(experiment_table_path(config, stem), summary)
    sample_paths = write_generation_artifacts(config, stem, sample_rows)
    summary["artifacts"] = {
        "summary_json": str(summary_path),
        **sample_paths,
    }
    return summary


def write_eval_summary(config: dict[str, Any], stem: str, payload: dict[str, Any]) -> Path:
    return write_json(experiment_table_path(config, stem), payload)


def sample_rows_from_examples(examples: Sequence[Any]) -> list[dict[str, Any]]:
    return [asdict(example) if hasattr(example, "__dataclass_fields__") else dict(example) for example in examples]
