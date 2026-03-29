from __future__ import annotations

import torch

from alignlab.eval.generations import build_generation_table
from alignlab.eval.gsm8k_eval import gsm8k_pass_at_1
from alignlab.eval.kl_eval import estimate_policy_reference_kl
from alignlab.eval.preference_eval import preference_accuracy_from_logprobs
from alignlab.eval.rm_eval import reward_model_win_rate_vs_sft
from alignlab.models.reward import last_non_pad_indices
from alignlab.rollout.verifiers import GSM8KAnswerVerifier


def test_last_non_pad_indices() -> None:
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    assert torch.equal(last_non_pad_indices(attention_mask), torch.tensor([2, 1]))


def test_reward_model_win_rate_vs_sft() -> None:
    win_rate = reward_model_win_rate_vs_sft([1.0, 0.2, 0.8], [0.4, 0.3, 0.5])
    assert abs(win_rate - (2.0 / 3.0)) < 1.0e-6


def test_preference_accuracy_from_logprobs() -> None:
    accuracy = preference_accuracy_from_logprobs(
        chosen_logprobs=torch.tensor([2.0, 0.0, 1.0]),
        rejected_logprobs=torch.tensor([1.0, 1.0, 0.5]),
    )
    assert abs(accuracy - (2.0 / 3.0)) < 1.0e-6


def test_kl_and_generation_table_helpers() -> None:
    kl = estimate_policy_reference_kl(
        policy_logprobs=torch.tensor([[0.2, 0.1]]),
        reference_logprobs=torch.tensor([[0.0, 0.0]]),
        mask=torch.tensor([[1, 1]], dtype=torch.bool),
    )
    table = build_generation_table(["p"], ["r"], rewards=[1.0])
    assert kl > 0.0
    assert table[0]["reward"] == 1.0


def test_gsm8k_pass_at_1() -> None:
    value = gsm8k_pass_at_1(["The answer is 42", "I think 7"], ["42", "8"], GSM8KAnswerVerifier())
    assert abs(value - 0.5) < 1.0e-6
