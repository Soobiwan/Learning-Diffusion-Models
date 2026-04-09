from __future__ import annotations

import torch

from alignlab.eval.generations import build_generation_table
from alignlab.eval.gsm8k_eval import gsm8k_pass_at_1
from alignlab.eval.kl_eval import estimate_policy_reference_full_vocab_kl, estimate_policy_reference_kl
from alignlab.eval.pa2_tools import preview_canonical_examples, verify_gsm8k_answer_extractor
from alignlab.eval.preference_eval import preference_accuracy_from_logprobs
from alignlab.eval.rm_eval import reward_model_win_rate_vs_sft
from alignlab.models.reward import last_non_pad_indices
from alignlab.models.value import CausalValueModel
from alignlab.data.schemas import VerifiableExample
from alignlab.rollout.verifiers import GSM8KAnswerVerifier
from tests.helpers import DummyCausalLM


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


def test_full_vocab_kl_helper_is_zero_for_identical_logits() -> None:
    logits = torch.tensor(
        [[[1.0, 0.0], [0.5, -0.5], [0.2, 0.8]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 1, 1]])
    kl = estimate_policy_reference_full_vocab_kl(logits, logits, labels)
    assert abs(kl) < 1.0e-8


def test_gsm8k_pass_at_1() -> None:
    value = gsm8k_pass_at_1(["The answer is 42", "I think 7"], ["42", "8"], GSM8KAnswerVerifier())
    assert abs(value - 0.5) < 1.0e-6


def test_pa2_tools_preview_and_gsm8k_verifier_summary() -> None:
    examples = [
        VerifiableExample(
            prompt="Solve this",
            gold_answer="42",
            meta={"answer_text": "#### 42"},
        ),
        VerifiableExample(
            prompt="Solve that",
            gold_answer="3",
            meta={"answer_text": "The answer is 3"},
        ),
    ]
    preview = preview_canonical_examples(examples, limit=1)
    summary = verify_gsm8k_answer_extractor(examples, verifier=GSM8KAnswerVerifier(), gold_limit=2, wrong_limit=2)
    assert preview[0]["gold_answer"] == "42"
    assert summary["gold_accuracy"] == 1.0
    assert summary["wrong_accuracy"] == 0.0


def test_value_head_uses_small_initialization() -> None:
    model = CausalValueModel(backbone=DummyCausalLM(hidden_size=16), hidden_size=16)
    assert float(model.value_head.weight.std().item()) < 0.05
    assert torch.allclose(model.value_head.bias, torch.zeros_like(model.value_head.bias))
