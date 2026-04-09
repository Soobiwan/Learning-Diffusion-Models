from __future__ import annotations

import torch

from alignlab.data.collators import PreferenceCollator
from alignlab.data.schemas import PreferenceExample
from alignlab.objectives.dpo import dpo_loss
from alignlab.objectives.grpo import clipped_group_policy_loss
from alignlab.objectives.ppo import clipped_policy_loss
from alignlab.rollout.advantages import group_relative_advantages, normalize_advantages
from alignlab.rollout.gae import compute_gae
from alignlab.rollout.logprobs import sequence_logprobs_from_logits
from tests.helpers import DummyCausalLM, DummyTokenizer


def test_sequence_logprob_masking_ignores_prompt_tokens() -> None:
    logits = torch.log(
        torch.tensor(
            [
                [
                    [0.1, 0.7, 0.2],
                    [0.1, 0.2, 0.7],
                    [0.8, 0.1, 0.1],
                    [0.2, 0.5, 0.3],
                ]
            ],
            dtype=torch.float32,
        )
    )
    labels = torch.tensor([[-100, 1, 2, -100]])
    sequence_logprobs, mask = sequence_logprobs_from_logits(logits, labels)
    expected = torch.log(torch.tensor(0.7)) + torch.log(torch.tensor(0.7))
    assert torch.allclose(sequence_logprobs, expected.unsqueeze(0), atol=1.0e-5)
    assert mask.sum().item() == 2


def test_dpo_loss_prefers_higher_chosen_margin() -> None:
    output = dpo_loss(
        policy_chosen_logps=torch.tensor([3.0, 2.0]),
        policy_rejected_logps=torch.tensor([1.0, 0.0]),
        reference_chosen_logps=torch.tensor([2.0, 1.0]),
        reference_rejected_logps=torch.tensor([1.5, -0.5]),
        beta=0.5,
    )
    assert output.loss.item() > 0.0
    assert float(output.metrics["preference_accuracy"]) == 1.0


def test_gae_computation_matches_simple_terminal_case() -> None:
    rewards = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[0.2, 0.1]])
    dones = torch.tensor([[0.0, 1.0]])
    advantages, returns = compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=1.0)
    assert torch.allclose(advantages, torch.tensor([[0.8, 0.9]]), atol=1.0e-5)
    assert torch.allclose(returns, torch.tensor([[1.0, 1.0]]), atol=1.0e-5)


def test_ppo_clipping_behavior() -> None:
    loss, ratios, clipped_fraction = clipped_policy_loss(
        new_logprobs=torch.log(torch.tensor([[2.0]])),
        old_logprobs=torch.log(torch.tensor([[1.0]])),
        advantages=torch.tensor([[1.0]]),
        mask=torch.tensor([[1]], dtype=torch.bool),
        clip_range=0.2,
    )
    assert ratios.item() == 2.0
    assert clipped_fraction.item() == 1.0
    assert loss.item() < 0.0


def test_ppo_clipping_blocks_gradient_for_positive_advantage_above_clip() -> None:
    new_logprobs = torch.tensor([[torch.log(torch.tensor(1.5)).item()]], requires_grad=True)
    old_logprobs = torch.log(torch.tensor([[1.0]]))
    advantages = torch.tensor([[1.0]])
    mask = torch.tensor([[1]], dtype=torch.bool)
    loss, ratios, clipped_fraction = clipped_policy_loss(
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        mask=mask,
        clip_range=0.2,
    )
    loss.backward()
    assert torch.allclose(loss.detach(), torch.tensor(-1.2), atol=1.0e-5)
    assert torch.allclose(ratios.detach(), torch.tensor([[1.5]]), atol=1.0e-5)
    assert torch.allclose(clipped_fraction.detach(), torch.tensor(1.0), atol=1.0e-5)
    assert torch.allclose(new_logprobs.grad, torch.tensor([[0.0]]), atol=1.0e-6)


def test_grpo_group_advantages() -> None:
    rewards = torch.tensor([1.0, 3.0, 2.0, 2.0])
    advantages, group_std = group_relative_advantages(rewards, group_size=2)
    assert torch.allclose(advantages[:2], torch.tensor([-1.0, 1.0]), atol=1.0e-5)
    assert torch.allclose(advantages[2:], torch.tensor([0.0, 0.0]), atol=1.0e-5)
    assert group_std.shape[0] == 2


def test_grpo_clipping_blocks_gradient_for_positive_advantage_above_clip() -> None:
    new_logprobs = torch.tensor([[torch.log(torch.tensor(1.5)).item()]], requires_grad=True)
    old_logprobs = torch.log(torch.tensor([[1.0]]))
    advantages = torch.tensor([[1.0]])
    mask = torch.tensor([[1]], dtype=torch.bool)
    loss, ratios, clipped_fraction = clipped_group_policy_loss(
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        mask=mask,
        clip_range=0.2,
    )
    loss.backward()
    assert torch.allclose(loss.detach(), torch.tensor(-1.2), atol=1.0e-5)
    assert torch.allclose(ratios.detach(), torch.tensor([[1.5]]), atol=1.0e-5)
    assert torch.allclose(clipped_fraction.detach(), torch.tensor(1.0), atol=1.0e-5)
    assert torch.allclose(new_logprobs.grad, torch.tensor([[0.0]]), atol=1.0e-6)


def test_grpo_sequence_normalization_uses_per_sequence_mean() -> None:
    loss, _, _ = clipped_group_policy_loss(
        new_logprobs=torch.zeros((2, 3)),
        old_logprobs=torch.zeros((2, 3)),
        advantages=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        mask=torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
        clip_range=0.2,
    )
    assert torch.allclose(loss, torch.tensor(-0.75), atol=1.0e-5)


def test_masked_advantage_normalization_ignores_padding_tokens() -> None:
    normalized = normalize_advantages(
        torch.tensor([[1.0, 1000.0], [3.0, 1000.0]]),
        mask=torch.tensor([[1, 0], [1, 0]], dtype=torch.bool),
    )
    assert torch.allclose(normalized[:, 0], torch.tensor([-1.0, 1.0]), atol=1.0e-5)
    assert torch.allclose(normalized[:, 1], torch.tensor([0.0, 0.0]), atol=1.0e-5)


def test_dpo_sequence_logprob_is_padding_invariant() -> None:
    tokenizer = DummyTokenizer()
    model = DummyCausalLM(vocab_size=tokenizer.vocab_size + 8)
    example = PreferenceExample(prompt="Human: hello Assistant:", chosen="good", rejected="bad")

    def _score(padding_side: str) -> torch.Tensor:
        tokenizer.padding_side = padding_side
        batch = PreferenceCollator(tokenizer=tokenizer, max_length=12)([example])
        outputs = model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        )
        score, _ = sequence_logprobs_from_logits(outputs.logits, batch["chosen_labels"])
        return score

    assert torch.allclose(_score("left"), _score("right"), atol=1.0e-5)
