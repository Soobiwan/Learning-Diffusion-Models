from __future__ import annotations

from dataclasses import dataclass

from alignlab.data.collators import PreferenceCollator, RewardModelCollator, SFTCollator
from alignlab.data.loaders import load_canonical_dataset
from alignlab.data.schemas import SFTExample
from alignlab.objectives.dpo import DPOObjective
from alignlab.trainers.pairwise_trainer import PairwiseTrainer
from alignlab.trainers.rm_trainer import RewardModelTrainer
from alignlab.trainers.sft_trainer import SFTTrainer
from tests.helpers import DummyCausalLM, DummyRewardModel


@dataclass
class FakeDataset:
    rows: list[dict[str, str]]

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices])


def test_hh_rlhf_real_data_path_sft_rm_dpo(monkeypatch, tokenizer) -> None:
    rows = [
        {
            "chosen": "Human: Hello\n\nAssistant: Nice to meet you",
            "rejected": "Human: Hello\n\nAssistant: No",
        }
    ]

    monkeypatch.setattr("alignlab.data.loaders.load_dataset", lambda *args, **kwargs: FakeDataset(rows))
    examples = load_canonical_dataset(adapter_name="hh_rlhf", path="ignored", split="train")
    assert examples[0].prompt == "Human: Hello\n\nAssistant:"

    sft_batch = SFTCollator(tokenizer=tokenizer, max_length=8)(
        [SFTExample(prompt=examples[0].prompt, response=examples[0].chosen)]
    )
    rm_batch = RewardModelCollator(tokenizer=tokenizer, max_length=8)(examples)
    dpo_batch = PreferenceCollator(tokenizer=tokenizer, max_length=8)(examples)

    sft_metrics = SFTTrainer(model=DummyCausalLM(vocab_size=tokenizer.vocab_size + 8), learning_rate=1.0e-3).train_batch(
        sft_batch
    )
    rm_metrics = RewardModelTrainer(
        model=DummyRewardModel(vocab_size=tokenizer.vocab_size + 8),
        learning_rate=1.0e-3,
    ).train_batch(rm_batch)
    dpo_metrics = PairwiseTrainer(
        model=DummyCausalLM(vocab_size=tokenizer.vocab_size + 8),
        objective=DPOObjective(beta=0.1),
        reference_model=DummyCausalLM(vocab_size=tokenizer.vocab_size + 8),
        learning_rate=1.0e-3,
    ).train_batch(dpo_batch)

    assert "loss" in sft_metrics
    assert "reward_accuracy" in rm_metrics
    assert "preference_accuracy" in dpo_metrics
