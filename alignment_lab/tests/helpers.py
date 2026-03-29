from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from alignlab.rollout.rewards import RewardFunction


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.padding_side = "right"
        self._token_to_id = {
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
            self.unk_token: self.unk_token_id,
            "Human:": 3,
            "Assistant:": 4,
            "hello": 5,
            "good": 6,
            "bad": 7,
            "42": 8,
            "math": 9,
            "?": 10,
            "The": 11,
            "answer": 12,
            "is": 13,
            "4": 14,
        }
        self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}

    def _ensure_token(self, token: str) -> int:
        if token not in self._token_to_id:
            idx = len(self._token_to_id)
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
        return self._token_to_id[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = [token for token in text.strip().split() if token]
        ids = [self._ensure_token(token) for token in tokens]
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        return ids

    def __call__(
        self,
        texts,
        padding: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(text) for text in texts]
        if truncation and max_length is not None:
            if self.padding_side == "left":
                encoded = [ids[-max_length:] for ids in encoded]
            else:
                encoded = [ids[:max_length] for ids in encoded]
        max_len = max(len(ids) for ids in encoded)
        padded_ids = []
        padded_mask = []
        for ids in encoded:
            pad_len = max_len - len(ids)
            pad = [self.pad_token_id] * pad_len
            mask_pad = [0] * pad_len
            if self.padding_side == "left":
                padded_ids.append(pad + ids)
                padded_mask.append(mask_pad + [1] * len(ids))
            else:
                padded_ids.append(ids + pad)
                padded_mask.append([1] * len(ids) + mask_pad)
        output = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
        }
        return output if return_tensors == "pt" else output

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> list[str]:
        rows = []
        for row in sequences.tolist():
            tokens = []
            for idx in row:
                if skip_special_tokens and idx in {self.pad_token_id, self.eos_token_id}:
                    continue
                tokens.append(self._id_to_token.get(int(idx), self.unk_token))
            rows.append(" ".join(tokens).strip())
        return rows

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)


class DummyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 16, generation_cycle: list[int] | None = None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)
        self.generation_cycle = generation_cycle or [6, 7]

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False):  # type: ignore[override]
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        output = SimpleNamespace(logits=logits)
        if output_hidden_states:
            output.hidden_states = (hidden, hidden)
        return output

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens: int = 1,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
    ):
        extra = []
        for row_idx in range(input_ids.size(0)):
            token = self.generation_cycle[row_idx % len(self.generation_cycle)]
            extra.append([token] * max_new_tokens)
        extra_tensor = torch.tensor(extra, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, extra_tensor], dim=-1)


class DummyRewardModel(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):  # type: ignore[override]
        hidden = self.embed(input_ids).mean(dim=1)
        logits = self.proj(hidden)
        return SimpleNamespace(logits=logits)


class DummyValueModel(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):  # type: ignore[override]
        hidden = self.embed(input_ids)
        return self.proj(hidden).squeeze(-1)


class KeywordRewardFunction(RewardFunction):
    def score_batch(self, prompts, responses, targets=None, meta=None):  # type: ignore[override]
        rewards = []
        for response in responses:
            if "good" in response or response.strip() == "42":
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return torch.tensor(rewards, dtype=torch.float32)
