from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class OverlayEmbedding(nn.Module):
    """Frozen text embeddings plus trainable rows for newly added visual tokens."""

    def __init__(self, base_embedding: nn.Embedding, original_vocab: int, num_new_tokens: int = 258):
        super().__init__()
        self.base = base_embedding
        self.original_vocab = original_vocab
        self.num_new_tokens = num_new_tokens
        for p in self.base.parameters():
            p.requires_grad_(False)
        dim = base_embedding.embedding_dim
        self.overlay = nn.Embedding(
            num_new_tokens,
            dim,
            device=base_embedding.weight.device,
        )
        with torch.no_grad():
            mean = base_embedding.weight[:original_vocab].mean(0)
            self.overlay.weight[0].copy_(mean)
            self.overlay.weight[1].copy_(mean)
            self.overlay.weight[2:].zero_()

    @property
    def image_id(self) -> int:
        return self.original_vocab

    @property
    def image_end_id(self) -> int:
        return self.original_vocab + 1

    def visual_id(self, code_idx: torch.Tensor) -> torch.Tensor:
        return code_idx + self.original_vocab + 2

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        safe_ids = input_ids.clamp(max=self.original_vocab - 1)
        out = self.base(safe_ids)
        mask = input_ids >= self.original_vocab
        if mask.any():
            overlay_ids = input_ids[mask] - self.original_vocab
            out = out.clone()
            out[mask] = self.overlay(overlay_ids).to(dtype=out.dtype)
        return out


class OverlayLMHead(nn.Module):
    """Frozen text output rows plus trainable logits for the newly added rows."""

    def __init__(self, base_head: nn.Module, original_vocab: int, overlay: OverlayEmbedding):
        super().__init__()
        self.base_head = base_head
        self.original_vocab = original_vocab
        self.overlay = overlay
        for p in self.base_head.parameters():
            p.requires_grad_(False)
        self.in_features = getattr(base_head, "in_features", overlay.overlay.embedding_dim)
        self.out_features = original_vocab + overlay.num_new_tokens

    @property
    def weight(self) -> torch.Tensor:
        base_weight = self.base_head.weight[:self.original_vocab]
        return torch.cat([base_weight, self.overlay.overlay.weight], dim=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_weight = self.base_head.weight[:self.original_vocab]
        base_bias = getattr(self.base_head, "bias", None)
        if base_bias is not None:
            base_bias = base_bias[:self.original_vocab]
        text_logits = F.linear(hidden_states, base_weight, base_bias)
        new_logits = F.linear(hidden_states, self.overlay.overlay.weight.to(dtype=hidden_states.dtype))
        return torch.cat([text_logits, new_logits], dim=-1)


class CodebookProjector(nn.Module):
    def __init__(self, code_dim: int = 64, hidden_dim: int = 960):
        super().__init__()
        self.proj = nn.Linear(code_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.proj.weight, a=5**0.5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.proj(codes)


@torch.no_grad()
def transplant_codebook_to_overlay(overlay: OverlayEmbedding, projector: CodebookProjector, codebook: torch.Tensor) -> None:
    projected = projector(codebook.to(next(projector.parameters()).device)).to(overlay.overlay.weight.device)
    overlay.overlay.weight[2:2 + codebook.shape[0]].copy_(projected)


def apply_logit_mask(logits: torch.Tensor, mode: str, original_vocab: int, num_visual: int = 256) -> torch.Tensor:
    """Mask generation logits so VQA emits text and image generation emits visual IDs."""
    masked = logits.clone()
    neg = torch.finfo(masked.dtype).min
    if mode == "vqa_text":
        masked[..., original_vocab:] = neg
    elif mode == "image":
        allowed = torch.zeros(masked.shape[-1], dtype=torch.bool, device=masked.device)
        allowed[original_vocab + 1: original_vocab + 2 + num_visual] = True
        masked[..., ~allowed] = neg
    else:
        raise ValueError(f"unknown mask mode {mode}")
    return masked
