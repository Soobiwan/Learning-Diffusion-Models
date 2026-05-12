from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Nearest-neighbor VQ layer with optional EMA updates and dead-code restart."""

    def __init__(
        self,
        num_codes: int = 256,
        code_dim: int = 64,
        beta: float = 0.25,
        ema: bool = False,
        gamma: float = 0.99,
        dead_threshold: float = 2.0,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.ema = ema
        self.gamma = gamma
        self.dead_threshold = dead_threshold
        embed = torch.empty(num_codes, code_dim).uniform_(-1.0 / num_codes, 1.0 / num_codes)
        self.codebook = nn.Parameter(embed, requires_grad=not ema)
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_sum", embed.clone())
        self.register_buffer("usage_count", torch.zeros(num_codes))

    def _distances(self, flat: torch.Tensor) -> torch.Tensor:
        codebook = self.codebook
        return flat.pow(2).sum(1, keepdim=True) - 2 * flat @ codebook.t() + codebook.pow(2).sum(1)

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, encodings: torch.Tensor) -> None:
        counts = encodings.sum(0)
        sums = encodings.t() @ flat
        self.ema_count.mul_(self.gamma).add_(counts, alpha=1 - self.gamma)
        self.ema_sum.mul_(self.gamma).add_(sums, alpha=1 - self.gamma)
        denom = self.ema_count.clamp_min(1e-5).unsqueeze(1)
        self.codebook.data.copy_(self.ema_sum / denom)

    @torch.no_grad()
    def restart_dead_codes(self, flat: torch.Tensor, counts: torch.Tensor) -> int:
        dead = counts < self.dead_threshold
        n_dead = int(dead.sum().item())
        if self.training and n_dead > 0 and flat.numel() > 0:
            idx = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
            self.codebook.data[dead] = flat[idx].detach()
            self.ema_sum[dead] = flat[idx].detach()
            self.ema_count[dead] = self.dead_threshold + 1
            self.usage_count[dead] = self.dead_threshold + 1
        return n_dead

    def forward(self, ze: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        b, c, h, w = ze.shape
        flat = ze.permute(0, 2, 3, 1).reshape(-1, c)
        ids = self._distances(flat).argmin(1)
        enc = F.one_hot(ids, self.num_codes).type_as(flat)
        zq_flat = F.embedding(ids, self.codebook)
        zq = zq_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        if self.ema and self.training:
            self._ema_update(flat.detach(), enc.detach())
            codebook_loss = flat.new_zeros(())
        else:
            codebook_loss = F.mse_loss(zq, ze.detach())
        commitment_loss = F.mse_loss(ze, zq.detach())
        zq_st = ze + (zq - ze).detach()

        avg_probs = enc.float().mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        batch_counts = enc.sum(0).detach()
        self.usage_count.mul_(self.gamma).add_(batch_counts, alpha=1 - self.gamma)
        dead_counts = self.ema_count if self.ema else self.usage_count
        dead = self.restart_dead_codes(flat.detach(), dead_counts)
        stats = {"perplexity": float(perplexity.item()), "dead_codes": dead}
        return zq_st, ids.view(b, h, w), codebook_loss, commitment_loss, stats
