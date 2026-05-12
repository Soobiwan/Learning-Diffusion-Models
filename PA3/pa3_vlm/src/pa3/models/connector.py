import torch
from torch import nn


class MLPConnector(nn.Module):
    """Maps continuous CLIP patch tokens into the SmolLM2 hidden space."""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 960):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(self._init)
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Connector trainable params: {n:,} (expected around 1.66M)")

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=5**0.5)
            nn.init.zeros_(module.bias)

    def forward(self, clip_patches: torch.Tensor) -> torch.Tensor:
        out = self.net(clip_patches)
        if out.ndim == 3:
            print_once = getattr(self, "_printed_shape", False)
            if not print_once:
                print(f"Sanity: Connector output should be [B, 49, 960] -> {tuple(out.shape)}")
                self._printed_shape = True
        return out


def rescale_if_needed(visual_embeds: torch.Tensor, text_embeds: torch.Tensor):
    with torch.no_grad():
        vnorm = visual_embeds.norm(dim=-1).mean()
        tnorm = text_embeds.norm(dim=-1).mean().clamp_min(1e-6)
        ratio = float((vnorm / tnorm).item())
        scale = 1.0
        if ratio < 0.3:
            scale = 0.3 / max(ratio, 1e-6)
        elif ratio > 3.0:
            scale = 3.0 / ratio
    return visual_embeds * scale, ratio, scale

