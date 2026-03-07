from typing import Literal, TypedDict

import torch
import torch.nn as nn

from ..config import MODEL_BASE_CHANNELS, MODEL_TIME_DIM, MODEL_VARIANT
from .blocks import ConvBlock
from .embeddings import SinusoidalTimeEmbedding

UNetVariant = Literal["deep"]


class UNetConfig(TypedDict):
    variant: UNetVariant
    time_dim: int
    base_channels: int


class TinyUNet(nn.Module):
    """
    Time-conditioned deep epsilon network for MNIST.
    This class is intentionally deep-only.
    """

    def __init__(
        self,
        time_dim: int = MODEL_TIME_DIM,
        base_channels: int = MODEL_BASE_CHANNELS,
        variant: UNetVariant = MODEL_VARIANT,
    ):
        super().__init__()
        if variant != "deep":
            raise ValueError(
                "TinyUNet is deep-only in this code version. "
                "Use variant='deep' with the deep checkpoints."
            )

        self.time_dim = int(time_dim)
        self.base_channels = int(base_channels)
        self.variant: UNetVariant = variant

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        c1 = self.base_channels
        c2 = self.base_channels * 2
        c3 = self.base_channels * 4

        self.in_conv = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.down1 = ConvBlock(c1, c1, self.time_dim)
        self.downsample1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)  # 28 -> 14
        self.down2 = ConvBlock(c2, c2, self.time_dim)
        self.downsample2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)  # 14 -> 7
        self.mid1 = ConvBlock(c3, c3, self.time_dim)
        self.mid2 = ConvBlock(c3, c3, self.time_dim)
        self.upsample1 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)  # 7 -> 14
        self.up1 = ConvBlock(c2 + c2, c2, self.time_dim)
        self.upsample2 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)  # 14 -> 28
        self.up2 = ConvBlock(c1 + c1, c1, self.time_dim)

        self.out = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h0 = self.in_conv(x)

        h1 = self.down1(h0, t_emb)
        h2_in = self.downsample1(h1)
        h2 = self.down2(h2_in, t_emb)
        h3 = self.downsample2(h2)

        h_mid = self.mid1(h3, t_emb)
        h_mid = self.mid2(h_mid, t_emb)

        h_up1 = self.upsample1(h_mid)
        h_up1 = torch.cat([h_up1, h2], dim=1)
        h_up1 = self.up1(h_up1, t_emb)
        h_up2 = self.upsample2(h_up1)
        h = torch.cat([h_up2, h1], dim=1)
        h = self.up2(h, t_emb)
        return self.out(h)


def infer_unet_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> UNetConfig:
    if "in_conv.weight" not in state_dict or "time_embed.1.weight" not in state_dict:
        raise ValueError("State dict is missing required UNet keys.")

    keys = state_dict.keys()
    if any(k.startswith("down1.") for k in keys):
        variant: UNetVariant = "deep"
    elif any(k.startswith("down_block.") for k in keys):
        raise ValueError("Legacy checkpoint detected, but TinyUNet is deep-only.")
    else:
        raise ValueError("Could not infer deep UNet variant from checkpoint keys.")

    base_channels = int(state_dict["in_conv.weight"].shape[0])
    time_dim = int(state_dict["time_embed.1.weight"].shape[0])
    return {
        "variant": variant,
        "time_dim": time_dim,
        "base_channels": base_channels,
    }
