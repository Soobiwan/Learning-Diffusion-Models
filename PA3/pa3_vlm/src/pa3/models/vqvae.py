import torch
from torch import nn
import torch.nn.functional as F

from .vector_quantizer import VectorQuantizer


def gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=8, num_channels=channels)


class VQVAE(nn.Module):
    def __init__(self, num_codes: int = 256, beta: float = 0.25, ema: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), gn(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), gn(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), gn(64), nn.ReLU(inplace=True),
        )
        self.quantizer = VectorQuantizer(num_codes=num_codes, code_dim=64, beta=beta, ema=ema)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), gn(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), gn(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid(),
        )
        self.beta = beta

    def _encode_raw(self, x: torch.Tensor):
        ze = self.encoder(x)
        if not getattr(self, "_printed_latent", False):
            print(f"Sanity: VQ-VAE latent should be [B, 64, 4, 4] -> {tuple(ze.shape)}")
            self._printed_latent = True
        return ze

    def encode(self, x: torch.Tensor):
        ze = self._encode_raw(x)
        zq, ids, cb, com, stats = self.quantizer(ze)
        if not getattr(self, "_printed_ids", False):
            print(f"Sanity: VQ-VAE token map should be [B, 4, 4] -> {tuple(ids.shape)}")
            self._printed_ids = True
        return zq, ids, cb, com, stats

    def decode_codes(self, ids: torch.Tensor) -> torch.Tensor:
        z = F.embedding(ids, self.quantizer.codebook).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        ze = self._encode_raw(x)
        zq, ids, cb, com, stats = self.quantizer(ze)
        if not getattr(self, "_printed_ids", False):
            print(f"Sanity: VQ-VAE token map should be [B, 4, 4] -> {tuple(ids.shape)}")
            self._printed_ids = True
        recon = self.decoder(zq)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + cb + self.beta * com
        with torch.no_grad():
            qgap = (ze.detach() - zq.detach()).pow(2).sum(dim=1).mean()
        stats.update({
            "recon_mse": float(recon_loss.item()),
            "codebook_loss": float(cb.item()),
            "commitment_loss": float(com.item()),
            "quantization_gap": float(qgap.item()),
        })
        return recon, ids, loss, stats
