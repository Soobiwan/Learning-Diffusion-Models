import math
from dataclasses import dataclass

import torch

from ..config import BETA_END, BETA_START, SCHEDULE_TYPE, TIMESTEPS


@dataclass
class DiffusionSchedule:
    schedule_type: str
    timesteps: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_prev: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor


def make_beta_schedule(
    timesteps: int,
    schedule_type: str = "linear",
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Build betas with explicit schedule type.
    Timesteps are always indexed as 0..T-1 in code.
    """
    if timesteps < 2:
        raise ValueError("timesteps must be >= 2.")
    if not (0.0 < beta_start < 1.0 and 0.0 < beta_end < 1.0):
        raise ValueError("beta_start and beta_end must be in (0, 1).")
    if beta_start > beta_end:
        raise ValueError("beta_start must be <= beta_end.")

    schedule_type = schedule_type.lower()
    if schedule_type == "linear":
        return torch.linspace(
            beta_start,
            beta_end,
            timesteps,
            device=device,
            dtype=torch.float32,
        )

    if schedule_type == "cosine":
        # Cosine alpha_bar schedule from improved DDPM; then convert to betas.
        s = 0.008
        x = torch.linspace(0, timesteps, timesteps + 1, device=device, dtype=torch.float64)
        alpha_bars = torch.cos(((x / timesteps) + s) / (1.0 + s) * (math.pi / 2.0)) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = betas.clamp(min=beta_start, max=beta_end)
        return betas.to(torch.float32)

    raise ValueError("Unsupported schedule_type. Use one of: ['linear', 'cosine'].")


def build_schedule(
    timesteps: int = TIMESTEPS,
    schedule_type: str = SCHEDULE_TYPE,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
    device: torch.device | str = "cpu",
) -> DiffusionSchedule:
    betas = make_beta_schedule(
        timesteps=timesteps,
        schedule_type=schedule_type,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
    )
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat(
        [torch.ones(1, device=device, dtype=alpha_bars.dtype), alpha_bars[:-1]],
        dim=0,
    )

    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    posterior_variance = posterior_variance.clamp(min=1e-20)

    posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
    posterior_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)

    return DiffusionSchedule(
        schedule_type=schedule_type,
        timesteps=timesteps,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        alpha_bars_prev=alpha_bars_prev,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
        sqrt_recip_alphas=sqrt_recip_alphas,
        posterior_variance=posterior_variance,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
    )


def extract(coefficients: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    coefficients: shape [T]
    t: shape [B] with values in [0, T-1]
    returns shape [B, 1, 1, 1] for broadcasting with images
    """
    out = coefficients.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


def sample_timesteps(batch_size: int, timesteps: int, device: torch.device | str) -> torch.Tensor:
    return torch.randint(0, timesteps, (batch_size,), device=device, dtype=torch.long)


def snr(schedule: DiffusionSchedule) -> torch.Tensor:
    return schedule.alpha_bars / (1.0 - schedule.alpha_bars)
