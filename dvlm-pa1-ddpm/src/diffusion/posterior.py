import torch

from .schedule import DiffusionSchedule, extract


def predict_x0_from_eps(
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_pred: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    sqrt_alpha_bar_t = extract(schedule.sqrt_alpha_bars, t, x_t.shape)
    sqrt_one_minus_alpha_bar_t = extract(schedule.sqrt_one_minus_alpha_bars, t, x_t.shape)
    return (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t


def q_posterior_mean_variance(
    x0: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    coef1 = extract(schedule.posterior_mean_coef1, t, x_t.shape)
    coef2 = extract(schedule.posterior_mean_coef2, t, x_t.shape)
    mean = coef1 * x0 + coef2 * x_t
    var = extract(schedule.posterior_variance, t, x_t.shape)
    return mean, var


def p_mean_from_eps(
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_pred: torch.Tensor,
    schedule: DiffusionSchedule,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Eq. (8): mu_theta(x_t, t) = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
    sqrt_recip_alpha_t = extract(schedule.sqrt_recip_alphas, t, x_t.shape)
    beta_t = extract(schedule.betas, t, x_t.shape)
    sqrt_one_minus_alpha_bar_t = extract(schedule.sqrt_one_minus_alpha_bars, t, x_t.shape)
    mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_pred)

    # DDPM ancestral variance uses beta_tilde (posterior variance).
    var = extract(schedule.posterior_variance, t, x_t.shape)
    x0_pred = predict_x0_from_eps(x_t=x_t, t=t, eps_pred=eps_pred, schedule=schedule)
    return mean, var, x0_pred


def p_sample_step(
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_pred: torch.Tensor,
    schedule: DiffusionSchedule,
    noise_generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    DDPM ancestral reverse step:
    x_{t-1} = mu_theta(x_t, t) + sqrt(beta_tilde_t) * z for t > 0, and no noise for t = 0.
    """
    mean, var, _ = p_mean_from_eps(x_t=x_t, t=t, eps_pred=eps_pred, schedule=schedule)
    if noise_generator is None:
        noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype)
    else:
        try:
            noise = torch.randn(
                x_t.shape,
                device=x_t.device,
                dtype=x_t.dtype,
                generator=noise_generator,
            )
        except TypeError:
            # Fallback for older torch versions with limited generator support.
            noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype)
    nonzero_mask = (t > 0).float().view(t.shape[0], 1, 1, 1)
    return mean + nonzero_mask * torch.sqrt(var) * noise
