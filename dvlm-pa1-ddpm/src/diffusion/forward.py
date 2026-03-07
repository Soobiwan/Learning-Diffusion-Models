import torch

from .schedule import DiffusionSchedule, extract


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Sample x_t from q(x_t | x_0):
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    """
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_alpha_bar_t = extract(schedule.sqrt_alpha_bars, t, x0.shape)
    sqrt_one_minus_alpha_bar_t = extract(schedule.sqrt_one_minus_alpha_bars, t, x0.shape)
    return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise


def sample_training_pair(
    x0: torch.Tensor,
    schedule: DiffusionSchedule,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (t, noise, x_t) used for L_simple.
    """
    t = torch.randint(0, schedule.timesteps, (x0.shape[0],), device=x0.device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0=x0, t=t, schedule=schedule, noise=noise)
    return t, noise, x_t


@torch.no_grad()
def forward_moment_sanity_check(
    x0: torch.Tensor,
    timestep: int,
    schedule: DiffusionSchedule,
    num_trials: int = 2048,
) -> dict[str, float]:
    """
    Empirically verify q(x_t | x_0) moments against Eq. (2).
    Uses population estimates over repeated noise draws.
    """
    if timestep < 0 or timestep >= schedule.timesteps:
        raise ValueError("timestep must be in [0, schedule.timesteps - 1].")
    if num_trials < 2:
        raise ValueError("num_trials must be >= 2.")

    if x0.ndim == 3:
        x0 = x0.unsqueeze(0)
    if x0.ndim != 4:
        raise ValueError("x0 must have shape [B,C,H,W] or [C,H,W].")

    device = x0.device
    t = torch.full((x0.shape[0],), timestep, device=device, dtype=torch.long)

    running_sum = torch.zeros_like(x0, dtype=torch.float64)
    running_sq_sum = torch.zeros_like(x0, dtype=torch.float64)

    for _ in range(num_trials):
        eps = torch.randn_like(x0)
        xt = q_sample(x0=x0, t=t, schedule=schedule, noise=eps)
        xt64 = xt.to(torch.float64)
        running_sum += xt64
        running_sq_sum += xt64 * xt64

    empirical_mean = running_sum / float(num_trials)
    empirical_var = (running_sq_sum / float(num_trials)) - empirical_mean * empirical_mean
    empirical_var = empirical_var.clamp(min=0.0)

    sqrt_alpha_bar_t = extract(schedule.sqrt_alpha_bars, t, x0.shape).to(torch.float64)
    one_minus_alpha_bar_t = extract(1.0 - schedule.alpha_bars, t, x0.shape).to(torch.float64)
    theoretical_mean = sqrt_alpha_bar_t * x0.to(torch.float64)
    theoretical_var = one_minus_alpha_bar_t

    mean_err = torch.abs(empirical_mean - theoretical_mean)
    var_err = torch.abs(empirical_var - theoretical_var)

    return {
        "timestep": float(timestep),
        "num_trials": float(num_trials),
        "empirical_mean_global": float(empirical_mean.mean().item()),
        "theoretical_mean_global": float(theoretical_mean.mean().item()),
        "empirical_var_global": float(empirical_var.mean().item()),
        "theoretical_var_global": float(theoretical_var.mean().item()),
        "mean_abs_error_mean": float(mean_err.mean().item()),
        "mean_abs_error_var": float(var_err.mean().item()),
        "max_abs_error_mean": float(mean_err.max().item()),
        "max_abs_error_var": float(var_err.max().item()),
    }
