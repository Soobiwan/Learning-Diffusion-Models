import torch

from .posterior import p_sample_step
from .schedule import DiffusionSchedule


@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    num_samples: int,
    device: torch.device | str,
    capture_steps: list[int] | None = None,
    seed: int | None = None,
    initial_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """
    Standard ancestral DDPM sampling.
    Timesteps are indexed as 0..T-1 in code.
    """
    model.eval()
    gen_device = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    noise_generator = None
    if seed is not None:
        noise_generator = torch.Generator(device=gen_device)
        noise_generator.manual_seed(seed)

    if initial_noise is None:
        if noise_generator is None:
            x_t = torch.randn(num_samples, 1, 28, 28, device=device)
        else:
            try:
                x_t = torch.randn(num_samples, 1, 28, 28, device=device, generator=noise_generator)
            except TypeError:
                x_t = torch.randn(num_samples, 1, 28, 28, device=device)
    else:
        if initial_noise.shape != (num_samples, 1, 28, 28):
            raise ValueError("initial_noise must have shape [num_samples, 1, 28, 28].")
        x_t = initial_noise.to(device=device, dtype=torch.float32).clone()
    captured: dict[int, torch.Tensor] = {}

    wanted = set(capture_steps or [])
    for step in reversed(range(schedule.timesteps)):
        t = torch.full((num_samples,), step, device=device, dtype=torch.long)
        eps_pred = model(x_t, t)
        x_t = p_sample_step(
            x_t=x_t,
            t=t,
            eps_pred=eps_pred,
            schedule=schedule,
            noise_generator=noise_generator,
        )
        if step in wanted:
            captured[step] = x_t.detach().cpu()

    return x_t.detach().cpu(), captured
