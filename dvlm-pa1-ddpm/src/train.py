from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .config import (
    BATCH_SIZE,
    BETA_END,
    BETA_START,
    CHECKPOINTS_DIR,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    LOG_EVERY,
    MODEL_BASE_CHANNELS,
    MODEL_TIME_DIM,
    MODEL_VARIANT,
    SAMPLE_EVERY,
    SCHEDULE_TYPE,
    SAMPLES_DIR,
    TIMESTEPS,
    TRAIN_STEPS,
    get_device,
)
from .data import get_mnist_dataloader
from .diffusion.ddpm import sample_ddpm
from .diffusion.forward import q_sample
from .diffusion.posterior import q_posterior_mean_variance
from .diffusion.schedule import DiffusionSchedule, build_schedule, sample_timesteps
from .models.unet import TinyUNet, infer_unet_config_from_state_dict
from .utils.io import save_checkpoint
from .utils.viz import save_image_grid


def _compute_grad_norm(model: nn.Module) -> float:
    total = torch.zeros(1, device=next(model.parameters()).device)
    for p in model.parameters():
        if p.grad is not None:
            total += torch.sum(p.grad.detach() ** 2)
    return float(torch.sqrt(total).item())


def _compute_param_norm(model: nn.Module) -> float:
    total = torch.zeros(1, device=next(model.parameters()).device)
    for p in model.parameters():
        total += torch.sum(p.detach() ** 2)
    return float(torch.sqrt(total).item())


def _run_training_loop(
    model: nn.Module,
    schedule: DiffusionSchedule,
    dataloader: DataLoader,
    steps: int,
    optimizer: torch.optim.Optimizer,
    sample_every: int,
    sample_prefix: str,
    grad_clip_norm: float = GRAD_CLIP_NORM,
    log_every: int = LOG_EVERY,
) -> dict[str, object]:
    device = next(model.parameters()).device
    losses: list[float] = []
    grad_norms: list[float] = []
    param_norms: list[float] = []
    step_ids: list[int] = []
    sample_paths: list[str] = []
    timestep_counts = torch.zeros(schedule.timesteps, dtype=torch.long)

    loader_iter = iter(dataloader)
    model.train()
    progress = tqdm(range(1, steps + 1), total=steps, desc="Task 4 Training")

    for step in progress:
        try:
            x0, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            x0, _ = next(loader_iter)

        x0 = x0.to(device)
        t = sample_timesteps(x0.shape[0], schedule.timesteps, device=device)
        timestep_counts += torch.bincount(t.detach().cpu(), minlength=schedule.timesteps)

        noise = torch.randn_like(x0)
        x_t = q_sample(x0=x0, t=t, schedule=schedule, noise=noise)
        noise_pred = model(x_t, t)

        loss = torch.mean((noise - noise_pred) ** 2)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        grad_norm = _compute_grad_norm(model)
        optimizer.step()
        param_norm = _compute_param_norm(model)

        losses.append(float(loss.item()))
        grad_norms.append(grad_norm)
        param_norms.append(param_norm)
        step_ids.append(step)

        if step % log_every == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}", grad=f"{grad_norm:.3f}", param=f"{param_norm:.1f}")

        if sample_every > 0 and step % sample_every == 0:
            samples, _ = sample_ddpm(
                model=model,
                schedule=schedule,
                num_samples=64,
                device=device,
                capture_steps=None,
            )
            out_path = SAMPLES_DIR / f"{sample_prefix}_step_{step:06d}.png"
            save_image_grid(samples, output_path=out_path, nrow=8, source_range=(-1.0, 1.0))
            sample_paths.append(str(out_path))

    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "param_norms": param_norms,
        "steps": step_ids,
        "timestep_counts": timestep_counts,
        "sample_paths": sample_paths,
    }


def train_ddpm(
    steps: int = TRAIN_STEPS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    timesteps: int = TIMESTEPS,
    unet_variant: str = MODEL_VARIANT,
    unet_time_dim: int = MODEL_TIME_DIM,
    unet_base_channels: int = MODEL_BASE_CHANNELS,
    schedule_type: str = SCHEDULE_TYPE,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
    sample_every: int = SAMPLE_EVERY,
    checkpoint_path: Path = CHECKPOINTS_DIR / "tiny_unet_mnist.pt",
    return_stats: bool = False,
) -> tuple[nn.Module, DiffusionSchedule, list[float]] | tuple[nn.Module, DiffusionSchedule, list[float], dict[str, object]]:
    device = get_device()
    dataloader = get_mnist_dataloader(batch_size=batch_size, train=True, shuffle=True)

    schedule = build_schedule(
        timesteps=timesteps,
        schedule_type=schedule_type,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
    )
    model = TinyUNet(
        variant=unet_variant,  # type: ignore[arg-type]
        time_dim=unet_time_dim,
        base_channels=unet_base_channels,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    stats = _run_training_loop(
        model=model,
        schedule=schedule,
        dataloader=dataloader,
        steps=steps,
        optimizer=optimizer,
        sample_every=sample_every,
        sample_prefix="train",
        grad_clip_norm=GRAD_CLIP_NORM,
    )
    losses: list[float] = stats["losses"]  # type: ignore[assignment]

    saved_path = save_checkpoint(model=model, path=checkpoint_path)
    stats["checkpoint_path"] = str(saved_path)
    if return_stats:
        return model, schedule, losses, stats
    return model, schedule, losses


def train_ddpm_overfit_subset(
    subset_size: int = 256,
    steps: int = 2000,
    batch_size: int = 64,
    lr: float = LEARNING_RATE,
    timesteps: int = TIMESTEPS,
    unet_variant: str = MODEL_VARIANT,
    unet_time_dim: int = MODEL_TIME_DIM,
    unet_base_channels: int = MODEL_BASE_CHANNELS,
    schedule_type: str = SCHEDULE_TYPE,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
    sample_every: int = 500,
    checkpoint_path: Path = CHECKPOINTS_DIR / "tiny_unet_mnist_overfit256.pt",
) -> tuple[nn.Module, DiffusionSchedule, list[float], dict[str, object]]:
    device = get_device()
    full_loader = get_mnist_dataloader(batch_size=batch_size, train=True, shuffle=True)
    base_dataset = full_loader.dataset
    subset_indices = list(range(min(subset_size, len(base_dataset))))
    subset_dataset = Subset(base_dataset, subset_indices)
    overfit_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    schedule = build_schedule(
        timesteps=timesteps,
        schedule_type=schedule_type,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
    )
    model = TinyUNet(
        variant=unet_variant,  # type: ignore[arg-type]
        time_dim=unet_time_dim,
        base_channels=unet_base_channels,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    stats = _run_training_loop(
        model=model,
        schedule=schedule,
        dataloader=overfit_loader,
        steps=steps,
        optimizer=optimizer,
        sample_every=sample_every,
        sample_prefix="overfit256",
        grad_clip_norm=GRAD_CLIP_NORM,
    )
    losses: list[float] = stats["losses"]  # type: ignore[assignment]

    saved_path = save_checkpoint(model=model, path=checkpoint_path)
    stats["checkpoint_path"] = str(saved_path)
    return model, schedule, losses, stats


def load_ddpm_checkpoint(
    checkpoint_path: Path = CHECKPOINTS_DIR / "tiny_unet_mnist.pt",
    timesteps: int = TIMESTEPS,
    schedule_type: str = SCHEDULE_TYPE,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
) -> tuple[nn.Module, DiffusionSchedule]:
    device = get_device()
    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        unet_cfg = infer_unet_config_from_state_dict(state_dict)
    except Exception as exc:
        raise RuntimeError(
            f"Could not infer a supported deep TinyUNet configuration from checkpoint: {exc}"
        ) from exc

    model = TinyUNet(
        variant=unet_cfg["variant"],
        time_dim=unet_cfg["time_dim"],
        base_channels=unet_cfg["base_channels"],
    ).to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. Re-train with current model settings "
            "(set RUN_TASK4_TRAIN=True) or load a checkpoint generated by this code version."
        ) from exc
    model.eval()

    schedule = build_schedule(
        timesteps=timesteps,
        schedule_type=schedule_type,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
    )
    return model, schedule


@torch.no_grad()
def one_step_posterior_sanity_check(
    schedule: DiffusionSchedule,
    batch_size: int = 64,
    timestep: int = 200,
    trials: int = 50,
    train: bool = False,
) -> dict[str, float | bool]:
    if timestep < 1:
        raise ValueError("Use timestep >= 1 for i >= 2 in 1-based notation.")
    if timestep >= schedule.timesteps:
        raise ValueError("timestep must be < schedule.timesteps.")

    device = schedule.betas.device
    loader = get_mnist_dataloader(batch_size=batch_size, train=train, shuffle=True)

    mse_xt_list = []
    mse_prev_list = []

    for i, (x0, _) in enumerate(loader):
        if i >= trials:
            break
        x0 = x0.to(device)
        t = torch.full((x0.shape[0],), timestep, device=device, dtype=torch.long)
        eps = torch.randn_like(x0)
        xt = q_sample(x0=x0, t=t, schedule=schedule, noise=eps)
        mean, var = q_posterior_mean_variance(x0=x0, x_t=xt, t=t, schedule=schedule)
        z = torch.randn_like(x0)
        x_prev = mean + torch.sqrt(var) * z

        mse_xt = torch.mean((xt - x0) ** 2).item()
        mse_prev = torch.mean((x_prev - x0) ** 2).item()
        mse_xt_list.append(mse_xt)
        mse_prev_list.append(mse_prev)

    mse_xt_mean = float(torch.tensor(mse_xt_list).mean().item())
    mse_prev_mean = float(torch.tensor(mse_prev_list).mean().item())
    return {
        "timestep": float(timestep),
        "trials": float(len(mse_xt_list)),
        "mse_xt_to_x0": mse_xt_mean,
        "mse_xprev_to_x0": mse_prev_mean,
        "condition_holds": bool(mse_prev_mean < mse_xt_mean),
    }


@torch.no_grad()
def noise_prediction_sanity_check(
    model: nn.Module,
    schedule: DiffusionSchedule,
    batch_size: int = 256,
    timestep: int = 200,
    train: bool = False,
) -> dict[str, float]:
    device = next(model.parameters()).device
    loader = get_mnist_dataloader(batch_size=batch_size, train=train, shuffle=True)
    x0, _ = next(iter(loader))
    x0 = x0.to(device)

    t = torch.full((x0.shape[0],), timestep, device=device, dtype=torch.long)
    eps = torch.randn_like(x0)
    xt = q_sample(x0=x0, t=t, schedule=schedule, noise=eps)
    eps_hat = model(xt, t)

    e = eps.flatten()
    h = eps_hat.flatten()
    e_center = e - e.mean()
    h_center = h - h.mean()
    corr = torch.sum(e_center * h_center) / (
        torch.sqrt(torch.sum(e_center**2) + 1e-12) * torch.sqrt(torch.sum(h_center**2) + 1e-12)
    )
    mse = torch.mean((eps - eps_hat) ** 2).item()

    return {
        "timestep": float(timestep),
        "noise_corr": float(corr.item()),
        "noise_mse": float(mse),
    }


def timestep_uniformity_sanity_check(
    schedule: DiffusionSchedule,
    num_draws: int = 200_000,
    batch_size: int = 1024,
) -> dict[str, object]:
    device = schedule.betas.device
    counts = torch.zeros(schedule.timesteps, dtype=torch.long)
    draws_done = 0

    while draws_done < num_draws:
        n = min(batch_size, num_draws - draws_done)
        t = sample_timesteps(n, schedule.timesteps, device=device)
        counts += torch.bincount(t.detach().cpu(), minlength=schedule.timesteps)
        draws_done += n

    freq = counts.float() / counts.sum().clamp(min=1)
    expected = torch.full_like(freq, 1.0 / schedule.timesteps)
    max_abs_err = torch.max(torch.abs(freq - expected)).item()
    rel_std = torch.std((freq / expected) - 1.0).item()

    return {
        "counts": counts,
        "freq": freq,
        "expected_freq": expected,
        "max_abs_freq_error": float(max_abs_err),
        "relative_std_error": float(rel_std),
    }


if __name__ == "__main__":
    train_ddpm()
