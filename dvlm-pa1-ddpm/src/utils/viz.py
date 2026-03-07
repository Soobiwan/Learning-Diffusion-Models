from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image


def _to_display_range(
    images: torch.Tensor, source_range: Tuple[float, float]
) -> torch.Tensor:
    """Map input tensors into [0, 1] for visualization."""
    low, high = source_range
    if (low, high) == (-1.0, 1.0):
        return (images.clamp(-1.0, 1.0) + 1.0) * 0.5
    if (low, high) == (0.0, 1.0):
        return images.clamp(0.0, 1.0)
    raise ValueError(f"Unsupported source range: {source_range}")


def save_image_grid(
    images: torch.Tensor,
    output_path: Path,
    nrow: int = 8,
    source_range: Tuple[float, float] = (-1.0, 1.0),
) -> Path:
    """Save a batch of images as a grid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu()
    images = _to_display_range(images, source_range=source_range)
    save_image(images, fp=str(output_path), nrow=nrow)
    return output_path


def plot_schedule(alpha_bars: torch.Tensor, snr_values: torch.Tensor) -> None:
    steps = torch.arange(alpha_bars.shape[0]).cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps, alpha_bars.detach().cpu().numpy())
    axes[0].set_title("alpha_bar(t)")
    axes[0].set_xlabel("t")

    axes[1].plot(steps, snr_values.detach().cpu().numpy())
    axes[1].set_title("SNR(t)")
    axes[1].set_xlabel("t")
    axes[1].set_yscale("log")
    plt.tight_layout()
    plt.show()


def plot_loss_curve(losses: list[float]) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.show()


def plot_training_diagnostics(
    losses: list[float],
    grad_norms: list[float],
    param_norms: list[float],
) -> None:
    steps = range(1, len(losses) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(steps, losses)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE")

    axes[1].plot(steps, grad_norms)
    axes[1].set_title("Gradient Norm")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("L2 Norm")

    axes[2].plot(steps, param_norms)
    axes[2].set_title("Parameter Norm")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("L2 Norm")

    plt.tight_layout()
    plt.show()


def plot_timestep_histogram(counts: torch.Tensor, title: str = "Timestep Sampling Histogram") -> None:
    counts = counts.detach().cpu()
    timesteps = torch.arange(counts.shape[0])
    plt.figure(figsize=(10, 3))
    plt.bar(timesteps.numpy(), counts.numpy(), width=1.0)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()
