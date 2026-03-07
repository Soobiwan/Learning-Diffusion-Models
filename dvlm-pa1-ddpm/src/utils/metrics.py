import torch
import torch.nn.functional as F


def mse_noise_loss(noise_pred: torch.Tensor, noise_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(noise_pred, noise_true)
