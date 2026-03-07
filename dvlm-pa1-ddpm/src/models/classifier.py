import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMNISTClassifier(nn.Module):
    """
    Small CNN for MNIST.
    Returns logits, and optionally penultimate-layer features.
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 14 * 14, feature_dim)
        self.fc2 = nn.Linear(feature_dim, 10)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = F.silu(self.conv1(x))
        x = self.pool(F.silu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        features = F.silu(self.fc1(x))
        logits = self.fc2(features)

        if return_features:
            return logits, features
        return logits
