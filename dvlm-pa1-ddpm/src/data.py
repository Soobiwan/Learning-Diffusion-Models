from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import BATCH_SIZE, DATA_ROOT, FIGURES_DIR, NUM_WORKERS
from .utils.viz import save_image_grid


def get_mnist_transform() -> transforms.Compose:
    # ToTensor gives [0, 1], Normalize(0.5, 0.5) maps to [-1, 1].
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_mnist_dataloader(
    batch_size: int = BATCH_SIZE,
    train: bool = True,
    shuffle: bool = True,
    num_workers: int = NUM_WORKERS,
    data_root: Path = DATA_ROOT,
) -> DataLoader:
    dataset = datasets.MNIST(
        root=str(data_root),
        train=train,
        transform=get_mnist_transform(),
        download=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def task0_sanity_stats(batch: torch.Tensor) -> Dict[str, object]:
    return {
        "shape": tuple(batch.shape),
        "dtype": str(batch.dtype),
        "min": float(batch.min().item()),
        "max": float(batch.max().item()),
    }


def run_task0(
    output_path: Path = FIGURES_DIR / "task0_real_grid.png",
    batch_size: int = BATCH_SIZE,
) -> Dict[str, object]:
    loader = get_mnist_dataloader(batch_size=batch_size, train=True, shuffle=True)
    images, labels = next(iter(loader))
    stats = task0_sanity_stats(images)

    if stats["dtype"] != "torch.float32":
        raise ValueError(f"Expected float32 images, got {stats['dtype']}")
    if len(stats["shape"]) != 4 or stats["shape"][1:] != (1, 28, 28):
        raise ValueError(f"Expected shape (B,1,28,28), got {stats['shape']}")
    if stats["min"] < -1.0001 or stats["max"] > 1.0001:
        raise ValueError(f"Expected range in [-1,1], got [{stats['min']}, {stats['max']}]")

    save_image_grid(images[:64], output_path=output_path, nrow=8, source_range=(-1.0, 1.0))
    stats["labels_min"] = int(labels.min().item())
    stats["labels_max"] = int(labels.max().item())
    stats["grid_path"] = str(output_path)
    return stats


if __name__ == "__main__":
    task0 = run_task0()
    print("Task 0 complete (MNIST only).")
    print(task0)
