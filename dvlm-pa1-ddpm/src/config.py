from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_ROOT / "figures"
SAMPLES_DIR = OUTPUT_ROOT / "samples"
CHECKPOINTS_DIR = OUTPUT_ROOT / "checkpoints"

# MNIST-only setup to keep everything simple.
IMAGE_CHANNELS = 1
IMAGE_SIZE = 28
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 2e-2
SCHEDULE_TYPE = "linear"

BATCH_SIZE = 128
NUM_WORKERS = 0
MODEL_VARIANT = "deep"
MODEL_BASE_CHANNELS = 64
MODEL_TIME_DIM = 256
LEARNING_RATE = 1e-4
TRAIN_STEPS = 12_000
GRAD_CLIP_NORM = 1.0
LOG_EVERY = 100
SAMPLE_EVERY = 3000


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
