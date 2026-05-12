import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def make_scaler() -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

