# Barebones DDPM (MNIST Only)

This repo is intentionally simplified:
- fixed dataset: MNIST
- fixed image scaling: `[-1, 1]`
- minimal DDPM implementation for Tasks 0 to 7
- code lives in `src/` and notebooks only call that code

## Install

```bash
pip install -r requirements.txt
```

## Run Task 0

```bash
python -m src.data
```

This checks:
- shape is `(B, 1, 28, 28)`
- dtype is `float32`
- value range is `[-1, 1]`

and saves:
- `outputs/figures/task0_real_grid.png`

## Run Training (Tasks 3 to 5)

```bash
python -m src.train
```

This trains a stronger time-conditioned U-Net with `L_simple`, saves periodic sample grids, and writes:
- checkpoint: `outputs/checkpoints/tiny_unet_mnist.pt`

## Notebooks

Only three notebooks are kept:
1. `notebooks/01_tasks_0_to_2_mnist_basics.ipynb`
2. `notebooks/02_tasks_3_to_5_train_and_sample.ipynb`
3. `notebooks/03_tasks_6_to_7_diagnostics_and_ablation.ipynb`

Notebook 03 includes:
- Task 7 evaluation/reporting:
  - fixed-seed denoising trajectory
  - MNIST feature-space FID/KID
  - classifier-based quality/diversity metrics
  - nearest-neighbor memorization grid
  - train-vs-test gap checks
  - ELBO/NLL-style bpd estimate
- Task 6 focused ablation:
  - schedule ablation (linear vs cosine)
- toy 2D metric utilities (SWD + RBF-MMD + scatter)

## Source Layout

- `src/data.py` - MNIST loading and Task 0 sanity checks
- `src/diffusion/` - schedule creation (`make_beta_schedule`), forward process checks, posterior math, and DDPM sampler
- `src/models/unet.py` - stronger time-conditioned epsilon model
- `src/train.py` - simple `L_simple` training loop
- `src/eval.py` - Task 7 evaluation/reporting + Task 6 ablation helpers
