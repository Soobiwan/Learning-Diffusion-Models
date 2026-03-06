# Diffusion Models from First Principles

This repository contains my implementation, notes, and experiments on denoising diffusion probabilistic models (DDPMs).

## Goal
Build a minimum functional DDPM from first principles, understand the theory clearly, and prepare for the viva and written test.

## Chosen setup
- Main image dataset: MNIST
- Toy dataset: Two Moons
- Diffusion timesteps: 1000
- Beta schedule: linear from `1e-4` to `2e-2`
- Model target: small U-Net with timestep conditioning

## Repository structure

- `notebooks/`  
  Jupyter notebooks for planning, derivations, sanity checks, training, sampling, and ablations.

- `src/`  
  Reusable implementation code for diffusion math, models, data loading, training, and evaluation.

- `docs/`  
  Working document, viva prep notes, and saved figures for explanation and reporting.

- `outputs/`  
  Checkpoints, plots, generated samples, and logs.

- `data/`  
  Raw and processed data.

## Notebook plan

- `00_assignment_plan.ipynb` — overall setup and roadmap
- `01_analytical_notes.ipynb` — derivations and theory notes
- `02_data_and_schedule.ipynb` — dataset loading, scaling, schedule plots
- `03_forward_posterior_checks.ipynb` — forward process and posterior sanity checks
- `04_model_and_training.ipynb` — epsilon model and training loop
- `05_sampling_and_ablation.ipynb` — ancestral sampling and one ablation
- `scratch.ipynb` — temporary experimentation

## Source code plan

- `src/diffusion/schedule.py` — beta schedule and precomputed coefficients
- `src/diffusion/forward.py` — forward noising process
- `src/diffusion/posterior.py` — posterior formulas and x0 recovery
- `src/diffusion/ddpm.py` — reverse-step sampling and ancestral sampling
- `src/models/unet.py` — timestep-conditioned epsilon network
- `src/train.py` — training loop
- `src/eval.py` — sampling, plotting, and evaluation helpers

## How I will work
1. Finish the analytical derivations in rough-but-correct form.
2. Implement the diffusion math first.
3. Verify numerical sanity checks before training.
4. Train a minimal model on MNIST.
5. Produce sample grids and denoising trajectories.
6. Prepare concise viva explanations.

## Notes
This repo is organized for speed and clarity:
- theory is documented in markdown and notebooks,
- implementation lives in `src/`,
- notebooks call reusable functions instead of holding all logic inline.