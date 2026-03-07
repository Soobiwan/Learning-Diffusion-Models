# Introduction To DVLMs

This repository is the parent workspace for learning Diffusion Vision Language Models from the ground up. It is structured more like a study folder than a polished library: there is one focused implementation subproject, generated artifacts, and a small amount of top-level data/output clutter that reflects actual experimentation.

If the subfolder `dvlm-pa1-ddpm/` is the implementation, this top-level README is the map.

Original references:
- Paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Original code: [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)

## What This Repo Is

The goal of this workspace is to make DVLMs feel inspectable. Each folder will hold its own mini project in trying to understand and deconstruct chapters crucial for understanding why and how DVLMs work.

This is not a general-purpose diffusion framework. It is just a repo I made and use to learn DVLMs.

## Part 1: DDPMs

  The core project lives in:

  - [`dvlm-pa1-ddpm/`](dvlm-pa1-ddpm/)

  Start there for almost everything:

  - implementation overview: [`dvlm-pa1-ddpm/README.md`](dvlm-pa1-ddpm/README.md)
  - narrative experiment notes: [`dvlm-pa1-ddpm/docs/working_doc.md`](dvlm-pa1-ddpm/docs/working_doc.md)
  - conceptual checklist: [`dvlm-pa1-ddpm/docs/things_to_understand.md`](dvlm-pa1-ddpm/docs/things_to_understand.md)
  - artifact log: [`dvlm-pa1-ddpm/run_notes.md`](dvlm-pa1-ddpm/run_notes.md)

  #### Folder Layout

  - `dvlm-pa1-ddpm/`
    The actual DDPM study project: notebooks, `src/`, checkpoints, figures, and markdown writeups.

  - `data/`
    Top-level MNIST data cache from experimentation at the repo root.

  - `outputs/`
    Top-level output spillover from early runs before everything was consolidated under `dvlm-pa1-ddpm/outputs/`.

  #### Learning Path

  If someone opens this repo cold, the intended reading order is:

  1. Read [`dvlm-pa1-ddpm/README.md`](dvlm-pa1-ddpm/README.md) for the compact project overview.
  2. Skim notebook 01 for data checks and forward-process verification.
  3. Skim notebook 02 for training, sampling, and sanity checks.
  4. Use notebook 03 for evaluation, schedule ablation, and timestep ablation.
  5. Read [`dvlm-pa1-ddpm/docs/working_doc.md`](dvlm-pa1-ddpm/docs/working_doc.md) for the more reflective writeup.
  6. Use [`dvlm-pa1-ddpm/docs/things_to_understand.md`](dvlm-pa1-ddpm/docs/things_to_understand.md) as the conceptual checklist.

  #### What The PA1 Folder Covers

  The `dvlm-pa1-ddpm/` folder currently covers:

  - MNIST preprocessing and normalization
  - forward diffusion and posterior math
  - a compact time-conditioned U-Net
  - `L_simple` training
  - ancestral DDPM sampling
  - denoising trajectories
  - classifier-feature diagnostics
  - nearest-neighbor memorization checks
  - schedule ablation
  - stronger repeated-seed timestep ablation

  That subfolder is where the actual DDPM story is told.

  #### Important Artifacts

  Representative outputs already in the repo:

  - real MNIST grid: [`dvlm-pa1-ddpm/outputs/figures/task0_real_grid.png`](dvlm-pa1-ddpm/outputs/figures/task0_real_grid.png)
  - final samples: [`dvlm-pa1-ddpm/outputs/samples/final_samples.png`](dvlm-pa1-ddpm/outputs/samples/final_samples.png)
  - denoising trajectory: [`dvlm-pa1-ddpm/outputs/samples/trajectory.png`](dvlm-pa1-ddpm/outputs/samples/trajectory.png)
  - task 7 nearest neighbors: [`dvlm-pa1-ddpm/outputs/figures/task7_nearest_neighbors.png`](dvlm-pa1-ddpm/outputs/figures/task7_nearest_neighbors.png)
  - schedule-ablation samples: [`dvlm-pa1-ddpm/outputs/samples/task6_linear_samples.png`](dvlm-pa1-ddpm/outputs/samples/task6_linear_samples.png) and [`dvlm-pa1-ddpm/outputs/samples/task6_cosine_samples.png`](dvlm-pa1-ddpm/outputs/samples/task6_cosine_samples.png)
  - repeated-seed timestep samples: for example [`dvlm-pa1-ddpm/outputs/samples/task6_timestep_t750_seed2026_samples.png`](dvlm-pa1-ddpm/outputs/samples/task6_timestep_t750_seed2026_samples.png)

  #### Things This Repo Is Trying To Make Clear

  - DDPM is not magic if each stage is inspected separately.
  - Good-looking samples are not enough; diagnostics matter.
  - Low training loss and good generation are related but not identical.
  - Ablations should be given enough compute and repeated enough times to be worth trusting.
