# DVLM PA1 — Working Document

## Assignment Scope
- Main image dataset: MNIST
- Toy dataset: Two Moons
- Goal: minimum functional DDPM with solid theory understanding for viva and written test
- Strategy: prioritize correctness, sanity checks, and explainability over polish

---

## What “Done” Means
By the end, I want:
- rough but correct analytical derivations,
- a working forward process,
- correct posterior formulas,
- a timestep-conditioned epsilon network,
- a training loop using `L_simple`,
- ancestral sampling,
- sample grids and denoising trajectories,
- one simple ablation,
- viva-ready explanations.

---

## Analytical Section

### Problem 1 — Forward process
**Target**
- Derive the closed-form marginal `q(x_i | x_0)`

**Key formulas**
- `q(x_i | x_{i-1}) = N(sqrt(alpha_i) x_{i-1}, (1 - alpha_i) I)`
- `alpha_i = 1 - beta_i`
- `bar(alpha)_i = prod_{s=1}^i alpha_s`
- `q(x_i | x_0) = N(sqrt(bar(alpha)_i) x_0, (1 - bar(alpha)_i) I)`

**To explain in viva**
- why repeated Gaussian corruption stays Gaussian,
- what `bar(alpha)_i` means,
- why later timesteps are noisier.

**Notes**
- 

---

### Problem 2 — Gaussian posterior
**Target**
- Derive `q(x_{i-1} | x_i, x_0)`

**Key formulas**
- posterior should be Gaussian,
- mean depends on both `x_i` and `x_0`,
- variance is the closed-form posterior variance.

**To explain in viva**
- why the posterior is Gaussian,
- where completing the square appears,
- why posterior variance is smaller than forward variance.

**Notes**
- 

---

### Problem 3 — ELBO
**Target**
- Explain how the diffusion objective comes from a variational lower bound

**To explain in viva**
- reconstruction term,
- KL terms,
- prior matching term,
- why training is broken into timestep-wise pieces.

**Notes**
- 

---

### Problem 4 — Epsilon parameterization
**Target**
- derive epsilon-prediction and `L_simple`

**Key formulas**
- `x_i = sqrt(bar(alpha)_i) x_0 + sqrt(1 - bar(alpha)_i) epsilon`
- predict `epsilon_theta(x_i, i)`
- training loss is MSE between true and predicted noise

**To explain in viva**
- why predicting epsilon is convenient,
- how `x_0` can be recovered from epsilon,
- why this simplifies training.

**Notes**
- 

---

### Problem 5 — Tweedie identity
**Target**
- connect denoising and score information

**To explain in viva**
- why conditional expectation gives a denoiser,
- how denoising reveals gradient information.

**Notes**
- 

---

### Problem 6 — DDPM and score matching
**Target**
- connect epsilon prediction to score estimation

**To explain in viva**
- effective score from epsilon predictor,
- role of noise scale,
- why diffusion training resembles score matching.

**Notes**
- 

---

## Coding Section

### Task 0 — Dataset and scaling
**Goal**
- load MNIST,
- scale images to `[-1, 1]`,
- visualize real samples.

**Checks**
- tensor shape,
- min/max values,
- batch visualization.

**Figures**
- `outputs/figures/task0_real_grid.png`

**Notes**
- Run command: `python -m src.data`
- Hardcoded to MNIST only, scaled to `[-1, 1]`, then saves `outputs/figures/task0_real_grid.png`.

---

### Task 1 — Schedule and forward process
**Goal**
- implement beta schedule,
- precompute diffusion coefficients,
- implement forward noising.

**Checks**
- schedule plot,
- `bar(alpha)` plot,
- SNR plot,
- noisy image progression,
- empirical forward mean/variance check against Eq. (2).

**Figures**
- 

**Notes**
- Implemented in `src/diffusion/schedule.py` and `src/diffusion/forward.py`.
- Notebook: `notebooks/01_tasks_0_to_2_mnist_basics.ipynb`.
- Added `forward_moment_sanity_check(...)` for the required empirical verification.

---

### Task 2 — Posterior quantities
**Goal**
- implement posterior mean and variance,
- implement `predict_x0_from_eps`

**Checks**
- positivity of posterior variance,
- confirm expected variance relation,
- numerical sanity check.

**Figures**
- 

**Notes**
- Implemented in `src/diffusion/posterior.py`.
- Notebook: `notebooks/01_tasks_0_to_2_mnist_basics.ipynb`.

---

### Task 3 — Epsilon network
**Goal**
- build small timestep-conditioned network

**Architecture notes**
- sinusoidal time embedding,
- small U-Net,
- GroupNorm + SiLU,
- minimal complexity first.

**Notes**
- Implemented minimal time-conditioned U-Net in `src/models/unet.py`.
- Notebook: `notebooks/02_tasks_3_to_5_train_and_sample.ipynb`.

---

### Task 4 — Training
**Goal**
- train with `L_simple`

**Hyperparameters**
- batch size:
- lr:
- optimizer:
- timesteps:
- epochs / steps:

**Checks**
- overfit tiny subset,
- loss decreases,
- epsilon prediction behaves sensibly.
- gradient norm / parameter norm tracking,
- timestep histogram sanity.

**Figures**
- 

**Notes**
- Implemented `L_simple` training loop in `src/train.py`.
- Notebook: `notebooks/02_tasks_3_to_5_train_and_sample.ipynb`.
- Added: `train_ddpm(..., return_stats=True)` for loss + grad norm + parameter norm + timestep counts.
- Task-4 checkpoint is saved to `outputs/checkpoints/tiny_unet_mnist.pt` and can be reloaded for checks/sampling.
- Added overfit test: `train_ddpm_overfit_subset(subset_size=256, ...)`.
- Added sanity checks in `src/train.py`:
  - `one_step_posterior_sanity_check`
  - `noise_prediction_sanity_check`
  - `timestep_uniformity_sanity_check`

---

### Task 5 — Sampling
**Goal**
- ancestral sampling from noise to image

**Checks**
- reverse loop runs without shape errors,
- sample grid looks structured,
- denoising trajectory improves gradually.

**Figures**
- 

**Notes**
- Implemented ancestral sampler in `src/diffusion/ddpm.py` with helpers in `src/eval.py`.
- Notebook: `notebooks/02_tasks_3_to_5_train_and_sample.ipynb`.
- Periodic sample grids during training are saved from `src/train.py` (`train_step_*.png` and `overfit256_step_*.png`).
- Task-5 notebook flow reloads the Task-4 checkpoint, runs sampling, and saves a Task-5 checkpoint (`outputs/checkpoints/tiny_unet_mnist_task5.pt`).

---

### Task 6 — Focused ablation
**Chosen ablation**
- schedule ablation: linear vs cosine schedule type

**What changes**
- baseline: `schedule_type = "linear"`
- variant: `schedule_type = "cosine"`
- everything else fixed (model, optimizer, steps, dataset, beta range)

**Result**
- generated sample grids are saved for both variants
- quantitative metrics are logged (feature FID/KID, classifier metrics, simple distribution metrics)

**Interpretation**
- this isolates schedule-shape effects without changing network or optimizer.
- implemented in `src/eval.py` as `run_task6_ablation`.

---

### Task 7 — Evaluation and reporting
**Goal**
- produce qualitative artifacts + quantitative quality/diversity/memorization metrics

**Expected artifacts**
- final sample grid (>= 64),
- fixed-seed denoising trajectory,
- nearest-neighbor memorization grid.

**Evaluation coverage**
- MNIST feature FID/KID
- classifier quality/diversity metrics
- train-vs-test FID/KID gap
- ELBO/NLL-style bpd estimate
- nearest-neighbor overfit check with a large train-reference pool

**Notes**
- Implemented in `src/eval.py` as `run_task7_diagnostics` and `run_task7_full_evaluation`.
- Feature extractor comparability is enforced via a persistent classifier checkpoint.
- Notebook: `notebooks/03_tasks_6_to_7_diagnostics_and_ablation.ipynb`.

---

## Bugs / Debugging Log

### Entry 1
- Date:
- Issue:
- Symptom:
- Cause:
- Fix:
- Lesson:

### Entry 2
- Date:
- Issue:
- Symptom:
- Cause:
- Fix:
- Lesson:

---

## Viva Questions

### Theory
- What is the forward diffusion process?
- Why is `q(x_i | x_0)` Gaussian?
- Why is the posterior Gaussian?
- Why predict epsilon instead of `x_0` directly?
- How does epsilon prediction relate to score estimation?

### Implementation
- Why scale images to `[-1, 1]`?
- Why sample timesteps uniformly?
- Why use timestep embeddings?
- Why do we not add noise at the final reverse step?
- Why do we need precomputed coefficients?

### Reflection
- What was the hardest part to debug?
- What would I improve with more time?
- Why did I choose MNIST and Two Moons?
