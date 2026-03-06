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
- 

**Notes**
- 

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
- noisy image progression.

**Figures**
- 

**Notes**
- 

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
- 

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
- 

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

**Figures**
- 

**Notes**
- 

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
- 

---

### Task 6 — Diagnostics
**Goal**
- produce basic diagnostics

**Expected artifacts**
- loss curve,
- sample grid,
- denoising trajectory,
- maybe nearest-neighbor check.

**Figures**
- 

**Notes**
- 

---

### Task 7 — Ablation
**Chosen ablation**
- 

**What changes**
- 

**Result**
- 

**Interpretation**
- 

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