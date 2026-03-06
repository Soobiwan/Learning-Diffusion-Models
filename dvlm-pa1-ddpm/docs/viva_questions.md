# DVLM PA1 — Viva Questions Prep

## Core pipeline

### 1. What is the forward process?
The forward process gradually corrupts a clean sample by adding Gaussian noise over many timesteps. Each step transforms `x_{i-1}` into `x_i` using a Gaussian transition with controlled variance.

### 2. Why is the forward marginal closed form?
Because the forward process is linear-Gaussian. Repeated composition of Gaussian transitions yields another Gaussian, so `q(x_i | x_0)` has a closed-form mean and variance.

### 3. Why is the posterior Gaussian?
Because it comes from multiplying Gaussian terms in a linear-Gaussian model. When you combine them and complete the square, the posterior remains Gaussian.

### 4. Why predict epsilon?
Predicting epsilon makes the learning target simple and stable. It turns the objective into a mean-squared error between true and predicted noise.

### 5. How do we recover `x_0` from predicted epsilon?
From
`x_i = sqrt(bar(alpha)_i) x_0 + sqrt(1 - bar(alpha)_i) epsilon`,
we rearrange to estimate `x_0` using the predicted noise.

### 6. What is ancestral sampling?
Start from pure Gaussian noise at the final timestep and repeatedly apply the learned reverse transition until reaching timestep 0.

---

## Design choices

### 7. Why MNIST?
It is small, fast, and easy to debug. It lets me verify the full DDPM pipeline without being blocked by compute.

### 8. Why Two Moons?
It is a simple 2D dataset that makes the forward and reverse dynamics visually interpretable.

### 9. Why use timestep embeddings?
The network must know the current noise level or timestep to predict the correct denoising direction.

### 10. Why precompute schedule quantities?
To avoid repeated computation and to keep formulas numerically consistent across training and sampling.

### 11. Why use `L_simple`?
It is the standard simplified DDPM training objective and directly supervises the model to predict the added noise.

---

## Debugging

### 12. What sanity checks matter most?
- correct scaling to `[-1, 1]`
- forward noising behaves as expected
- posterior variance is positive
- tiny-subset overfit works
- reverse loop has no shape or broadcast errors

### 13. If samples look bad, what might be wrong?
- schedule bug
- indexing bug
- wrong coefficient broadcasting
- broken timestep embedding
- `predict_x0_from_eps` formula error
- insufficient training

---

## Reflection

### 14. What would I do next with more time?
- stronger model
- better ablations
- more diagnostics
- alternate schedules
- compare epsilon-prediction with x0-prediction

### 15. What is the main intuition behind diffusion models?
They learn to reverse a gradual noising process. By learning how to denoise at every noise level, the model can transform random noise into realistic data.