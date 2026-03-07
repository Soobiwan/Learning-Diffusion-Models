# Things To Understand

This is the checklist I would want in front of me if someone asked whether I really understand what this repo is doing.

## Core DDPM mechanics

- Explain the forward process in words and then write down `q(x_t | x_{t-1})`.
- Explain why `q(x_t | x_0)` has a closed form.
- Explain the difference between `beta_t`, `alpha_t`, and `alpha_bar_t`.
- Explain why the reverse process is learned instead of computed exactly.
- Explain why predicting noise is a sensible training target.
- Explain how a noise prediction becomes a mean for `p_theta(x_{t-1} | x_t)`.
- Explain why timestep embeddings are necessary.

## Indexing and implementation details

- Be able to say clearly whether the code is using timesteps indexed from `0..T-1` or `1..T`.
- Explain where an off-by-one bug would most likely show up first.
- Explain why coefficient extraction has to broadcast to image shape.
- Explain why data range consistency matters from preprocessing through visualization.
- Explain why `alpha_bar_t` should usually be precomputed instead of recomputed ad hoc.

## Training objective

- Explain what `L_simple` is actually minimizing.
- Explain what information is ignored when using the simplified objective instead of the full variational bound.
- Explain why lower training MSE does not always imply better samples.
- Explain why uniform timestep sampling changes the effective weighting over learning signal.
- Explain what gradient clipping is protecting against here.

## Reverse sampling intuition

- Explain why the model starts from Gaussian noise.
- Explain why reverse sampling is stochastic in DDPM.
- Explain what the last few denoising steps are responsible for visually.
- Explain how errors can accumulate across many reverse steps.
- Explain why sampling is much slower than a one-shot generator.

## Schedules and timesteps

- Explain what a linear beta schedule is doing.
- Explain what a cosine schedule is trying to preserve differently.
- Explain why schedule choice changes optimization behavior.
- Explain how changing the number of timesteps affects both train-time difficulty and sample-time cost.
- Explain why a shorter chain can hurt quality even when it sounds computationally attractive.
- Explain why a timestep ablation should usually be repeated across multiple seeds before drawing conclusions.

## Evaluation and diagnostics

- Explain what the nearest-neighbor grid is checking and what it cannot prove.
- Explain why classifier confidence is only a proxy for quality.
- Explain why class entropy is a diversity clue, not a full diversity measurement.
- Explain why train-vs-test feature distances may hint at memorization.
- Explain why FID/KID computed from a small custom MNIST classifier should be treated carefully.
- Explain what BPD is trying to approximate and why the estimate here is rough.

## Repo-specific understanding

- Be able to walk through `src/diffusion/schedule.py` without hand-waving.
- Be able to describe what `q_sample` does from memory.
- Be able to describe the role of `p_mean_from_eps` in evaluation/sampling.
- Be able to explain what the sanity checks in `src/train.py` are guarding against.
- Be able to explain why notebook 03 now uses repeated-seed timestep ablations and loss-history plots instead of one-shot comparisons.
- Be able to explain why notebook 03 mixes qualitative plots with approximate metrics.

## Bigger picture questions

- If this works on MNIST, what assumptions stop it from transferring cleanly to harder datasets?
- Which parts of this repo are DDPM-specific, and which parts are common to many diffusion models?
- If I wanted faster sampling, what family of ideas would I look at next?
- If I wanted better likelihoods or better sample quality, which design choices would I revisit first?
- If I wanted to make this conditional, where would class information enter the pipeline?
