# Run Notes

Short version: this repo is a learning log with saved artifacts.

## Main artifacts

- `outputs/figures/task0_real_grid.png`: confirms the MNIST pipeline and normalization are sane.
- `outputs/samples/train_step_*.png`: useful for seeing whether the model is learning structure or just denoising blobs.
- `outputs/samples/final_samples.png`: the basic "can it generate digits?" checkpoint.
- `outputs/samples/trajectory.png`: fixed-seed reverse trajectory for debugging the denoising chain.
- `outputs/figures/task7_loss_curve.png`: rough sense of optimization progress.
- `outputs/figures/task7_nearest_neighbors.png`: first-stop memorization sanity check.
- notebook 03 now also plots schedule-ablation loss curves and timestep-ablation mean/std summaries directly inside the notebook.

## Ablations saved in the repo

- Schedule comparison:
  `outputs/samples/task6_linear_samples.png`
  `outputs/samples/task6_cosine_samples.png`
  `outputs/figures/task6_linear_nearest_neighbors.png`
  `outputs/figures/task6_cosine_nearest_neighbors.png`
  plus an in-notebook loss-curve comparison using the saved `loss_history` from each scheduler run

- Timestep-count comparison:
  repeated across seeds now, with artifacts like:
  `outputs/samples/task6_timestep_t1000_seed2026_samples.png`
  `outputs/samples/task6_timestep_t750_seed2026_samples.png`
  `outputs/samples/task6_timestep_t500_seed2026_samples.png`
  `outputs/samples/task6_timestep_t250_seed2026_samples.png`
  and matching nearest-neighbor grids in `outputs/figures/`

## Numbers I would mention out loud

From the current notebook outputs:

- forward-process mean/variance check at `t=300` matches theory closely
- posterior one-step sanity check at `t=200` moves samples slightly closer to `x_0`
- noise prediction correlation at `t=200` is about `0.981`
- Task 7 small-run classifier test accuracy is about `0.974`
- Task 7 small-run feature FID to test data is about `40.39`
- the schedule comparison is more useful now because notebook 03 shows both metrics and loss-curve behavior
- the timestep comparison is more trustworthy now because it aggregates across fixed seeds instead of depending on one run

## Things I would not overclaim

- these metrics are not meant as benchmark results
- MNIST is too simple to make strong generative-model claims
- some evaluations are intentionally approximate because the point was to understand the pipeline, not to build a publication-grade evaluation suite
