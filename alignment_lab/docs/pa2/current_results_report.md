# PA2 Current Results Report

Snapshot date: April 11, 2026

## Executive Summary

- Archived reward model is being reused from `artifacts/checkpoints/rm_hh_rlhf/final`.
- Its held-out preference accuracy is `0.5912` over `8552` pairs, which is slightly below the PA2 `>= 0.60` target.
- `SFT`, `DPO`, `PPO`, `GRPO`, `RLVR`, the DPO beta sweep, and the cross-method comparison artifacts are all present.
- On the shared HH comparison slice, the best method by RM win-rate is `GRPO` with `rm_win_rate_vs_sft = 0.6800`.
- `RLVR` did not improve over the baseline in this run: `pass@1 = 0.0000`.

## Reward Model

| Metric | Value |
| --- | --- |
| Preference accuracy | 0.5912 |
| Mean chosen score | -0.6998 |
| Mean rejected score | -0.9817 |
| Held-out pairs | 8552 |

Artifact: `artifacts/tables/rm_hh_rlhf_rm_final_eval.json`

## Per-Method Final Eval

| Method | RM score | RM win vs SFT | KL | Pref acc | Mean len | Peak VRAM (GB) | Runtime (min) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SFT | -2.4224 | 0.0000 | 0.0000 | - | 35.14 | 5.70 | 90.5 |
| DPO | -2.3325 | 0.4200 | 0.0174 | 0.5500 | 11.72 | 5.04 | 19.3 |
| PPO | -2.3239 | 0.5400 | 0.0138 | - | 19.66 | 4.37 | 68.4 |
| GRPO | -1.6772 | 0.7800 | 0.0253 | - | 21.84 | 2.31 | 44.5 |
| RLVR | - | - | 0.0000 | - | 0.00 | 1.16 | 2.5 |

Selected checkpoint notes:

- SFT best checkpoint came from step `400` with held-out perplexity `8.4716`.
- DPO best checkpoint came from step `50` with preference accuracy `0.5500`.
- PPO best checkpoint came from step `200` with RM win-rate `0.5400`.
- GRPO best checkpoint came from step `300` with RM win-rate `0.7800`.
- RLVR selected step `50` but the best `pass@1` stayed at `0.0000`.

## Shared HH Comparison

This table is the cleanest apples-to-apples ranking because every method is evaluated on the same fixed HH prompt slice.

| Method | RM score | RM win vs SFT | Pref acc | KL | Mean len | Peak VRAM (GB) | Runtime (min) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SFT | -2.3672 | 0.0000 | 0.5500 | 0.0000 | 35.30 | 5.70 | 90.5 |
| DPO | -2.2631 | 0.4500 | 0.5500 | 0.0179 | 11.36 | 5.04 | 19.3 |
| PPO | -2.3099 | 0.4000 | 0.5600 | 0.0126 | 24.04 | 4.37 | 68.4 |
| GRPO | -1.7515 | 0.6800 | 0.5300 | 0.0240 | 23.98 | 2.31 | 44.5 |

Takeaways:

- `GRPO` is strongest on the shared HH comparison slice.
- `PPO` is competitive but trails `GRPO` on RM score and win-rate.
- `DPO` improves over `SFT` but tends to produce shorter responses.
- `SFT` remains the baseline and has the lowest KL by design.

Artifact: `artifacts/tables/pa2_method_comparison_hh_summary.json`

## RLVR Comparison

| Method | Pass@1 | Format compliance | Mean len | KL | Peak VRAM (GB) | Runtime (min) |
| --- | --- | --- | --- | --- | --- | --- |
| SFT | 0.0000 | 0.0000 | 0.00 | 0.0000 | 5.70 | 90.5 |
| RLVR | 0.0000 | 0.0000 | 0.00 | 0.0000 | 1.16 | 2.5 |

RLVR verifier sanity check:

- Gold-answer extraction accuracy: `1.0000` over `20` checks
- Wrong-answer rejection rate: `1.0000` over `20` checks

Artifact: `artifacts/tables/pa2_method_comparison_rlvr_summary.json`

## DPO Beta Ablation

| Beta | RM score | RM win vs SFT | Pref acc | KL | Runtime (min) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | -2.3191 | 0.4650 | 0.5500 | 0.0504 | 4.4 |
| 0.10 | -2.3077 | 0.4700 | 0.5500 | 0.0436 | 4.4 |
| 0.50 | -2.2823 | 0.4700 | 0.5550 | 0.0263 | 4.7 |
| 1.00 | -2.3101 | 0.4400 | 0.5550 | 0.0192 | 4.7 |

Best observed tradeoff: `beta = 0.50`.

Artifact: `artifacts/tables/pa2_dpo_beta_ablation_summary.json`

## Important Caveats

- The archived reward model is useful, but it is still slightly below the PA2 `>= 0.60` preference-accuracy target.
- The scripted full pipeline log at `artifacts/run_logs/20260410_182639` stops at the original DPO OOM, but the later tables, logs, and samples show that the remaining stages were completed manually after the DPO memory fix.
- HH method ranking should be taken from the shared comparison summary, not from mixing per-method final eval tables with different prompt slices.
- `RLVR` is functionally complete but did not learn under the current compute budget; the extraction/verifier path is correct, the policy quality just did not improve.

## Judge Pack

- Shared 10-prompt unblinded pack: `artifacts/judge/pa2_hh_judge_pack_10_unblinded.json`
- Shared 10-prompt blinded pack: `artifacts/judge/pa2_hh_judge_pack_10_blinded.json`
- Judge markdown sheet: `artifacts/judge/pa2_hh_judge_pack_10.md`
- Blind answer key: `artifacts/judge/pa2_hh_judge_pack_10_key.json`
