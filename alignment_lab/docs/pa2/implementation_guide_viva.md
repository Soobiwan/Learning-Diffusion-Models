# PA2 Implementation Guide and Viva Notes

This is the main PA2 coding guide for the repository. It explains what is implemented, how the parts connect, what the important artifacts are, what caveats matter in practice, and how to answer common coding-section questions directly from the codebase.

If you need the requirement-by-requirement matrix first, start with [`docs/pa2/implementation_status.md`](./implementation_status.md).

## 1. What the PA2 path looks like in this repo

The PA2 workflow is implemented as a staged pipeline:

1. Load and canonicalize data through a dataset adapter.
2. Reuse the archived trained reward model checkpoint.
3. Train the SFT warm-start model.
4. Run a setup audit on the PPO stack to verify parsing, tokenizer settings, checkpoint-backed model counts, and memory state.
5. Train DPO, PPO, and GRPO from the SFT checkpoint.
6. Train RLVR on GSM8K from the same SFT checkpoint.
7. Run the DPO `beta` ablation sweep.
8. Aggregate everything with the PA2 comparison command.

The code paths are:

- Data adapters: [`src/alignlab/data/adapters/`](../../src/alignlab/data/adapters/)
- Collators and canonical schemas: [`src/alignlab/data/collators.py`](../../src/alignlab/data/collators.py), [`src/alignlab/data/schemas.py`](../../src/alignlab/data/schemas.py)
- Model loading and tokenizer normalization: [`src/alignlab/models/`](../../src/alignlab/models/)
- Training CLIs: [`src/alignlab/cli/`](../../src/alignlab/cli/)
- Trainers and objectives: [`src/alignlab/trainers/`](../../src/alignlab/trainers/), [`src/alignlab/objectives/`](../../src/alignlab/objectives/)
- Evaluation and reports: [`src/alignlab/eval/`](../../src/alignlab/eval/)
- PA2-only helpers: [`src/alignlab/eval/pa2_tools.py`](../../src/alignlab/eval/pa2_tools.py), [`src/alignlab/cli/setup_audit.py`](../../src/alignlab/cli/setup_audit.py), [`src/alignlab/cli/compare_pa2.py`](../../src/alignlab/cli/compare_pa2.py)

The final config namespace is the `pa2_*` family under [`configs/experiment/`](../../configs/experiment/). Those configs are the intended submission-facing runs.

## 1.1 Runtime logging

There are now two logging layers for PA2 runs:

- Structured experiment metrics are written by the Python CLIs to `artifacts/logs/<experiment>.jsonl`.
- End-to-end command output is written by [`scripts/run_pa2_pipeline.sh`](../../scripts/run_pa2_pipeline.sh) to `artifacts/run_logs/<timestamp>/`.

Each run-log directory contains:

- `manifest.txt` with the selected mode, Python path, and `PYTORCH_CUDA_ALLOC_CONF`
- one `*.log` file per stage, with stdout and stderr merged and preserved in order

This separation is useful in practice:

- the JSONL logs are best for plotting and metric extraction
- the stage logs are best for debugging crashes, downloads, warnings, and exact terminal output

## 1.2 Budget policy

The PA2 PDF explicitly says the recommended baseline configuration may be adjusted according to compute and memory budget. The current `pa2_*` configs use that allowance.

The main runtime choices are:

- reuse the archived fully trained `rm_hh_rlhf` checkpoint instead of retraining RM in the default PA2 runner
- fixed subset caps for SFT, DPO, PPO, and GRPO on HH-RLHF, plus a smaller fixed GSM8K subset for RLVR
- reduced online rollout budgets so PPO, GRPO, and RLVR fit a 10.7 GB GPU
- capped offline and online runs with explicit `max_steps`
- reduced periodic training-time evaluation frequency
- save-best checkpoint selection and patience-based early stopping on SFT, DPO, PPO, GRPO, and RLVR
- kept the final cross-method comparison at 200 prompts so the canonical report remains reasonably sized

In other words, training-time eval is intentionally cheaper than the final comparison pass.

## 2. End-to-end data flow

### 2.1 Canonical schemas

The repo reduces each task to a small number of canonical dataclasses in [`src/alignlab/data/schemas.py`](../../src/alignlab/data/schemas.py):

- `PreferenceExample` for HH-RLHF style chosen/rejected pairs
- `SFTExample` for prompt plus target response
- `VerifiableExample` for prompt plus gold answer, used by RLVR

The reason this matters in a viva: once data is canonicalized, almost all later code becomes dataset-agnostic.

### 2.2 HH-RLHF path

HH-RLHF rows are converted into a shared prompt with chosen and rejected continuations in [`src/alignlab/data/adapters/hh_rlhf.py`](../../src/alignlab/data/adapters/hh_rlhf.py). That same canonical format then feeds:

- RM training
- SFT warm-up by taking the `chosen` response
- DPO pairwise training
- PPO/GRPO prompt-only rollout collection
- HH evaluation

### 2.3 GSM8K path

GSM8K rows are converted into `VerifiableExample` objects in [`src/alignlab/data/adapters/gsm8k.py`](../../src/alignlab/data/adapters/gsm8k.py). The adapter now does three important PA2-specific things:

- extracts the gold numeric answer
- truncates the question body to 200 whitespace tokens before templating
- formats the prompt with the repo’s GSM8K template

That makes the RLVR path reproducible and assignment-aligned in code rather than by convention.

## 3. Setup audit and model-loading story

The PA2-specific setup audit is [`src/alignlab/cli/setup_audit.py`](../../src/alignlab/cli/setup_audit.py).

What it does:

- loads the selected config
- loads training examples and prints three canonical preview rows
- loads each relevant model section present in the config
- records tokenizer padding side and special-token ids
- records total and trainable parameter counts
- records CUDA allocation, reservation, peak allocation, total VRAM, and a simple fit check
- writes the result to `artifacts/tables/<experiment>_setup_audit.json`

Why this exists:

- C0 in PA2 explicitly asks for visible evidence that parsing and model loading are correct before training
- it gives one reproducible command instead of asking you to inspect internals manually

The tokenizer and special-token normalization path lives in [`src/alignlab/models/tokenizer_utils.py`](../../src/alignlab/models/tokenizer_utils.py). That file is important because a lot of silent generation bugs come from mismatched `pad_token_id` and `eos_token_id`.

### 3.1 Reference, reward, and value models

The repo keeps the policy, reference, reward, and value roles separate:

- Trainable policy bundle: [`src/alignlab/models/policy.py`](../../src/alignlab/models/policy.py)
- Frozen reference bundle: [`src/alignlab/models/reference.py`](../../src/alignlab/models/reference.py)
- Reward-model bundle: [`src/alignlab/models/reward.py`](../../src/alignlab/models/reward.py)
- Value-model bundle: [`src/alignlab/models/value.py`](../../src/alignlab/models/value.py)

That separation makes the offline and online methods shareable without TRL-style trainer-specific coupling.

## 4. How each method is implemented

## 4.1 Reward model

The reward-model training entry point is [`src/alignlab/cli/train_rm.py`](../../src/alignlab/cli/train_rm.py), but the default PA2 execution path now reuses the archived completed RM checkpoint instead of retraining it.

Implementation notes:

- The backbone is loaded as `AutoModelForSequenceClassification`.
- Pairwise chosen/rejected batches come from [`RewardModelCollator`](../../src/alignlab/data/collators.py).
- The loss is implemented in [`src/alignlab/objectives/reward_model.py`](../../src/alignlab/objectives/reward_model.py).
- The trainer is [`src/alignlab/trainers/rm_trainer.py`](../../src/alignlab/trainers/rm_trainer.py).
- Final evaluation writes preference accuracy, mean chosen/rejected score, score CSVs, and a histogram through [`src/alignlab/eval/pipeline.py`](../../src/alignlab/eval/pipeline.py).

Archived RM evidence reused by PA2:

- checkpoint: `artifacts/checkpoints/rm_hh_rlhf/final`
- held-out eval: `artifacts/tables/rm_hh_rlhf_rm_final_eval.json`
- held-out preference accuracy: `0.5912067294` on `8552` pairs

Rationale:

- the archived RM is a completed long-run artifact and is already good enough to score SFT/DPO/PPO/GRPO generations consistently
- retraining RM was the single biggest wall-clock bottleneck in the default pipeline on the current hardware
- the `pa2_*` policy configs still consume the RM through the PA2 alias [`configs/model/pa2_rm_hh_rlhf_checkpoint.yaml`](../../configs/model/pa2_rm_hh_rlhf_checkpoint.yaml), so the execution path stays config-driven

What to say if asked “How do you know the RM is wired correctly?”

- We use pairwise chosen/rejected batches, not scalar labels.
- The final RM eval is on held-out preference pairs.
- The archived checkpoint is materialized to disk and reused later through `pa2_rm_hh_rlhf_checkpoint`.

## 4.2 SFT

The SFT entry point is [`src/alignlab/cli/train_sft.py`](../../src/alignlab/cli/train_sft.py).

Implementation notes:

- `PreferenceExample` rows are converted into `SFTExample` rows by taking the chosen continuation.
- The response-only cross-entropy path is implemented in [`src/alignlab/objectives/sft.py`](../../src/alignlab/objectives/sft.py).
- Prompt tokens are masked out in [`src/alignlab/data/collators.py`](../../src/alignlab/data/collators.py).
- Held-out perplexity is computed by [`evaluate_sft_perplexity`](../../src/alignlab/eval/pipeline.py).
- Greedy sample checkpoints are generated at configurable intervals and written to `artifacts/samples/`.

Current PA2 budget:

- fixed HH subset: `20,000` chosen responses
- `max_steps = 600`
- held-out perplexity every `100` optimizer steps
- greedy sample checkpoints every `200` optimizer steps
- held-out perplexity slice: `500` prompts
- early stopping: `heldout_perplexity`, `mode=min`, `min_delta=0.02` relative, `patience=3`, `min_steps=200`

Why the held-out perplexity change matters:

- PA2 explicitly asks for periodic held-out monitoring rather than only train loss.
- The repo now logs both training metrics and held-out response-token perplexity every `evaluation.sft_eval_every_steps`.
- Each improving held-out evaluation saves a `best` checkpoint variant, and the selected checkpoint is promoted to `final` when training ends.

## 4.3 DPO

The DPO entry point is [`src/alignlab/cli/train_pairwise.py`](../../src/alignlab/cli/train_pairwise.py).

Implementation notes:

- Sequence log-probs are computed only over response tokens in [`src/alignlab/trainers/pairwise_trainer.py`](../../src/alignlab/trainers/pairwise_trainer.py).
- The objective is the logistic DPO loss in [`src/alignlab/objectives/dpo.py`](../../src/alignlab/objectives/dpo.py).
- The reference model stays frozen through [`src/alignlab/models/reference.py`](../../src/alignlab/models/reference.py).
- The CLI now logs an `initialization_sanity` event before training.
- The trainer logs `chosen_response_length`, `rejected_response_length`, and `response_length_gap` so verbosity drift is visible.

Important sanity checks:

- Padding invariance is covered by [`tests/unit/test_objectives_and_rollout.py`](../../tests/unit/test_objectives_and_rollout.py).
- Initialization sanity is logged before the first gradient update.

Current PA2 budget:

- shared HH subset: `20,000` preference pairs
- `train_batch_size = 2`
- `gradient_accumulation_steps = 1`
- `max_sequence_length = 512`
- main DPO run: `max_steps = 300`
- periodic evaluation every `50` optimizer steps
- training-time held-out prompt subset size: `50`
- training-time held-out pair subset size: `200`
- early stopping: `preference_accuracy`, `mode=max`, `min_delta=0.02`, `patience=3`, `min_steps=100`

Current DPO ablation budget:

- shared HH subset: `10,000` preference pairs
- `train_batch_size = 2`
- `gradient_accumulation_steps = 1`
- `max_sequence_length = 512`
- each `pa2_dpo_beta_*` run: `max_steps = 100`
- periodic evaluation disabled during training for the ablation runs
- final evaluation still runs at checkpoint save time

Rationale:

- the PDF baseline uses 1 epoch and evaluation every 25 steps, but also explicitly allows compute-budget adjustment
- on the current GPU, the original DPO microbatch and sequence length also OOMed during full training, so the PA2 config now uses a smaller pairwise microbatch and a shorter max sequence length
- the full DPO schedule plus four ablation runs would otherwise dominate total wall-clock time
- the canonical ablation comparison is still produced later by [`src/alignlab/cli/compare_pa2.py`](../../src/alignlab/cli/compare_pa2.py) on a larger shared evaluation set

What to say if asked “Why can DPO prefer longer answers?”

- DPO compares whole-response log-ratios.
- Longer chosen responses can accumulate more positive mass unless the model or dataset discourages it.
- That is why we now track response lengths in the trainer logs.

## 4.4 PPO

The PPO path is the `ppo` mode of [`src/alignlab/cli/train_online.py`](../../src/alignlab/cli/train_online.py).

Implementation notes:

- Rollout generation, old-policy caching, reference log-probs, rewards, terminals, and values are all collected in [`src/alignlab/trainers/online_rl_trainer.py`](../../src/alignlab/trainers/online_rl_trainer.py).
- GAE is implemented in [`src/alignlab/rollout/gae.py`](../../src/alignlab/rollout/gae.py).
- The PPO loss is implemented in [`src/alignlab/objectives/ppo.py`](../../src/alignlab/objectives/ppo.py).
- The value head is initialized small in [`src/alignlab/models/value.py`](../../src/alignlab/models/value.py).

New PA2-facing reporting:

- `gradient_norm`
- `ratio_start_mean`
- `ratio_start_min`
- `ratio_start_max`
- `mean_response_length`
- `nonzero_advantage_token_fraction`

Why the ratio-start logging matters:

- At the beginning of a PPO update, the policy being updated should still match the cached `pi_old`.
- That means the ratio should start near `1`.
- The code now records that directly on the first update minibatch.

Hardware-fit budget used in the current `pa2_ppo_hh_rlhf` config:

- shared HH subset: `20,000` prompts
- `max_steps = 300`
- `rollout_batch_size = 1`
- `update_minibatch_size = 1`
- `max_prompt_length = 256`
- `max_response_length = 64`
- `max_sequence_length = 384`
- `generation.max_new_tokens = 64`
- `evaluation.eval_every_steps = 50`
- training-time held-out prompt subset size: `50`
- early stopping: `rm_win_rate_vs_sft`, `mode=max`, `min_delta=0.02`, `patience=3`, `min_steps=100`

These values are lower than the original aspirational PA2 online budget because the larger version OOMed on a 10.7 GB GPU during rollout log-prob computation. The reduced values are the documented final budget for this hardware.

## 4.5 GRPO

GRPO uses the same online trainer as PPO but swaps out the objective and advantage construction.

Implementation notes:

- Group-relative rewards are normalized in [`src/alignlab/rollout/advantages.py`](../../src/alignlab/rollout/advantages.py).
- Sequence-level advantages are broadcast back over response tokens.
- The GRPO objective is implemented in [`src/alignlab/objectives/grpo.py`](../../src/alignlab/objectives/grpo.py).
- `degenerate_fraction` is logged when a group has effectively zero relative signal.

Why this matters:

- GRPO can produce batches with no useful learning signal when all group members get the same reward.
- That failure mode is now visible in logs instead of being silent.

Hardware-fit budget used in the current `pa2_grpo_hh_rlhf` config:

- shared HH subset: `20,000` prompts
- `max_steps = 300`
- `rollout_batch_size = 1`
- `group_size = 2`
- `update_minibatch_size = 1`
- `max_prompt_length = 256`
- `max_response_length = 64`
- `max_sequence_length = 384`
- `generation.max_new_tokens = 64`
- `generation.group_size = 2`
- `evaluation.eval_every_steps = 50`
- training-time held-out prompt subset size: `50`
- early stopping: `rm_win_rate_vs_sft`, `mode=max`, `min_delta=0.02`, `patience=3`, `min_steps=100`

## 4.6 RLVR on GSM8K

RLVR is also driven by [`src/alignlab/cli/train_online.py`](../../src/alignlab/cli/train_online.py), with the `rlvr` objective and the GSM8K verifier.

Implementation notes:

- Reward comes from numeric answer verification, not an RM.
- The verifier path is [`src/alignlab/rollout/verifiers.py`](../../src/alignlab/rollout/verifiers.py).
- The adapter handles boxed answers and truncation before training starts.
- `verify_gsm8k_answer_extractor` runs a PA2 precheck harness and writes a JSON artifact before the first rollout.

New RLVR-specific reporting:

- extractor precheck summary
- `pass_at_1`
- `format_compliance_rate`
- `mean_response_length`
- `kl_from_reference`
- `nonzero_advantage_token_fraction`

What to say if asked “Why is RLVR sparse?”

- The reward is concentrated at the terminal answer.
- If the model does not emit a valid answer or emits the wrong one, most tokens receive no direct positive credit.
- The new `nonzero_advantage_token_fraction` metric exposes how much of each batch actually carries signal.

Hardware-fit budget used in the current `pa2_rlvr_gsm8k` config:

- fixed GSM8K subset: `2,000` train problems
- `max_steps = 300`
- `rollout_batch_size = 1`
- `group_size = 2`
- `update_minibatch_size = 1`
- `max_prompt_length = 256`
- `max_response_length = 96`
- `max_sequence_length = 384`
- `generation.max_new_tokens = 96`
- `generation.group_size = 2`
- `evaluation.eval_every_steps = 50`
- training-time held-out prompt subset size: `100`
- early stopping: `pass_at_1`, `mode=max`, `min_delta=0.01`, `patience=2`, `min_steps=150`

There is also a deliberate degenerate-batch guard in [`src/alignlab/trainers/online_rl_trainer.py`](../../src/alignlab/trainers/online_rl_trainer.py): if every sequence advantage in a GRPO/RLVR minibatch is effectively zero, the trainer now skips the policy update and logs zeroed metrics instead of propagating `NaN` values. This was added after tiny-run RLVR sanity checks produced all-zero reward groups.

## 5. Evaluation and artifact story

The shared evaluation file is [`src/alignlab/eval/pipeline.py`](../../src/alignlab/eval/pipeline.py).

There are now two KL modes:

- `sampled`: uses only the sampled token path and is appropriate for rollout-time shaping
- `full_vocab`: computes exact per-token KL from full logits and is appropriate for final PA2 comparison tables

The low-level KL helpers are in:

- [`src/alignlab/rollout/kl.py`](../../src/alignlab/rollout/kl.py)
- [`src/alignlab/eval/kl_eval.py`](../../src/alignlab/eval/kl_eval.py)

This split is intentional:

- Sampled-token KL is cheaper and good enough for online training.
- Full-vocabulary KL is non-negative and better for evaluation tables, where stability matters more than speed.

### 5.1 Cross-method comparison

The new PA2 aggregator is [`src/alignlab/cli/compare_pa2.py`](../../src/alignlab/cli/compare_pa2.py).

It produces:

- HH comparison summary: `artifacts/tables/pa2_method_comparison_hh_summary.json/.csv`
- HH side-by-side samples: `artifacts/samples/pa2_method_comparison_hh_samples.json/.csv`
- RLVR comparison summary: `artifacts/tables/pa2_method_comparison_rlvr_summary.json/.csv`
- RLVR side-by-side samples: `artifacts/samples/pa2_method_comparison_rlvr_samples.json/.csv`
- DPO `beta` ablation table: `artifacts/tables/pa2_dpo_beta_ablation_summary.json/.csv`
- DPO `beta` ablation plot: `artifacts/plots/pa2_dpo_beta_ablation_beta_sweep.png`

The HH comparison rows include:

- method name
- RM score mean
- RM win-rate vs SFT
- KL from reference
- preference accuracy
- mean response length
- resource summary columns

The RLVR comparison rows include:

- method name
- pass@1
- format compliance
- mean response length
- KL from reference
- resource summary columns

## 6. How to answer common viva questions

This section is intentionally short-answer and code-linked.

### “How do you know the HH-RLHF parser is correct?”

Run the setup audit in [`src/alignlab/cli/setup_audit.py`](../../src/alignlab/cli/setup_audit.py). It previews three canonical examples taken after adapter conversion. The adapter itself is [`src/alignlab/data/adapters/hh_rlhf.py`](../../src/alignlab/data/adapters/hh_rlhf.py).

### “Why is the reference model frozen?”

Because DPO and KL-regularized online RL need a stable baseline. The frozen reference bundle is built through [`src/alignlab/models/reference.py`](../../src/alignlab/models/reference.py), and the trainable policy remains separate.

### “How do you compute DPO scores?”

The trainer computes response-only sequence log-probs for chosen and rejected completions in [`src/alignlab/trainers/pairwise_trainer.py`](../../src/alignlab/trainers/pairwise_trainer.py), then the DPO logistic loss uses the difference between policy and reference log-ratios in [`src/alignlab/objectives/dpo.py`](../../src/alignlab/objectives/dpo.py).

### “How do you know padding side does not change DPO sequence scores?”

There is a dedicated padding-invariance test in [`tests/unit/test_objectives_and_rollout.py`](../../tests/unit/test_objectives_and_rollout.py).

### “How do you know PPO starts from `rho = 1`?”

`OnlineRLTrainer.update_from_rollout` logs the first observed ratio statistics as `ratio_start_mean`, `ratio_start_min`, and `ratio_start_max` in [`src/alignlab/trainers/online_rl_trainer.py`](../../src/alignlab/trainers/online_rl_trainer.py). The integration test also checks that the metric exists.

### “How do you know the PPO clip is actually blocking gradients?”

The zero-gradient clipped positive-advantage sanity test is in [`tests/unit/test_objectives_and_rollout.py`](../../tests/unit/test_objectives_and_rollout.py).

### “Why are there two KL implementations?”

Training uses sampled-token KL for efficiency. Final evaluation uses exact full-vocabulary KL for a clean PA2 metric. The switch is controlled by `evaluation.kl_mode` in [`configs/defaults.yaml`](../../configs/defaults.yaml).

### “How do you know the GSM8K answer extractor works?”

Before RLVR training, [`src/alignlab/cli/train_online.py`](../../src/alignlab/cli/train_online.py) runs [`verify_gsm8k_answer_extractor`](../../src/alignlab/eval/pa2_tools.py) and writes `pa2_rlvr_gsm8k_extractor_precheck.json`. The extractor itself is in [`src/alignlab/data/adapters/gsm8k.py`](../../src/alignlab/data/adapters/gsm8k.py).

### “Why is RLVR harder to optimize than RM-based PPO/GRPO?”

Because the reward is sparse, binary, and verifier-defined. Most batches carry less dense token-level signal. The repo now makes that visible with `nonzero_advantage_token_fraction` and `degenerate_fraction`.

### “Where is the ablation?”

The official PA2 ablation is the DPO `beta` sweep. The configs are the `pa2_dpo_beta_*` files in [`configs/experiment/`](../../configs/experiment/), and the aggregate report is produced by [`src/alignlab/cli/compare_pa2.py`](../../src/alignlab/cli/compare_pa2.py).

### “How do you know which checkpoint was finally selected?”

The training CLIs now log `best_checkpoint` and `early_stopping` events to the JSONL logs, and they write a `checkpoint_selection` block into each resource summary. The selected `best` checkpoint is promoted into the experiment’s `final` checkpoint directory before the final evaluation is run.

## 7. Things to know and caveats

### 7.1 Memory and quantization

- The repo is still designed around tight-GPU-memory operation.
- Frozen reference and reward paths can stay quantized.
- Small hardware may still force lower batch sizes before you reach the nominal PA2 budget.
- On the current machine, the online PA2 configs were reduced and documented as the official hardware-fit budgets: PPO uses `1/1/256/64/384`, GRPO uses `1/2/1/256/64/384`, and RLVR uses `1/2/1/256/96/384` for rollout batch, group size, update minibatch, prompt length, response length, and sequence length respectively.

### 7.2 Periodic evaluation was reduced for practicality

- The PDF baseline suggests evaluation every 25 steps.
- The same baseline section also says the configuration may be adjusted according to compute and memory budget.
- In the current PA2 configs, periodic evaluation is reduced to `50` steps for main DPO, PPO, GRPO, and RLVR.
- The DPO ablation runs disable periodic evaluation during training and rely on final evaluation plus the later `compare_pa2` aggregation.
- Final cross-method comparison still uses 200 prompts by default, so the canonical report is not limited to the smaller training-time eval subsets.

### 7.3 Early stopping is metric-driven, not wall-clock-driven

- SFT stops on held-out perplexity.
- DPO stops on held-out preference accuracy.
- PPO and GRPO stop on RM win-rate vs SFT from the shared HH eval slice.
- RLVR stops on pass@1.
- The stop rules are there to cut clearly non-improving runs while keeping the cross-method setup consistent.

### 7.4 Full-vocab KL is for evaluation, not rollouts

- It is more stable and theoretically cleaner.
- It is also more expensive.
- That is why it is opt-in through `evaluation.kl_mode` and used by the `pa2_*` final configs.

### 7.5 Reward models are useful but imperfect

- RM score and RM win-rate are only proxies.
- High RM score does not automatically mean better human preference quality.
- That is why the comparison tables keep KL and qualitative sample rows beside RM-based metrics.

### 7.6 DPO length bias is real

- DPO can drift toward longer outputs depending on dataset structure and model behavior.
- That is why the trainer now logs response lengths directly.

### 7.7 RLVR degeneracy is expected sometimes

- With grouped rewards, some GRPO or RLVR batches will have nearly zero relative signal.
- Those cases are now visible through `degenerate_fraction`.

### 7.8 “PA2 complete in code” is not the same as “final evidence generated”

- The missing implementation gaps are closed.
- You still need the final `pa2_*` runs to produce the canonical tables, plots, and side-by-side samples for submission.

## 8. Reproducibility and exact run order

Assuming the repo is installed in editable mode or otherwise importable:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m alignlab.cli.train_sft --config configs/experiment/pa2_sft_hh_rlhf.yaml
python -m alignlab.cli.setup_audit --config configs/experiment/pa2_ppo_hh_rlhf.yaml
python -m alignlab.cli.train_pairwise --config configs/experiment/pa2_dpo_hh_rlhf.yaml
python -m alignlab.cli.train_online --config configs/experiment/pa2_ppo_hh_rlhf.yaml
python -m alignlab.cli.train_online --config configs/experiment/pa2_grpo_hh_rlhf.yaml
python -m alignlab.cli.train_online --config configs/experiment/pa2_rlvr_gsm8k.yaml
python -m alignlab.cli.train_pairwise --config configs/experiment/pa2_dpo_beta_001.yaml
python -m alignlab.cli.train_pairwise --config configs/experiment/pa2_dpo_beta_010.yaml
python -m alignlab.cli.train_pairwise --config configs/experiment/pa2_dpo_beta_050.yaml
python -m alignlab.cli.train_pairwise --config configs/experiment/pa2_dpo_beta_100.yaml
python -m alignlab.cli.compare_pa2 --num-prompts 200 --sample-limit 5
```

For a fully logged run, use the wrapper script instead:

```bash
bash scripts/run_pa2_pipeline.sh full
```

For a cheap end-to-end sanity pass, use:

```bash
bash scripts/run_pa2_pipeline.sh smoke
```

Recommended order rationale:

- The archived RM must exist before the SFT final RM-eval and the HH comparison stack.
- SFT must exist before DPO, PPO, GRPO, and RLVR.
- The PPO-stack setup audit must run after RM and SFT because that config loads checkpoint-backed policy, reference, and reward sections.
- The DPO `beta` sweep should be run after the main SFT and RM checkpoints exist.
- `compare_pa2` should be last because it expects the final checkpoints and resource summaries.

Expected runtime shape on the current GPU:

- DPO remains the most expensive offline stage in the default PA2 path.
- The archived RM is intentionally reused so that only the policy-side methods remain in the default PA2 execution path.
- PPO, GRPO, and RLVR are much cheaper now because the rollout budgets are smaller and periodic evaluation is less frequent.
- The DPO ablation sweep is still the biggest multi-run cost, which is why those runs use a shorter capped budget and no periodic eval during training.

If hardware forces a fallback, document it in the write-up in this order:

1. Reduce rollout or prompt batch size.
2. Reduce response length.
3. Quantize frozen models.

That ordering matches the repo’s intended tradeoff strategy and keeps the core algorithmic structure unchanged as long as possible. On the current 10.7 GB GPU, the online PA2 configs already reflect that fallback process and should be treated as the submission-facing hardware-fit defaults.

## 9. Which artifact answers which PA2 question

- C0 parser/model-load evidence: `artifacts/tables/<experiment>_setup_audit.json`
- C1 RM quality: `artifacts/tables/rm_hh_rlhf_rm_final_eval.json` and the matching histogram
- C2 SFT training behavior: `artifacts/logs/pa2_sft_hh_rlhf.jsonl` and `artifacts/samples/pa2_sft_hh_rlhf_sft_samples_step_*.json`
- C3 PPO final quality: `artifacts/tables/pa2_ppo_hh_rlhf_ppo_final_eval.json`
- C4 DPO final quality: `artifacts/tables/pa2_dpo_hh_rlhf_dpo_final_eval.json`
- C5 GRPO final quality: `artifacts/tables/pa2_grpo_hh_rlhf_grpo_final_eval.json`
- C6 RLVR quality and extractor sanity: `artifacts/tables/pa2_rlvr_gsm8k_rlvr_final_eval.json` and `artifacts/tables/pa2_rlvr_gsm8k_extractor_precheck.json`
- C7 ablation: `artifacts/tables/pa2_dpo_beta_ablation_summary.json` and `artifacts/plots/pa2_dpo_beta_ablation_beta_sweep.png`
- C8 cross-method comparison: `artifacts/tables/pa2_method_comparison_hh_summary.json`, `artifacts/samples/pa2_method_comparison_hh_samples.json`, `artifacts/tables/pa2_method_comparison_rlvr_summary.json`, and `artifacts/samples/pa2_method_comparison_rlvr_samples.json`
