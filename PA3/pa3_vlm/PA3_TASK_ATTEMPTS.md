# PA3 Task Attempts

This is the baseline-first PA3 plan. It keeps completed artifacts and avoids rerunning expensive ablation tables unless explicitly requested.

Fast run:

```bash
cd pa3_vlm
PARTA_GPU=0 PARTB_GPU=1 bash src/pa3/scripts/run_pa_fast.sh
```

Useful switches:

```bash
FORCE=1 bash src/pa3/scripts/run_part_a.sh              # rerun even if checkpoints exist
FORCE=1 bash src/pa3/scripts/run_part_b.sh
RUN_OPTIONAL_ABLATIONS=1 bash src/pa3/scripts/run_part_a.sh
RUN_OPTIONAL_ABLATIONS=1 bash src/pa3/scripts/run_part_b.sh
PARTA_FULL_EVAL=1 bash src/pa3/scripts/run_part_a.sh
PARTB_FULL_EVAL=1 bash src/pa3/scripts/run_part_b.sh
```

## Runtime Policy

- Default scripts now run the required baseline path only.
- Existing completed artifacts are reused:
  - `weights/connector_phaseA1.pt`
  - `weights/partA_phase2_lambda_0.2.pt`
  - `weights/vqvae_best.pt`
  - `outputs/vqvae_ablation_results.csv`
- Optional ablation sweeps are not default because they were what pushed the earlier run far past the PDF timing.
- Part A now caches frozen CLIP patch tokens once per unique CIFAR image, so Phase 2 does not recompute CLIP features for all five VQA templates.
- Part A/Part B training eval PPL uses 100 Alpaca examples, matching the PDF baseline eval setting; initial `PPL0` remains 1,000 Alpaca examples.

## TASK-A-C0

Implemented in `src/pa3/data/cifar_part_a.py`, `src/pa3/models/smollm_lora.py`, and `src/pa3/eval/eval_ppl.py`.

- Loads CIFAR-10 with seed-42 stratified subsets: 1,000/class train and 200/class test.
- Uses `CLIPImageProcessor` for 32 to 224 resize and CLIP normalization.
- Caches frozen CLIP patch tokens per unique CIFAR image.
- Builds 10,000 rotating-template captions.
- Builds 50,000 train and 10,000 val VQA pairs using the five required templates.
- Loads frozen `openai/clip-vit-base-patch32`, confirms 50 tokens before CLS removal, then uses 49 patches.
- Loads `HuggingFaceTB/SmolLM2-360M-Instruct` in fp16 and prints hidden/vocab sanity checks.
- Loads 1,000 Alpaca examples and computes baseline `PPL0`.

## TASK-A-C1

Implemented in `src/pa3/train/train_part_a_phase1.py`.

- Trains only the MLP connector: `Linear(768,960) -> GELU -> Linear(960,960)`.
- Uses Kaiming initialization and reports about 1.66M trainable parameters.
- Freezes CLIP and LM.
- Uses `[BOS emb, V1:49, caption embs]` through `inputs_embeds`.
- Masks BOS and visual positions with `-100`; caption IDs receive loss.
- Uses Adam `3e-4`, batch 32, 1 epoch, GradScaler.
- Measures connector/text norm ratio and persistently rescales connector output if needed.
- Generates 5 held-out captions.
- Saves `weights/connector_phaseA1.pt`.

## TASK-A-C2

Implemented in `src/pa3/train/train_part_a_phase2.py`.

- Loads Phase 1 connector and applies LoRA with `r=16`, `alpha=32`, targets `q_proj,k_proj,v_proj,o_proj`, dropout `0.05`.
- Trains connector + LoRA with mixed VQA and Alpaca replay.
- Uses VQA sequence `[BOS emb, V, question embs, answer embs]`; labels cover answer + EOS only.
- Uses `Lmixed = LVQA + 0.2 * LLM`.
- Uses AdamW `5e-4`, grad accumulation 4, OneCycleLR with 10% warmup, GradScaler.
- Logs quick VQA accuracy and replay PPL summaries.
- Lambda sweep is implemented and optional via `RUN_OPTIONAL_ABLATIONS=1`; already completed partial checkpoints exist for `0.0`, `0.05`, and `0.2`.

## TASK-A-C3

Implemented in `src/pa3/train/train_part_a_phase3.py`.

- Loads `weights/partA_phase2_lambda_0.2.pt`.
- Runs 1 epoch of VQA-only alignment with replay weight `0`.
- Uses learning rate `2e-4`.
- Saves `weights/connector_phaseA3.pt`.

## TASK-A-C4

Implemented in `src/pa3/eval/eval_part_a_vqa.py`.

- Computes exact-match VQA accuracy.
- Reports overall, per-template, and per-class accuracy.
- Computes majority and text-only baselines.
- Writes six qualitative examples with top-5 logits.
- Writes phase/PPL/R plots from logged summaries.
- Default fast eval uses 500 examples; set `PARTA_FULL_EVAL=1` for the full 10,000-pair val set.

## TASK-A-C5

Implemented in `src/pa3/eval/eval_part_a_modality_gap.py`.

- Uses a fixed seed-42 CIFAR subset.
- Computes modality gap, within-visual cosine, within-text cosine, and cross-modal cosine.
- Writes phase plot and PCA fallback visualization.
- Checks A1, A2, and A3 checkpoints when present.
- Norm-loss rerun is implemented as an optional command in `run_part_a.sh`.

## TASK-A-C6

Implemented options:

- LoRA rank sweep in `train_part_a_phase2.py --run-ablation-table`.
- Lambda replay sweep in `train_part_a_phase2.py --sweep`.

Fast default does not run these because they dominate runtime. Use `RUN_OPTIONAL_ABLATIONS=1` when final ablation tables are required.

## TASK-B-C0

Implemented in `src/pa3/data/synthetic_part_b.py` and `src/pa3/train/train_part_b_lm.py`.

- Generates six synthetic classes: spiral, triangle, circle, cross, checkerboard, gradient.
- Uses 1,000/class with 80/20 stratified split.
- Saves a 6 by 5 visual grid at `outputs/synthetic_grid.png`.
- Builds four VQA templates and one image-generation prompt per image.
- Loads SmolLM2 in fp16 with left padding for decoder-only generation.
- Loads 1,000 Alpaca examples and computes replay PPL.

## TASK-B-C1

Implemented in `src/pa3/models/vqvae.py`, `src/pa3/models/vector_quantizer.py`, and `src/pa3/train/train_vqvae.py`.

- Encoder: Conv 3->32, Conv 32->64, Conv 64->64 with GroupNorm + ReLU.
- Latent shape is `[B,64,4,4]`.
- Vector quantizer has K=256, d=64, straight-through estimator, gradient and EMA modes, and dead-code restart threshold 2.
- Decoder uses two transposed convs and final sigmoid conv.
- Trains 80 epochs, batch 64, Adam `3e-4`.
- Logs reconstruction MSE, codebook loss, perplexity, dead codes, and quantization gap.
- Saves usage histogram, codebook cosine heatmap, and token-map visualization.
- Completed artifacts exist:
  - `weights/vqvae_best.pt`
  - `outputs/vqvae_summary.csv`
  - `outputs/vqvae_ablation_results.csv`

## TASK-B-C2

Implemented in `src/pa3/models/overlay_embedding.py` and `src/pa3/train/train_part_b_lm.py`.

- Adds `<image>`, `</image>`, and 256 visual tokens.
- Resizes token embeddings before applying LoRA.
- Uses an overlay embedding with frozen text rows and 258 trainable new rows.
- Initializes special rows from mean text embedding and visual rows from projector transplant.
- Projector warm-up maps VQ codebook vectors from 64 to 960 before transplant.
- Verifies visual/text norm ratio and rescales if outside `[0.2,5]`.
- Adds overlay LM head so new visual tokens can be generated without unfreezing the full text output head.

## TASK-B-C3

Implemented in `src/pa3/data/synthetic_part_b.py`.

- VQA sequence: `[BOS, <image>, 16 visual IDs, </image>, question, answer, EOS]`.
- Image-gen sequence: `[BOS, prompt, <image>, 16 visual IDs, </image>, EOS]`.
- Labels mask prefixes with `-100`.
- Visual IDs use `codebook_index + Vtxt + 2`.
- Pre-encodes all multimodal samples before LM training.
- Moves VQ-VAE back to CPU before LM fine-tuning.
- Writes token-type traces to `outputs/partB_token_type_debug.txt`.

## TASK-B-C4

Implemented in `src/pa3/train/train_part_b_lm.py`.

- Applies LoRA with `r=16`, `alpha=32`, targets `q/k/v/o_proj`.
- Trains LoRA + overlay rows with separate LR groups: LoRA `5e-4`, overlay `5e-5`.
- Uses sequential VQA, image-gen, and text replay forward/backward passes.
- Uses `Lmixed = LVQA + 0.2 * LLM + 0.5 * LIMG`.
- Uses batch 16, grad accumulation 4, OneCycleLR with 10% warmup, GradScaler, and gradient checkpointing.
- Baseline checkpoint is saved/copied as `weights/partB_lm_baseline.pt`.
- Ablation table and break-protection run are implemented and optional via `RUN_OPTIONAL_ABLATIONS=1`.

## TASK-B-C5

Implemented in `src/pa3/eval/eval_part_b_vqa.py` and `src/pa3/eval/eval_part_b_imagegen.py`.

- Evaluates VQA exact-match accuracy, per-template, per-class.
- Computes majority and text-only baselines.
- Writes confusion matrix for the shape template.
- Generates 12 images with image-token logit masking and decodes through frozen VQ-VAE.
- Writes before/after logit-mask histograms.
- Runs temperature sweep for `T = 0.5, 1.0, 1.5`.
- Writes qualitative VQA/image-generation notes and spatial-coherence commentary.
- Default fast eval uses 500 VQA examples; set `PARTB_FULL_EVAL=1` for full 4,800.

## TASK-B-C6

Completed/implemented options:

- Codebook-size and update ablations are completed in `outputs/vqvae_ablation_results.csv`.
- Loss-weight ablation and break-protection are implemented in `train_part_b_lm.py --run-ablation-table`.
- No-projector and frozen-embedding baselines are implemented as `--no-projector` and `--frozen-embedding`.

Fast default reuses completed VQ-VAE ablations and does not run LM ablations unless `RUN_OPTIONAL_ABLATIONS=1`.
