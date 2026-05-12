# PA3 Working Document

This file is a lightweight viva prep log. The scripts write the quantitative artifacts under `outputs/`; paste representative plots and notes here after long runs.

## Analytical Notes

- CLIP uses a batch softmax contrastive loss: every other caption/image in the batch is an implicit negative, so larger batches create harder relative-ranking problems and usually stronger semantic structure. False negatives can push paraphrases or co-valid captions apart.
- SigLIP-style sigmoid losses score each image-text pair independently with BCE-style terms. This weakens batch coupling, tolerates multiple plausible positives better, and is often more stable with noisy web data.
- CLIP/SigLIP embeddings live on a semantic unit sphere and deliberately discard pixel-level detail that is irrelevant to retrieval. This helps VQA-style conditioning but is insufficient for autoregressive image-token generation because there is no discrete image vocabulary or decoder.
- VQ-VAE code maps preserve local spatial structure through a 4x4 grid of code IDs. They are less semantically abstract than CLIP features, but they give the LM a unified discrete visual token stream and a decoder for reconstruction.
- Language replay controls catastrophic forgetting. The forgetting ratio used throughout the code is `R = PPLfine / PPL0`; values above 1 mean text-only language modeling degraded after multimodal fine-tuning.
- Modality gap is measured as the distance between normalized visual-token and text-token mean embeddings. The fixed seed-42 subset keeps measurements comparable across checkpoints.

## Run Checklist

- Part A: run `bash src/pa3/scripts/run_part_a.sh` from `pa3_vlm`.
- Part B: run `bash src/pa3/scripts/run_part_b.sh` from `pa3_vlm`.
- Check `outputs/ppl0.txt` before interpreting R.
- Check `outputs/partA_vqa_metrics.csv`, `outputs/partA_modality_gap.csv`, and `outputs/partA_qualitative.txt`.
- Check `outputs/vqvae_summary.csv`, `outputs/partB_vqa_metrics.csv`, `outputs/partB_generated_grid.png`, `outputs/partB_temperature_sweep.png`, and `outputs/partB_imagegen_analysis.txt`.
- For ablations, inspect `outputs/partA_lora_rank_ablation.csv`, `outputs/partB_ablation_table.csv`, and `outputs/vqvae_ablation_results.csv`.

## Implementation Audit - 2026-05-11

- Matched the code paths against the PDF requirements for Part A and Part B: frozen CLIP patches, SmolLM2 fp16, LoRA targets, Alpaca replay guards, VQ-VAE codebook/restart logic, overlay vocabulary expansion, token masks, and mixed-objective loops.
- Fixed Part A VQA answer collation so EOS is appended before padding; labels now cover only answer plus EOS, with no pad token between an answer and EOS.
- Aligned Part A Phase 2 LR with the PDF baseline (`5e-4`) and made Phase 1 use GradScaler plus persistent connector norm rescaling before checkpoint save.
- Fixed dtype boundaries for continuous visual embeddings entering the fp16 LM.
- Fixed Part B overlay training so new visual/special rows stay fp32 for GradScaler while forward passes cast them to the LM dtype.
- Added an overlay output head for the 258 new tokens, keeping original text logits frozen while making visual-token image generation trainable without opening the full 49k text head.
- Smoke checks passed for Part A Phase 2 debug training, Part A VQA eval reload, Part B LM debug training, Part B VQA eval reload, and Part B image generation reload.
