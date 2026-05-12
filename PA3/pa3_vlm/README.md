# AI623 PA3 Vision-Language Models

This repository implements both assignment paths from scratch with PyTorch, HuggingFace Transformers, PEFT/LoRA, torchvision, datasets, numpy, matplotlib, and sklearn.

Part A is a continuous-connector VLM: frozen CLIP ViT patch embeddings are mapped by a learned MLP into SmolLM2-360M-Instruct hidden space and passed through `inputs_embeds`.

Part B is a discrete-token VLM: a VQ-VAE learns 4x4 code maps for synthetic 16x16 images, visual code IDs are added to the LM vocabulary, and an overlay embedding protects the frozen text embedding table.

## Install

```bash
cd pa3_vlm
pip install -r requirements.txt
export PYTHONPATH=src
```

## Run Part A

```bash
bash src/pa3/scripts/run_part_a.sh
```

For a smoke test:

```bash
PYTHONPATH=src python -m pa3.train.train_part_a_phase1 --debug
PYTHONPATH=src python -m pa3.train.train_part_a_phase2 --debug
PYTHONPATH=src python -m pa3.eval.eval_part_a_vqa
```

Part A checkpoints are saved under `weights/`, including `connector_phaseA1.pt`, `partA_phase2_lambda_*.pt`, `partA_phase2_lnorm.pt`, LoRA-rank ablation checkpoints, and `connector_phaseA3.pt`. Captions, PPL/R summaries, VQA metrics, text-only and majority baselines, qualitative examples with top-5 logits, and modality-gap plots are saved under `outputs/`.

## Run Part B

```bash
bash src/pa3/scripts/run_part_b.sh
```

For a smoke test:

```bash
PYTHONPATH=src python -m pa3.train.train_vqvae --debug
PYTHONPATH=src python -m pa3.train.train_part_b_lm --debug
PYTHONPATH=src python -m pa3.eval.eval_part_b_imagegen
```

Part B saves `weights/vqvae_best.pt`, `weights/partB_lm.pt`, `outputs/synthetic_grid.png`, VQ-VAE codebook plots, generated image grids, and metrics CSV files.

## Individual Scripts

Part A:

```bash
PYTHONPATH=src python -m pa3.eval.eval_ppl --config configs/part_a.yaml
PYTHONPATH=src python -m pa3.train.train_part_a_phase1 --config configs/part_a.yaml
PYTHONPATH=src python -m pa3.train.train_part_a_phase2 --config configs/part_a.yaml --sweep
PYTHONPATH=src python -m pa3.train.train_part_a_phase3 --config configs/part_a.yaml
PYTHONPATH=src python -m pa3.eval.eval_part_a_vqa --full
PYTHONPATH=src python -m pa3.eval.eval_part_a_modality_gap
```

Part B:

```bash
PYTHONPATH=src python -m pa3.train.train_vqvae --run-ablations
PYTHONPATH=src python -m pa3.train.train_vqvae --codebook-size 128 --beta 0.25
PYTHONPATH=src python -m pa3.train.train_vqvae --ema
PYTHONPATH=src python -m pa3.train.train_part_b_lm --lambda-replay 0.2 --gamma-img 0.5
PYTHONPATH=src python -m pa3.train.train_part_b_lm --run-ablation-table
PYTHONPATH=src python -m pa3.train.train_part_b_lm --no-projector
PYTHONPATH=src python -m pa3.eval.eval_part_b_vqa --full
PYTHONPATH=src python -m pa3.eval.eval_part_b_imagegen
```

## Important Implementation Notes

The code prints sanity checks requested in the assignment:

- SmolLM2 hidden size: expected `960`.
- SmolLM2 vocabulary before Part B tokens: expected `49152`.
- CLIP output before CLS removal: expected `50` tokens.
- Part A patches: expected `[B, 49, 768]`.
- Connector output: expected `[B, 49, 960]`.
- VQ-VAE latent: expected `[B, 64, 4, 4]`.
- VQ-VAE token map: expected `[B, 4, 4]`.

Loss masking is explicit: image tokens, question tokens, prompt tokens, and other prefixes use `labels=-100`, so loss is only computed on target answers, responses, or visual-token targets.

Part B multimodal samples are pre-encoded before LM training, then the VQ-VAE is moved back to CPU. Token-type traces for three VQA and three image-generation examples are written to `outputs/partB_token_type_debug.txt`.

## Common Debugging Issues

- Loss near zero at step 0 usually means labels were accidentally all `-100` or answer tokens were omitted.
- VQ-VAE codebook collapse shows up as low perplexity and many dead codes; try EMA update, lower beta, or inspect `outputs/part_b_vqvae/usage_hist.png`.
- Visual tokens leaking into Alpaca raises an error in `alpaca_collate` when `forbidden_start_id` is set.
- CLIP receiving gradients is prevented by `load_frozen_clip`; verify no CLIP parameter has `requires_grad=True`.
- Logit masking failure means VQA may emit visual IDs or image generation may emit text IDs; use `apply_logit_mask`.
- Embedding norm divergence is monitored by connector/overlay norm ratios.
- In Part B, resize token embeddings before applying LoRA. The script does this before `apply_lora`.

## Viva Notes

- `inputs_embeds` in Part A lets continuous CLIP patch vectors enter a decoder-only LM without inventing fake token IDs.
- CLIP is frozen to keep visual features stable and reduce GPU memory.
- `labels=-100` for visual/question tokens prevents the LM from being trained to reproduce the prompt.
- Language replay mixes Alpaca text loss with VQA loss to reduce catastrophic forgetting.
- Forgetting ratio `R = PPLfine / PPL0`; values above 1 indicate worse language modeling after fine-tuning.
- Modality gap measures how far normalized visual and text embedding means are from each other.
- VQ-VAE enables discrete visual tokens by replacing image patches with codebook IDs.
- Overlay embeddings protect text knowledge because original text rows remain frozen while new visual rows train separately.
- Logit masking is needed because VQA should generate text only, while image generation should generate visual tokens inside the image region.
- Image generation is harder than VQA because the model must produce a spatially coherent sequence of many dependent visual tokens, not just a short answer.
