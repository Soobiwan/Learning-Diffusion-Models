#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

force="${FORCE:-0}"

if [[ "${RUN_OPTIONAL_ABLATIONS:-0}" == "1" && ( "$force" == "1" || ! -f outputs/vqvae_ablation_results.csv ) ]]; then
  python -m pa3.train.train_vqvae --config configs/part_b.yaml --run-ablations
elif [[ -f outputs/vqvae_ablation_results.csv ]]; then
  echo "Using existing VQ-VAE ablation table"
fi

if [[ "$force" == "1" || ! -f weights/vqvae_best.pt ]]; then
  python -m pa3.train.train_vqvae --config configs/part_b.yaml
else
  echo "Using existing weights/vqvae_best.pt"
fi

if [[ "$force" == "1" || ! -f weights/partB_lm_baseline.pt ]]; then
  python -m pa3.train.train_part_b_lm --config configs/part_b.yaml --vqvae-ckpt weights/vqvae_best.pt --lambda-replay 0.2 --gamma-img 0.5
  cp weights/partB_lm.pt weights/partB_lm_baseline.pt
else
  echo "Using existing weights/partB_lm_baseline.pt"
  cp weights/partB_lm_baseline.pt weights/partB_lm.pt
fi

eval_args=()
if [[ "${PARTB_FULL_EVAL:-0}" == "1" ]]; then
  eval_args+=(--full)
fi
python -m pa3.eval.eval_part_b_vqa --config configs/part_b.yaml --checkpoint weights/partB_lm_baseline.pt --vqvae-ckpt weights/vqvae_best.pt "${eval_args[@]}"
python -m pa3.eval.eval_part_b_imagegen --config configs/part_b.yaml --checkpoint weights/partB_lm_baseline.pt --vqvae-ckpt weights/vqvae_best.pt

if [[ "${RUN_OPTIONAL_ABLATIONS:-0}" == "1" ]]; then
  python -m pa3.train.train_part_b_lm --config configs/part_b.yaml --vqvae-ckpt weights/vqvae_best.pt --run-ablation-table
fi
