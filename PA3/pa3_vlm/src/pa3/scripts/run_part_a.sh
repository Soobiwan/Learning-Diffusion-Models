#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

force="${FORCE:-0}"

if [[ "$force" == "1" || ! -f outputs/ppl0.txt ]]; then
  python -m pa3.eval.eval_ppl --config configs/part_a.yaml --n 1000
else
  echo "Using existing outputs/ppl0.txt"
fi

if [[ "$force" == "1" || ! -f weights/connector_phaseA1.pt ]]; then
  python -m pa3.train.train_part_a_phase1 --config configs/part_a.yaml
else
  echo "Using existing weights/connector_phaseA1.pt"
fi

if [[ "$force" == "1" || ! -f weights/partA_phase2_lambda_0.2.pt ]]; then
  python -m pa3.train.train_part_a_phase2 --config configs/part_a.yaml --connector-ckpt weights/connector_phaseA1.pt --lambda-replay 0.2
else
  echo "Using existing weights/partA_phase2_lambda_0.2.pt"
fi

if [[ "$force" == "1" || ! -f weights/connector_phaseA3.pt ]]; then
  python -m pa3.train.train_part_a_phase3 --config configs/part_a.yaml --connector-ckpt weights/partA_phase2_lambda_0.2.pt
else
  echo "Using existing weights/connector_phaseA3.pt"
fi

eval_args=()
if [[ "${PARTA_FULL_EVAL:-0}" == "1" ]]; then
  eval_args+=(--full)
fi
python -m pa3.eval.eval_part_a_vqa --config configs/part_a.yaml --checkpoint weights/connector_phaseA3.pt "${eval_args[@]}"
python -m pa3.eval.eval_part_a_modality_gap --config configs/part_a.yaml

if [[ "${RUN_OPTIONAL_ABLATIONS:-0}" == "1" ]]; then
  python -m pa3.train.train_part_a_phase2 --config configs/part_a.yaml --connector-ckpt weights/connector_phaseA1.pt --sweep
  python -m pa3.train.train_part_a_phase2 --config configs/part_a.yaml --connector-ckpt weights/connector_phaseA1.pt --norm-loss-weight 0.01 --output-ckpt weights/partA_phase2_lnorm.pt
  python -m pa3.train.train_part_a_phase2 --config configs/part_a.yaml --connector-ckpt weights/connector_phaseA1.pt --run-ablation-table
fi
