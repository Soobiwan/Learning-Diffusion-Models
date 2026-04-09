#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-full}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="$(date +"%Y%m%d_%H%M%S")"
RUN_LOG_DIR="artifacts/run_logs/$RUN_ID"
mkdir -p "$RUN_LOG_DIR"

MANIFEST_PATH="$RUN_LOG_DIR/manifest.txt"
{
  echo "run_id=$RUN_ID"
  echo "mode=$MODE"
  echo "root_dir=$ROOT_DIR"
  echo "python_bin=$PYTHON_BIN"
  echo "pytorch_cuda_alloc_conf=$PYTORCH_CUDA_ALLOC_CONF"
} > "$MANIFEST_PATH"

run_step() {
  local step_name="$1"
  shift
  local log_path="$RUN_LOG_DIR/${step_name}.log"
  echo "[$(date +"%F %T")] START $step_name" | tee -a "$MANIFEST_PATH"
  echo "command: $*" | tee -a "$MANIFEST_PATH"
  "$@" 2>&1 | tee "$log_path"
  local status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "[$(date +"%F %T")] FAIL $step_name exit_code=$status log=$log_path" | tee -a "$MANIFEST_PATH"
    exit "$status"
  fi
  echo "[$(date +"%F %T")] DONE $step_name log=$log_path" | tee -a "$MANIFEST_PATH"
}

run_smoke() {
  run_step "00_train_rm_smoke" \
    "$PYTHON_BIN" -m alignlab.cli.train_rm --config configs/experiment/pa2_rm_hh_rlhf.yaml --sample-limit 32 --max-steps 2
  run_step "01_train_sft_smoke" \
    "$PYTHON_BIN" -m alignlab.cli.train_sft --config configs/experiment/pa2_sft_hh_rlhf.yaml --sample-limit 32 --max-steps 2
  run_step "02_setup_audit" \
    "$PYTHON_BIN" -m alignlab.cli.setup_audit --config configs/experiment/pa2_ppo_hh_rlhf.yaml
  run_step "03_train_dpo_smoke" \
    "$PYTHON_BIN" -m alignlab.cli.train_pairwise --config configs/experiment/pa2_dpo_hh_rlhf.yaml --sample-limit 32 --max-steps 2
  run_step "04_train_ppo_smoke" \
    "$PYTHON_BIN" -m alignlab.cli.train_online --config configs/experiment/pa2_ppo_hh_rlhf.yaml --sample-limit 32 --max-steps 2
  run_step "05_train_grpo_smoke" \
    "$PYTHON_BIN" -m alignlab.cli.train_online --config configs/experiment/pa2_grpo_hh_rlhf.yaml --sample-limit 32 --max-steps 2
  run_step "06_train_rlvr_smoke" \
    "$PYTHON_BIN" -m alignlab.cli.train_online --config configs/experiment/pa2_rlvr_gsm8k.yaml --sample-limit 32 --max-steps 2
}

run_full() {
  run_step "00_train_rm" \
    "$PYTHON_BIN" -m alignlab.cli.train_rm --config configs/experiment/pa2_rm_hh_rlhf.yaml
  run_step "01_train_sft" \
    "$PYTHON_BIN" -m alignlab.cli.train_sft --config configs/experiment/pa2_sft_hh_rlhf.yaml
  run_step "02_setup_audit" \
    "$PYTHON_BIN" -m alignlab.cli.setup_audit --config configs/experiment/pa2_ppo_hh_rlhf.yaml
  run_step "03_train_dpo" \
    "$PYTHON_BIN" -m alignlab.cli.train_pairwise --config configs/experiment/pa2_dpo_hh_rlhf.yaml
  run_step "04_train_ppo" \
    "$PYTHON_BIN" -m alignlab.cli.train_online --config configs/experiment/pa2_ppo_hh_rlhf.yaml
  run_step "05_train_grpo" \
    "$PYTHON_BIN" -m alignlab.cli.train_online --config configs/experiment/pa2_grpo_hh_rlhf.yaml
  run_step "06_train_rlvr" \
    "$PYTHON_BIN" -m alignlab.cli.train_online --config configs/experiment/pa2_rlvr_gsm8k.yaml
  run_step "07_train_dpo_beta_001" \
    "$PYTHON_BIN" -m alignlab.cli.train_pairwise --config configs/experiment/pa2_dpo_beta_001.yaml
  run_step "08_train_dpo_beta_010" \
    "$PYTHON_BIN" -m alignlab.cli.train_pairwise --config configs/experiment/pa2_dpo_beta_010.yaml
  run_step "09_train_dpo_beta_050" \
    "$PYTHON_BIN" -m alignlab.cli.train_pairwise --config configs/experiment/pa2_dpo_beta_050.yaml
  run_step "10_train_dpo_beta_100" \
    "$PYTHON_BIN" -m alignlab.cli.train_pairwise --config configs/experiment/pa2_dpo_beta_100.yaml
  run_step "11_compare_pa2" \
    "$PYTHON_BIN" -m alignlab.cli.compare_pa2 --num-prompts 200 --sample-limit 5
}

case "$MODE" in
  smoke)
    run_smoke
    ;;
  full)
    run_full
    ;;
  *)
    echo "Unknown mode '$MODE'. Use 'smoke' or 'full'." >&2
    exit 2
    ;;
esac

echo "Logs written to $RUN_LOG_DIR"
