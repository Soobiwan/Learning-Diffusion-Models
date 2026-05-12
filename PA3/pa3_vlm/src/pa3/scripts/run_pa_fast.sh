#!/usr/bin/env bash
set -euo pipefail

parta_gpu="${PARTA_GPU:-0}"
partb_gpu="${PARTB_GPU:-1}"

mkdir -p outputs

echo "Starting Part A on GPU ${parta_gpu}"
(CUDA_VISIBLE_DEVICES="${parta_gpu}" bash src/pa3/scripts/run_part_a.sh > outputs/run_part_a_fast.log 2>&1) &
pid_a=$!

echo "Starting Part B on GPU ${partb_gpu}"
(CUDA_VISIBLE_DEVICES="${partb_gpu}" bash src/pa3/scripts/run_part_b.sh > outputs/run_part_b_fast.log 2>&1) &
pid_b=$!

status=0
wait "${pid_a}" || status=$?
wait "${pid_b}" || status=$?

if [[ "${status}" == "0" ]]; then
  echo "PA3 fast baseline run completed."
else
  echo "PA3 fast baseline run failed; inspect outputs/run_part_a_fast.log and outputs/run_part_b_fast.log" >&2
fi
exit "${status}"
