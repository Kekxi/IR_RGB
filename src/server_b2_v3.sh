#!/usr/bin/env bash
# Use the LabelImg-corrected v3 dataset and fully reviewed weather conditions.
set -euo pipefail
CODE=/work/home/kekexiong/fog_ship_code_20261019
export DATA=/work/home/kekexiong/datasets/MTDSP_v3_labelreview_20261020/dataset
export MANIFEST="$CODE/manifests_v3/fusion_eligible.yaml"
export CONDITIONS="$CODE/val_conditions_reviewed.csv"
export RUNS="$CODE/runs_v3"
export B1_RUNS="$CODE/runs"
export DEBUG_RUNS="$CODE/runs_debug_v3"
export EVAL_ROOT="$CODE/eval_v3"
export OUTPUT_SUFFIX="_v3"
if [[ "${1:-}" == train ]]; then
  [[ $# -eq 3 ]] || { echo 'Usage: train control|small SEED' >&2; exit 2; }
  variant="$2"; seed="$3"
  [[ "$variant" == control || "$variant" == small ]] || { echo 'Variant must be control or small' >&2; exit 2; }
  [[ "$seed" =~ ^[0-9]+$ ]] || exit 2
  run="$DEBUG_RUNS/b2_${variant}_s${seed}"
  b1="$B1_RUNS/b1_s${seed}/weights/best.pt"
  debug_check=("${PYTHON:-python}" "$CODE/check_b2_debug.py" --variant "$variant" --run "$run" --b1 "$b1" --imgsz "${IMGSZ:-960}" --batch "${BATCH:-2}")
  if [[ -e "$run" ]]; then
    "${debug_check[@]}"
  else
    bash "$CODE/server_b2.sh" debug "$variant" "$seed"
    "${debug_check[@]}"
  fi
fi
exec bash "$CODE/server_b2.sh" "$@"
