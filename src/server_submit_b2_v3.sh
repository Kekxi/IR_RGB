#!/usr/bin/env bash
# Prepare and test v3 on the login node, then submit a GPU debug + training job.
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
stage="${1:-}"; variant="${2:-}"; seed="${3:-}"
if [[ "$stage" != train ]]; then
  exec bash "$CODE/server_submit_b2.sh" "$@"
fi
[[ $# -eq 3 && ( "$variant" == control || "$variant" == small ) && "$seed" =~ ^[0-9]+$ ]] || {
  echo 'Usage: bash server_submit_b2_v3.sh train control|small SEED' >&2; exit 2;
}
[[ -f "$CODE/check_b2_debug.py" && -f "$CODE/server_b2_v3.sh" ]] || {
  echo 'Upload updated B2 v3 scripts first' >&2; exit 1;
}
python_path="$(command -v python)"
[[ -x "$python_path" ]] || { echo 'Activate the B1 Python environment first' >&2; exit 1; }
cd "$CODE"
bash "$CODE/server_prepare_mtdsp_v3.sh"
bash "$CODE/server_b2_v3.sh" check
bash "$CODE/server_b2_v3.sh" smoke "$variant" "$seed"
[[ ! -e "$CODE/runs_v3/b2_${variant}_s${seed}" ]] || {
  echo "Training output already exists: $CODE/runs_v3/b2_${variant}_s${seed}" >&2; exit 1;
}
sbatch --partition=xhhgnormal02 --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 \
  --chdir="$CODE" --job-name="fog_train_b2v3_${variant}_s${seed}" \
  --output="$CODE/slurm-%x-%j.out" \
  --export="ALL,PYTHON=$python_path,BATCH=${BATCH:-2},IMGSZ=${IMGSZ:-960},WORKERS=${WORKERS:-2}" \
  --wrap="bash $CODE/server_b2_v3.sh train $variant $seed"
