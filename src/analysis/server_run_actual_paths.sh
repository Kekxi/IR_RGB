#!/usr/bin/env bash
# MTDSP fog-ship B0/B1 experiment on kekexiong's server.
# Run after activating an environment with CUDA PyTorch and ultralytics 8.3.163.
set -euo pipefail

BASE=/work/home/kekexiong
CODE="$BASE/fog_ship_code_20261019"
DATA="$BASE/datasets/MTDSP_v2_final_20261019/dataset"
WEIGHTS="$CODE/yolo11m.pt"
MANIFESTS="$CODE/manifests"
RUNS="$CODE/runs"
EVAL="$CODE/eval"
PYTHON="${PYTHON:-python}"
BATCH="${BATCH:-2}"
IMGSZ="${IMGSZ:-960}"
WORKERS="${WORKERS:-2}"

usage() {
  cat <<'EOF'
Usage:
  bash server_run_actual_paths.sh check
  bash server_run_actual_paths.sh debug
  bash server_run_actual_paths.sh train b0 42
  bash server_run_actual_paths.sh train b1 42
  bash server_run_actual_paths.sh val b0 42
  bash server_run_actual_paths.sh val b1 42
  bash server_run_actual_paths.sh groups b0 42
  bash server_run_actual_paths.sh groups b1 42

Optional environment: PYTHON=/path/to/conda/env/bin/python BATCH=1 IMGSZ=640 WORKERS=2
For a formal B0/B1 comparison, use the same BATCH and IMGSZ for both variants.
EOF
}

need_file() { [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }; }
need_dir() { [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }; }
require_cuda() {
  "$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('No CUDA GPU in this shell. Run this stage inside a GPU compute-node allocation; a login node is insufficient.')
print('CUDA GPU:', torch.cuda.get_device_name(0))
PY
}

command_name="${1:-}"
if [[ -z "$command_name" ]]; then usage; exit 2; fi
need_dir "$CODE"
need_dir "$DATA"
need_file "$CODE/run_experiment.py"
need_file "$CODE/prepare_data.py"
need_file "$DATA/pairs.csv"
need_file "$WEIGHTS"
cd "$CODE"

case "$command_name" in
  check)
    "$PYTHON" - <<'PY'
import torch, ultralytics
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('ultralytics:', ultralytics.__version__)
assert ultralytics.__version__ == '8.3.163', 'Use ultralytics 8.3.163'
if not torch.cuda.is_available():
    print('No GPU here; data preparation and CPU smoke tests may still run on the login node.')
PY
    "$PYTHON" prepare_data.py --dataset "$DATA" --output "$MANIFESTS"
    cat "$MANIFESTS/manifest_summary.json"
    "$PYTHON" run_experiment.py smoke --variant b0 --weights "$WEIGHTS"
    "$PYTHON" run_experiment.py smoke --variant b1 --weights "$WEIGHTS"
    ;;
  debug)
    need_file "$MANIFESTS/fusion_eligible.yaml"
    require_cuda
    [[ ! -e "$CODE/runs_debug/b1_debug" ]] || { echo 'Debug run already exists' >&2; exit 1; }
    "$PYTHON" run_experiment.py train --variant b1 --weights "$WEIGHTS" \
      --data "$MANIFESTS/fusion_eligible.yaml" --project "$CODE/runs_debug" \
      --name b1_debug --epochs 1 --patience 1 --imgsz 640 --batch 1 \
      --workers 2 --fraction 0.02 --device 0
    ;;
  train)
    variant="${2:-}"; seed="${3:-}"
    [[ "$variant" == b0 || "$variant" == b1 ]] || { usage; exit 2; }
    [[ "$seed" =~ ^[0-9]+$ ]] || { usage; exit 2; }
    need_file "$MANIFESTS/fusion_eligible.yaml"
    require_cuda
    name="${variant}_s${seed}"
    [[ ! -e "$RUNS/$name" ]] || { echo "Run already exists: $RUNS/$name" >&2; exit 1; }
    "$PYTHON" run_experiment.py train --variant "$variant" --weights "$WEIGHTS" \
      --data "$MANIFESTS/fusion_eligible.yaml" --project "$RUNS" --name "$name" \
      --epochs 220 --patience 40 --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" \
      --seed "$seed" --device 0
    ;;
  val)
    variant="${2:-}"; seed="${3:-}"
    [[ "$variant" == b0 || "$variant" == b1 ]] || { usage; exit 2; }
    [[ "$seed" =~ ^[0-9]+$ ]] || { usage; exit 2; }
    name="${variant}_s${seed}"
    checkpoint="$RUNS/$name/weights/best.pt"
    need_file "$checkpoint"
    need_file "$MANIFESTS/fusion_eligible.yaml"
    require_cuda
    "$PYTHON" run_experiment.py val --variant "$variant" --weights "$checkpoint" \
      --data "$MANIFESTS/fusion_eligible.yaml" --project "$EVAL" \
      --name "$name" --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" --device 0
    ;;
  groups)
    variant="${2:-}"; seed="${3:-}"
    [[ "$variant" == b0 || "$variant" == b1 ]] || { usage; exit 2; }
    [[ "$seed" =~ ^[0-9]+$ ]] || { usage; exit 2; }
    name="${variant}_s${seed}"
    need_file "$CODE/val_conditions.csv"
    need_file "$RUNS/$name/weights/best.pt"
    require_cuda
    "$PYTHON" eval_groups.py --dataset "$DATA" --conditions "$CODE/val_conditions.csv" \
      --weights "$RUNS/$name/weights/best.pt" --output "$CODE/groups_$name" \
      --imgsz "$IMGSZ" --batch "$BATCH" --device 0
    ;;
  *) usage; exit 2 ;;
esac
