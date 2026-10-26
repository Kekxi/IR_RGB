#!/usr/bin/env bash
# Execute on login node for check/smoke, or inside a Slurm GPU job for debug/train/val/groups.
set -euo pipefail
CODE=/work/home/kekexiong/fog_ship_code_20261019
DATA="${DATA:-/work/home/kekexiong/datasets/MTDSP_v2_final_20261019/dataset}"
MANIFEST="${MANIFEST:-$CODE/manifests/fusion_eligible.yaml}"
CONDITIONS="${CONDITIONS:-$CODE/val_conditions.csv}"
RUNS="${RUNS:-$CODE/runs}"
B1_RUNS="${B1_RUNS:-$CODE/runs}"
DEBUG_RUNS="${DEBUG_RUNS:-$CODE/runs_debug}"
EVAL_ROOT="${EVAL_ROOT:-$CODE/eval}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
PYTHON="${PYTHON:-python}"
BATCH="${BATCH:-2}"
IMGSZ="${IMGSZ:-960}"
WORKERS="${WORKERS:-2}"

usage() { echo 'Usage: bash server_b2.sh check | smoke|debug|train|val|groups control|align|small SEED | shift|size b1|control|align|small SEED' >&2; exit 2; }
need_file() { [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }; }
need_cuda() { "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "GPU required; submit with server_submit_b2.sh"; print(torch.cuda.get_device_name(0))'; }

stage="${1:-}"; variant="${2:-}"; seed="${3:-}"
need_file "$CODE/b2_variants.py"
need_file "$CODE/run_b2.py"
need_file "$CODE/fog_reliability.py"
need_file "$CODE/rgb_ir_midfusion.py"
need_file "$MANIFEST"
cd "$CODE"
if [[ "$stage" == check ]]; then
  [[ $# -eq 1 ]] || usage
  "$PYTHON" -c 'import ultralytics; assert ultralytics.__version__ == "8.3.163", ultralytics.__version__; import b2_variants; print("B2 imports OK")'
  echo 'Manifest and B2 imports OK. Smoke requires a finished B1 best.pt.'
  exit 0
fi
[[ "$stage" == smoke || "$stage" == debug || "$stage" == train || "$stage" == val || "$stage" == groups || "$stage" == shift || "$stage" == size ]] || usage
[[ "$variant" == control || "$variant" == align || "$variant" == small || ( ( "$stage" == shift || "$stage" == size ) && "$variant" == b1 ) ]] || usage
[[ "$seed" =~ ^[0-9]+$ ]] || usage
[[ $# -eq 3 ]] || usage
b1="$B1_RUNS/b1_s${seed}/weights/best.pt"
name="b2_${variant}_s${seed}"
b2="$RUNS/$name/weights/best.pt"
if [[ "$stage" == smoke || "$stage" == debug || "$stage" == train || ( ( "$stage" == shift || "$stage" == size ) && "$variant" == b1 ) ]]; then need_file "$b1"; else need_file "$b2"; fi
if [[ "$stage" == smoke ]]; then
  "$PYTHON" run_b2.py smoke --variant "$variant" --weights "$b1"
  exit 0
fi
need_cuda
case "$stage" in
  debug)
    [[ ! -e "$DEBUG_RUNS/${name}" ]] || { echo 'Debug run exists' >&2; exit 1; }
    "$PYTHON" run_b2.py debug --variant "$variant" --weights "$b1" --data "$MANIFEST" \
      --project "$DEBUG_RUNS" --name "$name" --imgsz "$IMGSZ" --batch "$BATCH" \
      --device 0 --workers "$WORKERS" --seed "$seed"
    ;;
  train)
    [[ ! -e "$RUNS/$name" ]] || { echo "Run exists: $RUNS/$name" >&2; exit 1; }
    "$PYTHON" run_b2.py train --variant "$variant" --weights "$b1" --data "$MANIFEST" \
      --project "$RUNS" --name "$name" --epochs 100 --patience 25 --imgsz "$IMGSZ" \
      --batch "$BATCH" --workers "$WORKERS" --device 0 --seed "$seed"
    ;;
  val)
    [[ ! -e "$EVAL_ROOT/$name" ]] || { echo "Evaluation exists: $EVAL_ROOT/$name" >&2; exit 1; }
    "$PYTHON" run_b2.py val --variant "$variant" --weights "$b2" --data "$MANIFEST" \
      --project "$EVAL_ROOT" --name "$name" --imgsz "$IMGSZ" --batch "$BATCH" \
      --workers "$WORKERS" --device 0
    ;;
  groups)
    need_file "$CONDITIONS"
    "$PYTHON" eval_groups.py --dataset "$DATA" --conditions "$CONDITIONS" \
      --weights "$b2" --output "$CODE/groups_${name}${OUTPUT_SUFFIX}" --imgsz "$IMGSZ" \
      --batch "$BATCH" --device 0
    ;;
  shift)
    need_file "$CODE/eval_shift.py"
    if [[ "$variant" == b1 ]]; then checkpoint="$b1"; output="$CODE/shift_b1_s${seed}${OUTPUT_SUFFIX}";
    else checkpoint="$b2"; output="$CODE/shift_${name}${OUTPUT_SUFFIX}"; fi
    [[ ! -e "$output" ]] || { echo "Shift output exists: $output" >&2; exit 1; }
    "$PYTHON" eval_shift.py --weights "$checkpoint" --data "$MANIFEST" --output "$output" \
      --imgsz "$IMGSZ" --batch "$BATCH" --device 0
    ;;
  size)
    need_file "$CODE/eval_size_recall.py"
    if [[ "$variant" == b1 ]]; then checkpoint="$b1"; output="$CODE/size_b1_s${seed}${OUTPUT_SUFFIX}";
    else checkpoint="$b2"; output="$CODE/size_${name}${OUTPUT_SUFFIX}"; fi
    [[ ! -e "$output" ]] || { echo "Size output exists: $output" >&2; exit 1; }
    "$PYTHON" eval_size_recall.py --weights "$checkpoint" --data "$MANIFEST" --output "$output" \
      --imgsz "$IMGSZ" --batch "$BATCH" --device 0
    ;;
esac
