#!/usr/bin/env bash
# B3 phase-1 unique-ship evaluation. Existing checkpoints are used; no training.
# bash server_submit_b3_discovery_v4.sh prepare
# bash server_submit_b3_discovery_v4.sh submit
# bash server_submit_b3_discovery_v4.sh status
# bash server_submit_b3_discovery_v4.sh score   # rerun after human correspondence review
set -euo pipefail

CODE="${CODE_ROOT:-/work/home/kekexiong/fog_ship_code_20261019}"
DATA="${DATA_ROOT:-/work/home/kekexiong/datasets/MTDSP_v4_labelreview_20261021/dataset}"
PYTHON="${PYTHON:-python}"
PARTITION="${PARTITION:-xhhgnormal02}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-2}"
REVIEW_IMAGES="${REVIEW_IMAGES:-1}"
ROOT="${DISCOVERY_ROOT:-$CODE/eval_b3_discovery_v4}"
PREP="$ROOT/prepared"
EXPORTS="$ROOT/exports"
SCORES="$ROOT/scores"
CONDITIONS="${CONDITIONS:-$CODE/val_conditions_reviewed.csv}"
RGB_WEIGHTS="${RGB_WEIGHTS:-$CODE/runs_modalities_v4/v4_rgb_only_s42/weights/best.pt}"
IR_WEIGHTS="${IR_WEIGHTS:-$CODE/runs_modalities_v4/v4_ir_only_s42/weights/best.pt}"
B1_WEIGHTS="${B1_WEIGHTS:-$CODE/runs_modalities_v4/v4_dual_b1_s42/weights/best.pt}"
B3_WEIGHTS="${B3_WEIGHTS:-$CODE/runs_b3_v4/b3_dual_native_s42_v4/weights/best.pt}"

need() { [[ -f "$1" ]] || { echo "Missing: $1" >&2; exit 1; }; }
usage() { echo 'Usage: prepare | submit | status | score | --run' >&2; exit 2; }
stage="${1:-}"
[[ $# -eq 1 ]] || usage

if [[ "$stage" == prepare ]]; then
  need "$DATA/pairs.csv"
  need "$CONDITIONS"
  need "$CODE/b3_discovery_export.py"
  [[ ! -e "$PREP" ]] || { echo "Prepared output already exists: $PREP" >&2; exit 1; }
  args=()
  [[ "$REVIEW_IMAGES" == 1 ]] && args+=(--review-images)
  "$PYTHON" "$CODE/b3_discovery_export.py" prepare \
    --dataset "$DATA" --conditions "$CONDITIONS" --output "$PREP" "${args[@]}"
  exit 0
fi

if [[ "$stage" == status ]]; then
  "$PYTHON" - "$ROOT" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = root / 'prepared' / 'prepare_summary.json'
print('prepared:', json.loads(summary.read_text()) if summary.is_file() else 'missing')
for variant in ('rgb_only', 'ir_only', 'b1', 'b3_rgb', 'b3_ir'):
    path = root / 'exports' / variant / 'export_summary.json'
    print(variant + ':', json.loads(path.read_text()) if path.is_file() else 'pending')
audit = root / 'scores' / 'correspondence_audit.json'
print('correspondence:', json.loads(audit.read_text()) if audit.is_file() else 'not audited')
PY
  exit 0
fi

if [[ "$stage" == score ]]; then
  need "$PREP/pairs.jsonl"
  need "$PREP/correspondence_review.csv"
  need "$CODE/b3_discovery_score.py"
  for variant in rgb_only ir_only b1 b3_rgb b3_ir; do
    need "$EXPORTS/$variant/predictions.jsonl"
    need "$EXPORTS/$variant/export_summary.json"
  done
  "$PYTHON" "$CODE/b3_discovery_score.py" --prepared "$PREP" \
    --exports "$EXPORTS" --correspondence "$PREP/correspondence_review.csv" \
    --output "$SCORES" --budgets 0.1 0.25 0.5
  exit 0
fi

[[ "$stage" == submit || "$stage" == --run ]] || usage
need "$PREP/pairs.jsonl"
need "$CODE/b3_discovery_export.py"
need "$CODE/b3_discovery_score.py"
need "$CODE/b3_dual_head.py"
need "$CODE/fog_reliability.py"
need "$CODE/rgb_ir_midfusion.py"
for path in "$RGB_WEIGHTS" "$IR_WEIGHTS" "$B1_WEIGHTS" "$B3_WEIGHTS"; do need "$path"; done
if [[ "$stage" == submit ]]; then
  command -v sbatch >/dev/null || { echo 'sbatch unavailable' >&2; exit 1; }
  python_path="$(command -v "$PYTHON")"
  [[ -x "$python_path" ]] || { echo "Python unavailable: $python_path" >&2; exit 1; }
  sbatch --partition="$PARTITION" --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 \
    --chdir="$CODE" --job-name=b3discover_v4 \
    --output="$CODE/slurm-%x-%j.out" \
    --export="ALL,PYTHON=$python_path,CODE_ROOT=$CODE,DATA_ROOT=$DATA,DISCOVERY_ROOT=$ROOT,CONDITIONS=$CONDITIONS,RGB_WEIGHTS=$RGB_WEIGHTS,IR_WEIGHTS=$IR_WEIGHTS,B1_WEIGHTS=$B1_WEIGHTS,B3_WEIGHTS=$B3_WEIGHTS,IMGSZ=$IMGSZ,BATCH=$BATCH,WORKERS=$WORKERS" \
    --wrap="bash $CODE/server_submit_b3_discovery_v4.sh --run"
  exit 0
fi

"$PYTHON" -c 'import torch,ultralytics; assert torch.cuda.is_available(); assert ultralytics.__version__ == "8.3.163"; print(torch.cuda.get_device_name(0))'
mkdir -p "$EXPORTS"
for variant in rgb_only ir_only b1 b3_rgb b3_ir; do
  case "$variant" in
    rgb_only) weights="$RGB_WEIGHTS" ;;
    ir_only) weights="$IR_WEIGHTS" ;;
    b1) weights="$B1_WEIGHTS" ;;
    b3_rgb|b3_ir) weights="$B3_WEIGHTS" ;;
  esac
  if [[ -f "$EXPORTS/$variant/predictions.jsonl" && -f "$EXPORTS/$variant/export_summary.json" ]]; then
    echo "Already exported: $variant"
    continue
  fi
  "$PYTHON" "$CODE/b3_discovery_export.py" export \
    --prepared "$PREP" --variant "$variant" --weights "$weights" \
    --output "$EXPORTS" --imgsz "$IMGSZ" --batch "$BATCH" \
    --workers "$WORKERS" --device 0
done
bash "$CODE/server_submit_b3_discovery_v4.sh" score
echo "Prediction export finished. Complete $PREP/correspondence_review.csv and rerun score."
