#!/usr/bin/env bash
# B3 phase-2 V4: matched B3 continuation, P3 direct transfer, P3 local transfer.
# Login node: bash server_submit_b3_phase2_v4.sh prepare
#             bash server_submit_b3_phase2_v4.sh submit all
#             bash server_submit_b3_phase2_v4.sh status
#             bash server_submit_b3_phase2_v4.sh compare
set -euo pipefail

CODE="${CODE_ROOT:-/work/home/kekexiong/fog_ship_code_20261019}"
DATA="${DATA_ROOT:-/work/home/kekexiong/datasets/MTDSP_v4_labelreview_20261021/dataset}"
PYTHON="${PYTHON:-python}"
PARTITION="${PARTITION:-xhhgnormal02}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-2}"
EPOCHS="${EPOCHS:-40}"
SEED="${SEED:-42}"
MANIFESTS="$CODE/manifests_b3_v4"
MANIFEST="$MANIFESTS/fusion_eligible.yaml"
BASELINE="$CODE/eval_b3_discovery_v4"
PREP="$BASELINE/prepared"
RUNS="$CODE/runs_b3_phase2_v4"
EVAL="$CODE/eval_b3_phase2_v4"
B3_WEIGHTS="${B3_WEIGHTS:-$CODE/runs_b3_v4/b3_dual_native_s${SEED}_v4/weights/best.pt}"

need() { [[ -f "$1" ]] || { echo "Missing: $1" >&2; exit 1; }; }
die() { echo "$*" >&2; exit 1; }
usage() { die 'Usage: prepare | submit all|control|direct|local | status | eval control|direct|local | compare | --run control|direct|local'; }
variant_ok() { [[ "$1" == control || "$1" == direct || "$1" == local ]]; }
run_name() { echo "b3p2_${1}_s${SEED}_v4"; }

stage="${1:-}"
if [[ "$stage" == prepare ]]; then
  [[ $# -eq 1 ]] || usage
  need "$DATA/pairs.csv"
  need "$B3_WEIGHTS"
  need "$PREP/pairs.jsonl"
  need "$PREP/fusion.yaml"
  need "$CODE/prepare_data.py"
  need "$CODE/run_b3_phase2.py"
  need "$CODE/b3_phase2.py"
  need "$CODE/b3_phase2_compare.py"
  need "$CODE/b3_discovery_export.py"
  need "$CODE/b3_discovery_score.py"
  need "$CODE/b3_discovery_prefill.py"
  for variant in rgb_only ir_only b1 b3_rgb b3_ir; do
    need "$BASELINE/exports/$variant/predictions.jsonl"
    need "$BASELINE/exports/$variant/export_summary.json"
  done
  "$PYTHON" -c 'import torch, ultralytics, scipy; assert ultralytics.__version__ == "8.3.163", ultralytics.__version__; print("ultralytics", ultralytics.__version__, "torch", torch.__version__)'
  "$PYTHON" -c 'import inspect; from b3_discovery_score import score; assert "correspondence_provenance" in inspect.signature(score).parameters, "Upload the B3 phase-2 b3_discovery_score.py"'
  "$PYTHON" - "$B3_WEIGHTS" <<'PY'
import sys
from pathlib import Path
from b3_phase2 import B3Phase2DetectionModel  # checkpoint class registration
from b3_dual_head import B3DualHeadDetectionModel
from ultralytics import YOLO
model = YOLO(str(Path(sys.argv[1]).resolve())).model
assert type(model) is B3DualHeadDetectionModel, type(model)
print('B3 phase-1 source checkpoint verified')
PY
  if [[ ! -f "$MANIFEST" ]]; then
    "$PYTHON" "$CODE/prepare_data.py" --dataset "$DATA" --output "$MANIFESTS"
  fi
  need "$MANIFEST"
  echo "Prepared. B3 source: $B3_WEIGHTS"
  exit 0
fi

if [[ "$stage" == status ]]; then
  [[ $# -eq 1 ]] || usage
  "$PYTHON" - "$RUNS" "$EVAL" "$SEED" "$EPOCHS" <<'PY'
import csv, json, sys
from pathlib import Path
runs, evaluation = map(Path, sys.argv[1:3])
seed, expected = sys.argv[3:]
for variant in ('control', 'direct', 'local'):
    name = f'b3p2_{variant}_s{seed}_v4'
    run = runs / name
    path = run / 'results.csv'
    if not path.is_file():
        print(name, 'training: pending')
        continue
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))
    best = max(rows, key=lambda row: float(row['metrics/mAP50-95(B)']))
    print(name, 'epochs:', len(rows), '/', expected, 'best IR mAP50:95:', best['metrics/mAP50-95(B)'])
    for head in ('rgb', 'ir'):
        native = evaluation / 'native_all' / variant / head / 'native_metrics.json'
        fog = evaluation / 'exports' / variant / f'b3_{head}' / 'export_summary.json'
        print(' ', 'native_all', head, json.loads(native.read_text(encoding='utf-8'))['mAP50_95'] if native.is_file() else 'pending')
        print(' ', 'fog', head, json.loads(fog.read_text(encoding='utf-8'))['mAP50_95'] if fog.is_file() else 'pending')
print('comparison:', (evaluation / 'comparison' / 'comparison.md').is_file())
PY
  exit 0
fi

if [[ "$stage" == compare ]]; then
  [[ $# -eq 1 ]] || usage
  need "$PREP/pairs.jsonl"
  need "$CODE/b3_discovery_score.py"
  "$PYTHON" -c 'import inspect; from b3_discovery_score import score; assert "correspondence_provenance" in inspect.signature(score).parameters, "Upload the B3 phase-2 b3_discovery_score.py"'
  for v in rgb_only ir_only b1 b3_rgb b3_ir; do
    need "$BASELINE/exports/$v/predictions.jsonl"
    need "$BASELINE/exports/$v/export_summary.json"
  done
  for mode in control direct local; do
    for head in rgb ir; do
      need "$EVAL/exports/$mode/b3_$head/predictions.jsonl"
      need "$EVAL/exports/$mode/b3_$head/export_summary.json"
    done
  done
  "$PYTHON" "$CODE/b3_phase2_compare.py" --prepared "$PREP" \
    --baseline-exports "$BASELINE/exports" --phase2-exports "$EVAL/exports" \
    --output "$EVAL/comparison"
  echo "Comparison: $EVAL/comparison/comparison.md"
  exit 0
fi

[[ "$stage" == submit || "$stage" == eval || "$stage" == --run ]] || usage
[[ $# -eq 2 ]] || usage
mode="$2"
if [[ "$stage" == submit && "$mode" == all ]]; then
  for one in control direct local; do
    bash "$CODE/server_submit_b3_phase2_v4.sh" submit "$one"
  done
  exit 0
fi
variant_ok "$mode" || usage
[[ "$SEED" =~ ^[0-9]+$ ]] || die 'SEED must be an integer'
[[ "$EPOCHS" =~ ^[0-9]+$ ]] && (( EPOCHS > 0 )) || die 'EPOCHS must be positive'
name="$(run_name "$mode")"
run="$RUNS/$name"
need "$B3_WEIGHTS"
need "$MANIFEST"
need "$PREP/fusion.yaml"
need "$CODE/run_b3_phase2.py"
need "$CODE/b3_phase2.py"
need "$CODE/b3_discovery_export.py"

if [[ "$stage" == submit ]]; then
  [[ ! -e "$run" ]] || die "Training output already exists: $run"
  command -v sbatch >/dev/null || die 'sbatch unavailable'
  python_path="$(command -v "$PYTHON")"
  [[ -x "$python_path" ]] || die "Python unavailable: $python_path"
  sbatch --partition="$PARTITION" --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 \
    --chdir="$CODE" --job-name="b3p2_${mode}_s${SEED}" \
    --output="$CODE/slurm-%x-%j.out" \
    --export="ALL,PYTHON=$python_path,CODE_ROOT=$CODE,DATA_ROOT=$DATA,B3_WEIGHTS=$B3_WEIGHTS,IMGSZ=$IMGSZ,BATCH=$BATCH,WORKERS=$WORKERS,EPOCHS=$EPOCHS,SEED=$SEED" \
    --wrap="bash $CODE/server_submit_b3_phase2_v4.sh --run $mode"
  exit 0
fi

if [[ "$stage" == --run ]]; then
  "$PYTHON" -c 'import torch,ultralytics; assert torch.cuda.is_available(), "GPU required"; assert ultralytics.__version__ == "8.3.163"; print(torch.cuda.get_device_name(0))'
  [[ ! -e "$run" ]] || die "Training output already exists: $run"
  cd "$CODE"
  "$PYTHON" run_b3_phase2.py train --variant "$mode" --weights "$B3_WEIGHTS" \
    --data "$MANIFEST" --project "$RUNS" --name "$name" \
    --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" \
    --workers "$WORKERS" --device 0 --seed "$SEED"
  "$PYTHON" - "$run" "$EPOCHS" <<'PY'
import csv,sys
from pathlib import Path
run,expected=Path(sys.argv[1]),int(sys.argv[2])
rows=list(csv.DictReader((run/'results.csv').open(encoding='utf-8-sig',newline='')))
assert len(rows)==expected and int(rows[-1]['epoch'])==expected,(len(rows),expected)
assert (run/'weights'/'best.pt').is_file()
assert all(k in rows[-1] for k in ('train/rgb_box_loss','train/ir_box_loss'))
print('Training verified:',run)
PY
fi

need "$run/weights/best.pt"
cd "$CODE"
for head in rgb ir; do
  native="$EVAL/native_all/$mode/$head"
  if [[ ! -f "$native/native_metrics.json" ]]; then
    [[ ! -e "$native" ]] || die "Incomplete native validation folder: $native"
    "$PYTHON" run_b3_phase2.py val --head "$head" --weights "$run/weights/best.pt" \
      --data "$MANIFEST" --project "$EVAL/native_all/$mode" --name "$head" \
      --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" --device 0
  fi
  fog="$EVAL/exports/$mode/b3_$head"
  if [[ ! -f "$fog/predictions.jsonl" || ! -f "$fog/export_summary.json" ]]; then
    [[ ! -e "$fog" ]] || die "Incomplete fog export folder: $fog"
    "$PYTHON" b3_discovery_export.py export --prepared "$PREP" \
      --variant "b3_$head" --weights "$run/weights/best.pt" \
      --output "$EVAL/exports/$mode" --imgsz "$IMGSZ" \
      --batch "$BATCH" --workers "$WORKERS" --device 0
  fi
done
echo "Finished $mode: $run and $EVAL/exports/$mode"
