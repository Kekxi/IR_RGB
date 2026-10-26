#!/usr/bin/env bash
# B3 phase 1: train B1 RGB fusion head plus a native IR head on V4.
# Login node: bash server_submit_b3_v4.sh prepare
#             bash server_submit_b3_v4.sh submit 42
#             bash server_submit_b3_v4.sh status 42
set -euo pipefail

CODE="${CODE_ROOT:-/work/home/kekexiong/fog_ship_code_20261019}"
DATA="${DATA_ROOT:-/work/home/kekexiong/datasets/MTDSP_v4_labelreview_20261021/dataset}"
PYTHON="${PYTHON:-python}"
PARTITION="${PARTITION:-xhhgnormal02}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-2}"
EPOCHS="${EPOCHS:-100}"
MANIFESTS="$CODE/manifests_b3_v4"
MANIFEST="$MANIFESTS/fusion_eligible.yaml"
RUNS="$CODE/runs_b3_v4"
EVAL="$CODE/eval_b3_v4"

die() { echo "$*" >&2; exit 1; }
need() { [[ -f "$1" ]] || die "Missing: $1"; }
usage() { die 'Usage: prepare | submit SEED | status SEED | --run SEED'; }

stage="${1:-}"
if [[ "$stage" == prepare ]]; then
  [[ $# -eq 1 ]] || usage
  need "$DATA/pairs.csv"
  need "$CODE/prepare_data.py"
  need "$CODE/rgb_ir_midfusion.py"
  need "$CODE/fog_reliability.py"
  need "$CODE/b3_dual_head.py"
  need "$CODE/run_b3.py"
  "$PYTHON" -c 'import ultralytics; assert ultralytics.__version__ == "8.3.163", ultralytics.__version__'
  "$PYTHON" "$CODE/prepare_data.py" --dataset "$DATA" --output "$MANIFESTS"
  need "$MANIFEST"
  exit 0
fi

[[ "$stage" == submit || "$stage" == status || "$stage" == --run ]] || usage
[[ $# -eq 2 ]] || usage
seed="$2"
[[ "$seed" =~ ^[0-9]+$ ]] || die 'SEED must be an integer'
[[ "$EPOCHS" =~ ^[0-9]+$ ]] && (( EPOCHS > 0 )) || die 'EPOCHS must be positive'
name="b3_dual_native_s${seed}_v4"
run="$RUNS/$name"
b1="${B1_WEIGHTS:-$CODE/runs_modalities_v4/v4_dual_b1_s${seed}/weights/best.pt}"

if [[ "$stage" == status ]]; then
  if [[ -f "$run/results.csv" ]]; then
    "$PYTHON" - "$run" "$EVAL/$name" <<'PY'
import csv, json, sys
from pathlib import Path
run, evaluation = map(Path, sys.argv[1:])
rows = list(csv.DictReader((run / 'results.csv').open(encoding='utf-8-sig', newline='')))
best = max(rows, key=lambda row: float(row['metrics/mAP50-95(B)']))
print('run:', run, 'epochs:', len(rows), 'best epoch:', best['epoch'],
      'best IR AP50-95:', best['metrics/mAP50-95(B)'])
for head in ('rgb', 'ir'):
    path = evaluation / head / 'native_metrics.json'
    print(head, json.loads(path.read_text(encoding='utf-8')) if path.is_file() else 'not evaluated')
PY
  else
    echo "$run: no results.csv yet"
  fi
  exit 0
fi

need "$b1"
need "$MANIFEST"
need "$CODE/run_b3.py"
need "$CODE/b3_dual_head.py"
need "$CODE/rgb_ir_midfusion.py"
need "$CODE/fog_reliability.py"
[[ ! -e "$run" ]] || die "Training output already exists: $run"
[[ ! -e "$EVAL/$name" ]] || die "Evaluation output already exists: $EVAL/$name"

if [[ "$stage" == submit ]]; then
  command -v sbatch >/dev/null || die 'sbatch unavailable on this server'
  python_path="$(command -v "$PYTHON")"
  [[ -x "$python_path" ]] || die "Python unavailable: $python_path"
  sbatch --partition="$PARTITION" --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 \
    --chdir="$CODE" --job-name="b3v4_s${seed}" \
    --output="$CODE/slurm-%x-%j.out" \
    --export="ALL,PYTHON=$python_path,CODE_ROOT=$CODE,DATA_ROOT=$DATA,B1_WEIGHTS=$b1,IMGSZ=$IMGSZ,BATCH=$BATCH,WORKERS=$WORKERS,EPOCHS=$EPOCHS" \
    --wrap="bash $CODE/server_submit_b3_v4.sh --run $seed"
  exit 0
fi

"$PYTHON" -c 'import torch,ultralytics; assert torch.cuda.is_available(), "Slurm GPU required"; assert ultralytics.__version__ == "8.3.163"; print(torch.cuda.get_device_name(0))'
cd "$CODE"
"$PYTHON" run_b3.py train --weights "$b1" --data "$MANIFEST" \
  --project "$RUNS" --name "$name" --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" --device 0 --seed "$seed"

"$PYTHON" - "$run" "$EPOCHS" <<'PY'
import csv, sys
from pathlib import Path
run, expected = Path(sys.argv[1]), int(sys.argv[2])
rows = list(csv.DictReader((run / 'results.csv').open(encoding='utf-8-sig', newline='')))
assert len(rows) == expected and int(rows[-1]['epoch']) == expected, (len(rows), expected)
assert (run / 'weights' / 'best.pt').is_file()
for key in ('train/rgb_box_loss', 'train/ir_box_loss'):
    assert key in rows[-1], key
print('B3 training verified:', run)
PY

for head in rgb ir; do
  "$PYTHON" run_b3.py val --head "$head" --weights "$run/weights/best.pt" \
    --data "$MANIFEST" --project "$EVAL/$name" --name "$head" \
    --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" --device 0
done
echo "B3 training and both native-head validations finished: $run"
