#!/usr/bin/env bash
# B2 P4 alignment versus continued-training control on the frozen v4 label snapshot.
# Login node: prepare | submit train|val|groups|shift control|align SEED | status [SEED]
# GPU job (called by Slurm): --run train|val|groups|shift control|align SEED
set -euo pipefail

CODE="${CODE_ROOT:-/work/home/kekexiong/fog_ship_code_20261019}"
DATA="${DATA_ROOT:-/work/home/kekexiong/datasets/MTDSP_v4_labelreview_20261021/dataset}"
V3_DATA="${V3_DATA_ROOT:-/work/home/kekexiong/datasets/MTDSP_v3_labelreview_20261020/dataset}"
B1_RUNS="${B1_RUNS:-$CODE/runs}"
MANIFEST="$CODE/manifests_align_v4/fusion_eligible.yaml"
RUNS="$CODE/runs_align_v4"
EVAL="$CODE/eval_align_v4"
CONDITIONS="$CODE/val_conditions_reviewed.csv"
PYTHON="${PYTHON:-python}"
PARTITION="${PARTITION:-xhhgnormal02}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-2}"

die() { echo "$*" >&2; exit 1; }
need() { [[ -f "$1" ]] || die "Missing: $1"; }
usage() { die 'Usage: prepare | status [SEED] | submit train|val|groups|shift control|align SEED | --run train|val|groups|shift control|align SEED'; }
run_name() { echo "b2_${1}_fixed100_s${2}"; }

stage="${1:-}"
if [[ "$stage" == prepare ]]; then
  [[ $# -eq 1 ]] || usage
  need "$V3_DATA/pairs.csv"
  need "$CODE/prepare_data.py"
  need "$CODE/prepare_v4_from_v3.py"
  need "$CODE/label_corrections_v4/manifest.json"
  need "$CONDITIONS"
  "$PYTHON" -c 'import torch, ultralytics; assert ultralytics.__version__ == "8.3.163"; print(torch.__version__, ultralytics.__version__)'
  "$PYTHON" "$CODE/prepare_v4_from_v3.py" --source "$V3_DATA" --destination "$DATA" --patch "$CODE/label_corrections_v4"
  "$PYTHON" "$CODE/prepare_data.py" --dataset "$DATA" --output "$CODE/manifests_align_v4"
  "$PYTHON" - "$CODE/manifests_align_v4/manifest_summary.json" "$CONDITIONS" <<'PY'
import csv, json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
assert (s['pairs'], s['train'], s['val']) == (2145, 1266, 877), s
rows = list(csv.DictReader(Path(sys.argv[2]).open(encoding='utf-8-sig', newline='')))
assert len(rows) == 877 and len({r['id'] for r in rows}) == 877
assert {r['visibility'] for r in rows} == {'haze_fog', 'clear'}
print('Data and conditions OK: 1266 train, 877 val')
PY
  exit 0
fi
if [[ "$stage" == status ]]; then
  [[ $# -le 2 ]] || usage
  status_seed="${2:-42}"
  [[ "$status_seed" =~ ^[0-9]+$ ]] || usage
  for variant in control align; do
    run="$RUNS/$(run_name "$variant" "$status_seed")"
    if [[ -f "$run/results.csv" ]]; then
      "$PYTHON" - "$run" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1]); rows = list(csv.DictReader((p/'results.csv').open(newline='', encoding='utf-8-sig')))
best = max(rows, key=lambda x: float(x['metrics/mAP50-95(B)']))
print(p, 'epochs=', len(rows), 'best_epoch=', best['epoch'], 'best_AP50-95=', best['metrics/mAP50-95(B)'])
PY
    else echo "$run: not finished"; fi
  done
  exit 0
fi

if [[ "$stage" == submit ]]; then
  [[ $# -eq 4 ]] || usage
  action="$2"; variant="$3"; seed="$4"
elif [[ "$stage" == --run ]]; then
  [[ $# -eq 4 ]] || usage
  action="$2"; variant="$3"; seed="$4"
else usage; fi
[[ "$variant" == control || "$variant" == align ]] || usage
[[ "$seed" =~ ^[0-9]+$ ]] || usage
[[ "$action" == train || "$action" == val || "$action" == groups || "$action" == shift ]] || usage
need "$MANIFEST"
need "$CODE/run_b2.py"
need "$CODE/b2_variants.py"
need "$CODE/fog_reliability.py"
need "$CODE/rgb_ir_midfusion.py"
b1="$B1_RUNS/b1_s${seed}/weights/best.pt"
need "$b1"
name="$(run_name "$variant" "$seed")"
run="$RUNS/$name"
if [[ "$action" == train ]]; then
  [[ ! -e "$run" ]] || die "Training output already exists: $run"
else need "$run/weights/best.pt"; fi
case "$action" in
  val) [[ ! -e "$EVAL/$name" ]] || die "Validation output already exists: $EVAL/$name" ;;
  groups) [[ ! -e "$EVAL/groups_$name" ]] || die "Group output already exists: $EVAL/groups_$name" ;;
  shift) [[ ! -e "$EVAL/shift_$name" ]] || die "Shift output already exists: $EVAL/shift_$name" ;;
esac

if [[ "$stage" == submit ]]; then
  command -v sbatch >/dev/null || die 'sbatch unavailable on this server'
  python_path="$(command -v "$PYTHON")"
  [[ -x "$python_path" ]] || die "Python unavailable: $python_path"
  sbatch --partition="$PARTITION" --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 \
    --chdir="$CODE" --job-name="alignv4_${action}_${variant}_s${seed}" \
    --output="$CODE/slurm-%x-%j.out" \
    --export="ALL,PYTHON=$python_path,CODE_ROOT=$CODE,DATA_ROOT=$DATA,B1_RUNS=$B1_RUNS,IMGSZ=$IMGSZ,BATCH=$BATCH,WORKERS=$WORKERS" \
    --wrap="bash $CODE/server_submit_b2_align_v4.sh --run $action $variant $seed"
  exit 0
fi

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Slurm GPU required"; print(torch.cuda.get_device_name(0))'
cd "$CODE"
case "$action" in
  train)
    "$PYTHON" run_b2.py train --variant "$variant" --weights "$b1" --data "$MANIFEST" \
      --project "$RUNS" --name "$name" --epochs 100 --patience 100 \
      --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" --device 0 --seed "$seed"
    "$PYTHON" - "$run" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1]); rows = list(csv.DictReader((p/'results.csv').open(newline='', encoding='utf-8-sig')))
assert len(rows) == 100 and int(rows[-1]['epoch']) == 100, f'Expected 100 epochs, got {len(rows)}'
assert (p/'weights/best.pt').is_file(), 'best.pt missing'
print('100 epochs and best.pt verified:', p)
PY
    ;;
  val)
    "$PYTHON" run_b2.py val --variant "$variant" --weights "$run/weights/best.pt" --data "$MANIFEST" \
      --project "$EVAL" --name "$name" --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" --device 0
    ;;
  groups)
    need "$CONDITIONS"
    "$PYTHON" eval_groups.py --dataset "$DATA" --conditions "$CONDITIONS" \
      --weights "$run/weights/best.pt" --output "$EVAL/groups_$name" \
      --imgsz "$IMGSZ" --batch "$BATCH" --device 0
    ;;
  shift)
    "$PYTHON" eval_shift.py --weights "$run/weights/best.pt" --data "$MANIFEST" \
      --output "$EVAL/shift_$name" --imgsz "$IMGSZ" --batch "$BATCH" --device 0
    ;;
esac
