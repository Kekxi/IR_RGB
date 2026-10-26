#!/usr/bin/env bash
# Matched V4 YOLO11m baselines: RGB-only, IR-only, B1 dual-stream.
# Login node: prepare | submit train|groups rgb|ir|dual SEED | status [SEED]
# Slurm job: --run train|groups rgb|ir|dual SEED
set -euo pipefail

CODE="${CODE_ROOT:-/work/home/kekexiong/fog_ship_code_20261019}"
DATA="${DATA_ROOT:-/work/home/kekexiong/datasets/MTDSP_v4_labelreview_20261021/dataset}"
PRETRAINED="${PRETRAINED:-$CODE/yolo11m.pt}"
MANIFESTS="$CODE/manifests_modalities_v4"
RUNS="$CODE/runs_modalities_v4"
EVAL="$CODE/eval_modalities_v4"
CONDITIONS="$CODE/val_conditions_reviewed.csv"
PYTHON="${PYTHON:-python}"
PARTITION="${PARTITION:-xhhgnormal02}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-2}"
EPOCHS="${EPOCHS:-100}"

die() { echo "$*" >&2; exit 1; }
need() { [[ -f "$1" ]] || die "Missing: $1"; }
usage() { die 'Usage: prepare | status [SEED] | submit train|groups rgb|ir|dual SEED | --run train|groups rgb|ir|dual SEED'; }
run_name() { if [[ "$1" == dual ]]; then echo "v4_dual_b1_s${2}"; else echo "v4_${1}_only_s${2}"; fi; }

stage="${1:-}"
if [[ "$stage" == prepare ]]; then
  [[ $# -eq 1 ]] || usage
  need "$DATA/pairs.csv"
  need "$PRETRAINED"
  need "$CODE/prepare_data.py"
  need "$CODE/prepare_v4_modalities.py"
  need "$CONDITIONS"
  "$PYTHON" -c 'import torch, ultralytics; assert ultralytics.__version__ == "8.3.163"; print(torch.__version__, ultralytics.__version__)'
  "$PYTHON" "$CODE/prepare_data.py" --dataset "$DATA" --output "$MANIFESTS"
  "$PYTHON" "$CODE/prepare_v4_modalities.py" --dataset "$DATA" --output "$MANIFESTS"
  "$PYTHON" - "$MANIFESTS" "$CONDITIONS" <<'PY'
import csv, json, sys
from pathlib import Path
p = Path(sys.argv[1])
a = json.loads((p/'manifest_summary.json').read_text(encoding='utf-8'))
b = json.loads((p/'modality_summary.json').read_text(encoding='utf-8'))
assert (a['train'], a['val'], b['train'], b['val']) == (1266, 877, 1266, 877)
rows = list(csv.DictReader(Path(sys.argv[2]).open(encoding='utf-8-sig', newline='')))
assert len(rows) == 877 and len({r['id'] for r in rows}) == 877
assert {r['visibility'] for r in rows} == {'haze_fog', 'clear'}
print('V4 manifests OK: RGB, IR and dual use 1266 train / 877 val pairs')
PY
  exit 0
fi
if [[ "$stage" == status ]]; then
  [[ $# -le 2 ]] || usage
  seed="${2:-42}"
  [[ "$seed" =~ ^[0-9]+$ ]] || usage
  for variant in rgb ir dual; do
    run="$RUNS/$(run_name "$variant" "$seed")"
    if [[ -f "$run/results.csv" ]]; then
      "$PYTHON" - "$run" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = list(csv.DictReader((p/'results.csv').open(encoding='utf-8-sig', newline='')))
best = max(rows, key=lambda r: float(r['metrics/mAP50-95(B)']))
print(p.name, 'epochs=', len(rows), 'best_epoch=', best['epoch'],
      'best_AP50-95=', best['metrics/mAP50-95(B)'])
PY
    else echo "$run: no results yet"; fi
  done
  exit 0
fi

if [[ "$stage" == submit || "$stage" == --run ]]; then
  [[ $# -eq 4 ]] || usage
  action="$2"; variant="$3"; seed="$4"
else usage; fi
[[ "$action" == train || "$action" == groups ]] || usage
[[ "$variant" == rgb || "$variant" == ir || "$variant" == dual ]] || usage
[[ "$seed" =~ ^[0-9]+$ ]] || usage
[[ "$EPOCHS" =~ ^[0-9]+$ ]] && (( EPOCHS > 0 )) || die 'EPOCHS must be positive'
need "$PRETRAINED"
need "$CODE/run_v4_modalities.py"
need "$CODE/rgb_ir_midfusion.py"
need "$CODE/fog_reliability.py"
need "$CODE/eval_v4_modality_groups.py"
need "$CONDITIONS"
if [[ "$variant" == dual ]]; then
  manifest="$MANIFESTS/fusion_eligible.yaml"
else
  manifest="$MANIFESTS/${variant}_eligible.yaml"
fi
need "$manifest"
name="$(run_name "$variant" "$seed")"
run="$RUNS/$name"
if [[ "$action" == train ]]; then
  [[ ! -e "$run" ]] || die "Training output already exists: $run"
else
  need "$run/weights/best.pt"
  [[ ! -e "$EVAL/$name" ]] || die "Evaluation output already exists: $EVAL/$name"
fi

if [[ "$stage" == submit ]]; then
  command -v sbatch >/dev/null || die 'sbatch unavailable on this server'
  python_path="$(command -v "$PYTHON")"
  [[ -x "$python_path" ]] || die "Python unavailable: $python_path"
  sbatch --partition="$PARTITION" --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 \
    --chdir="$CODE" --job-name="v4_${action}_${variant}_s${seed}" \
    --output="$CODE/slurm-%x-%j.out" \
    --export="ALL,PYTHON=$python_path,CODE_ROOT=$CODE,DATA_ROOT=$DATA,PRETRAINED=$PRETRAINED,IMGSZ=$IMGSZ,BATCH=$BATCH,WORKERS=$WORKERS,EPOCHS=$EPOCHS" \
    --wrap="bash $CODE/server_submit_v4_modalities.sh --run $action $variant $seed"
  exit 0
fi

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Slurm GPU required"; print(torch.cuda.get_device_name(0))'
cd "$CODE"
case "$action" in
  train)
    "$PYTHON" run_v4_modalities.py train --variant "$variant" --weights "$PRETRAINED" \
      --data "$manifest" --project "$RUNS" --name "$name" --epochs "$EPOCHS" \
      --imgsz "$IMGSZ" --batch "$BATCH" --workers "$WORKERS" --device 0 --seed "$seed"
    "$PYTHON" - "$run" "$EPOCHS" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1]); expected = int(sys.argv[2])
rows = list(csv.DictReader((p/'results.csv').open(encoding='utf-8-sig', newline='')))
assert len(rows) == expected and int(rows[-1]['epoch']) == expected, (len(rows), expected)
assert (p/'weights/best.pt').is_file()
print('Training verified:', p)
PY
    ;;
  groups)
    "$PYTHON" eval_v4_modality_groups.py --variant "$variant" --dataset "$DATA" \
      --conditions "$CONDITIONS" --weights "$run/weights/best.pt" \
      --output "$EVAL/$name" --imgsz "$IMGSZ" --batch "$BATCH" --device 0
    ;;
esac
