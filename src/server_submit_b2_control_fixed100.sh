#!/usr/bin/env bash
# Paired control for B2 small: restart from B1 and train all 100 epochs.
set -euo pipefail

CODE=/work/home/kekexiong/fog_ship_code_20261019
SEED=42
DATA="$CODE/manifests_v3/fusion_eligible.yaml"
B1="$CODE/runs/b1_s42/weights/best.pt"
RUN="$CODE/runs_v3/b2_control_fixed100_s42"
PYTHON="${PYTHON:-python}"

need_file() { [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }; }
need_file "$CODE/run_b2.py"
need_file "$CODE/server_prepare_mtdsp_v3.sh"
cd "$CODE"

if [[ "${1:-}" == --run ]]; then
  [[ $# -eq 1 ]] || { echo 'No arguments after --run' >&2; exit 2; }
  need_file "$DATA"
  need_file "$B1"
  [[ ! -e "$RUN" ]] || { echo "Run already exists: $RUN" >&2; exit 1; }
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Slurm GPU required"; print(torch.cuda.get_device_name(0))'
  "$PYTHON" run_b2.py train --variant control --weights "$B1" --data "$DATA" \
    --project "$CODE/runs_v3" --name b2_control_fixed100_s42 \
    --epochs 100 --patience 100 --imgsz 960 --batch 2 --workers 2 \
    --device 0 --seed "$SEED"
  "$PYTHON" - "$RUN" <<'PY'
import csv
import sys
from pathlib import Path
run = Path(sys.argv[1])
assert (run / "weights" / "best.pt").is_file(), "Best checkpoint missing"
with (run / "results.csv").open(newline="", encoding="utf-8-sig") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 100 and int(rows[-1]["epoch"]) == 100, f"Expected 100 epochs, got {len(rows)}"
print(f"Fixed-100 control finished: {run}")
PY
  exit 0
fi

[[ $# -eq 0 ]] || { echo 'Usage: bash server_submit_b2_control_fixed100.sh' >&2; exit 2; }
python_path="$(command -v python)"
[[ -x "$python_path" ]] || { echo 'Activate pt206_cu124_py311 first' >&2; exit 1; }
bash "$CODE/server_prepare_mtdsp_v3.sh"
bash "$CODE/server_b2_v3.sh" check
bash "$CODE/server_b2_v3.sh" smoke control "$SEED"
need_file "$DATA"
need_file "$B1"
[[ ! -e "$RUN" ]] || { echo "Run already exists: $RUN" >&2; exit 1; }
sbatch --partition=xhhgnormal02 --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 \
  --chdir="$CODE" --job-name="fog_b2_control_fixed100_s42" \
  --output="$CODE/slurm-%x-%j.out" --export="ALL,PYTHON=$python_path" \
  --wrap="bash $CODE/server_submit_b2_control_fixed100.sh --run"
