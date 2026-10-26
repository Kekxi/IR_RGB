#!/usr/bin/env bash
# Submit one fully reviewed B0/B1 fog-vs-clear validation job.
set -euo pipefail
CODE=/work/home/kekexiong/fog_ship_code_20261019
DATA=/work/home/kekexiong/datasets/MTDSP_v2_final_20261019/dataset
CONDITIONS="$CODE/val_conditions_reviewed.csv"
variant="${1:-}"; seed="${2:-}"
[[ "$variant" == b0 || "$variant" == b1 ]] || { echo 'Usage: bash server_submit_visibility_groups.sh b0|b1 SEED' >&2; exit 2; }
[[ "$seed" =~ ^[0-9]+$ && $# -eq 2 ]] || { echo 'Provide numeric seed' >&2; exit 2; }
python_path="$(command -v python)"
weights="$CODE/runs/${variant}_s${seed}/weights/best.pt"
output="$CODE/groups_${variant}_s${seed}_reviewed"
for file in "$CODE/eval_groups.py" "$CODE/audit_visibility_conditions.py" "$CONDITIONS" "$weights"; do
  [[ -f "$file" ]] || { echo "Missing: $file" >&2; exit 1; }
done
[[ ! -e "$output" ]] || { echo "Output exists: $output" >&2; exit 1; }
"$python_path" "$CODE/audit_visibility_conditions.py" --dataset "$DATA" --conditions "$CONDITIONS" --strict
sbatch --partition=xhhgnormal02 --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 \
  --chdir="$CODE" --job-name="fog_groups_${variant}_s${seed}" \
  --output="$CODE/slurm-%x-%j.out" --export="ALL,PYTHON=$python_path" \
  --wrap="'$python_path' '$CODE/eval_groups.py' --dataset '$DATA' --conditions '$CONDITIONS' --weights '$weights' --output '$output' --imgsz 960 --batch 2 --device 0"
