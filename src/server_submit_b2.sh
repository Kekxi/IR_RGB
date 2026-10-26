#!/usr/bin/env bash
# Submit B2 GPU stages using the same partition and paths as the B0/B1 jobs.
set -euo pipefail
CODE=/work/home/kekexiong/fog_ship_code_20261019
stage="${1:-}"; variant="${2:-}"; seed="${3:-}"
[[ "$stage" == debug || "$stage" == train || "$stage" == val || "$stage" == groups || "$stage" == shift || "$stage" == size ]] || { echo 'Stage: debug|train|val|groups|shift|size' >&2; exit 2; }
[[ "$variant" == control || "$variant" == align || "$variant" == small || ( ( "$stage" == shift || "$stage" == size ) && "$variant" == b1 ) ]] || { echo 'Variant: control|align|small; shift/size also accept b1' >&2; exit 2; }
[[ "$seed" =~ ^[0-9]+$ && $# -eq 3 ]] || { echo 'Provide numeric B1 seed' >&2; exit 2; }
[[ -f "$CODE/server_b2.sh" ]] || { echo 'Upload server_b2.sh first' >&2; exit 1; }
python_path="$(command -v python)"
[[ -x "$python_path" ]] || { echo "Python unavailable: $python_path" >&2; exit 1; }
sbatch --partition=xhhgnormal02 --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 \
  --chdir="$CODE" --job-name="fog_${stage}_b2_${variant}_s${seed}" \
  --output="$CODE/slurm-%x-%j.out" \
  --export="ALL,PYTHON=$python_path,BATCH=${BATCH:-2},IMGSZ=${IMGSZ:-960},WORKERS=${WORKERS:-2}" \
  --wrap="bash $CODE/server_b2.sh $stage $variant $seed"
