#!/usr/bin/env bash
# Submit one fog-ship stage to the Slurm GPU partition used by existing MTDSP jobs.
set -euo pipefail

CODE=/work/home/kekexiong/fog_ship_code_20261019
STAGE="${1:-}"
VARIANT="${2:-}"
SEED="${3:-}"

case "$STAGE" in
  debug)
    [[ $# -eq 1 ]] || { echo 'Usage: bash server_submit_actual_paths.sh debug' >&2; exit 2; }
    JOB_NAME=fog_b1_debug
    ;;
  train|val|groups)
    [[ $# -eq 3 && ( "$VARIANT" == b0 || "$VARIANT" == b1 ) && "$SEED" =~ ^[0-9]+$ ]] || {
      echo 'Usage: bash server_submit_actual_paths.sh train|val|groups b0|b1 SEED' >&2
      exit 2
    }
    JOB_NAME="fog_${STAGE}_${VARIANT}_s${SEED}"
    ;;
  *)
    echo 'Usage: bash server_submit_actual_paths.sh debug OR train|val|groups b0|b1 SEED' >&2
    exit 2
    ;;
esac

[[ -f "$CODE/server_run_actual_paths.sh" ]] || { echo 'Upload server_run_actual_paths.sh first' >&2; exit 1; }
[[ -f "$CODE/manifests/fusion_eligible.yaml" ]] || { echo 'Run check on login node first' >&2; exit 1; }
PYTHON_PATH="$(command -v python)"
[[ -x "$PYTHON_PATH" ]] || { echo "Python not executable: $PYTHON_PATH" >&2; exit 1; }
WRAP="bash $CODE/server_run_actual_paths.sh $STAGE"
if [[ "$STAGE" != debug ]]; then WRAP="$WRAP $VARIANT $SEED"; fi

sbatch --partition=xhhgnormal02 --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 \
  --chdir="$CODE" --job-name="$JOB_NAME" --output="$CODE/slurm-%x-%j.out" \
  --export="ALL,PYTHON=$PYTHON_PATH,BATCH=${BATCH:-2},IMGSZ=${IMGSZ:-960},WORKERS=${WORKERS:-2}" \
  --wrap="$WRAP"
