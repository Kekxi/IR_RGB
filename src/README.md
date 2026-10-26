# Source code and server scripts

The top-level files preserve the original flat Python import layout. Key entry points: `run_experiment.py` (B0/B1), `run_b2.py` (B2 control/small/align), `run_v4_modalities.py` (RGB-only/IR-only/B1), `run_b3.py` (B3 first stage), and `run_b3_phase2.py` (control/direct/local). The corresponding `server_submit_*.sh` files encode the HPC runs. `b3_discovery_export.py`, `b3_discovery_score.py`, and `b3_phase2_compare.py` generate native and cross-modal evaluations.

The B3 phase-2 scripts require Ultralytics **8.3.163** and the already-trained B3 first-stage checkpoint. Input paths in server scripts refer to the original cluster; set the documented environment variables for a new cluster. Files under [analysis](analysis/README.md) are additional diagnostics run from the local workspace. Model checkpoints, dataset images, and raw prediction exports are intentionally outside this code directory.
