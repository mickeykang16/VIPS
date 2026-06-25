#!/usr/bin/env bash
#
# V2X-Real two-stage EPDMS evaluation.
#
# Activate the conda env your config needs first, then run with CONFIG=...:
#
#   conda activate vips                          # baselines (constant_velocity/stop/human)
#   CONFIG=configs/eval/constant_velocity.py \
#     bash scripts/evaluation/run_cv_pdm_score_evaluation_v2xreal_stage2_coop_temporal_parallel.sh
#
#   conda activate <cos-v2x-env>                 # CoS-V2X (its own mmcv/mmdet3d env)
#   CONFIG=configs/eval/cos_v2x.py \
#     bash scripts/evaluation/run_cv_pdm_score_evaluation_v2xreal_stage2_coop_temporal_parallel.sh
#
# Each config names the env it needs (CONDA_ENV). The python entry reads the agent,
# machine paths, traffic policy and GPUs/workers from the config (configs/eval/*.py
# + configs/eval/paths.py). The reported numbers use TRAFFIC_POLICY=log_replay (the
# default); idm and constant_velocity are optional.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export NAVSIM_DEVKIT_ROOT="$ROOT"
cd "$ROOT"

# Use the active conda env's python (CONDA_PREFIX is set by `conda activate`, and
# is reliable even when PATH still points elsewhere). Falls back to `python`.
PY="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
exec "${PY:-python}" navsim/planning/script/run_pdm_score_v2xreal_stage2_temporal_parallel.py \
    ${CONFIG:+--config "$CONFIG"} --yes "$@"
