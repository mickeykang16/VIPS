#!/usr/bin/env bash
# Build the V2X-Real metric cache (one command, both stages):
#   stage 1 = base-pose caches, stage 2 = offset-pose caches.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

# Machine paths from configs/eval/paths.py (git-ignored); explicit env still wins.
[[ -f "$NAVSIM_DEVKIT_ROOT/configs/eval/paths.py" ]] && \
    eval "$(python3 "$SCRIPT_DIR/eval_config.py" "$NAVSIM_DEVKIT_ROOT/configs/eval/paths.py")"

CONDA_ENV="${CONDA_ENV:-vips}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" && -f "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"; conda activate "$CONDA_ENV"
fi

: "${V2XREAL_PKL_PATH:?Set V2XREAL_PKL_PATH in configs/eval/paths.py}"
: "${V2XREAL_MAP_ROOT:?Set V2XREAL_MAP_ROOT in configs/eval/paths.py}"
export NUPLAN_MAPS_ROOT="$V2XREAL_MAP_ROOT"
export V2XREAL_DATA_ROOT="${V2XREAL_DATA_ROOT:-$V2XREAL_MAP_ROOT}"
export PDMS_V2="${PDMS_V2:-true}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_DEVKIT_ROOT/exp_v2xreal}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache_v2xreal_stage2_coop_2hz_5s}"
NUM_CPUS="${NUM_CPUS:-$(nproc)}"
FORCE="$(echo "${FORCE:-true}" | tr '[:upper:]' '[:lower:]')"

cd "$NAVSIM_DEVKIT_ROOT"
# Use the active conda env's python (reliable even if PATH still points elsewhere).
PY="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"; PY="${PY:-python}"
echo "Building metric cache -> $METRIC_CACHE_PATH  (CPUs: $NUM_CPUS, force: $FORCE)"

echo ">>> Stage 1/2: base-pose caches"
"$PY" navsim/planning/script/run_metric_caching_v2xreal.py \
    metric_cache_path="$METRIC_CACHE_PATH" \
    worker=ray_distributed \
    worker.threads_per_node="$NUM_CPUS" \
    force_feature_computation="$FORCE"

echo ">>> Stage 2/2: offset-pose caches"
FORCE_FLAG=(); [[ "$FORCE" == "true" ]] && FORCE_FLAG+=(--force)
"$PY" navsim/planning/script/generate_novel_view_metric_cache.py \
    --v2xreal_pkl_path="$V2XREAL_PKL_PATH" \
    --map_root="$NUPLAN_MAPS_ROOT" \
    --output_dir="$METRIC_CACHE_PATH" \
    --num_workers="$NUM_CPUS" \
    "${FORCE_FLAG[@]}"

echo "Metric cache complete: $METRIC_CACHE_PATH"
