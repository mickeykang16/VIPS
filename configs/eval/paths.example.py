# Machine-specific V2X-Real data paths.
#
#   cp configs/eval/paths.example.py configs/eval/paths.py
#   # then set V2XREAL_ROOT for your machine
#
# paths.py is git-ignored and loaded automatically by the eval + metric-caching
# scripts. It holds ONLY where the V2X-Real data lives on this machine — model
# paths (CoS-V2X, ADMLP) live in their experiment configs (configs/eval/*.py).
#
# Paths may be absolute, or relative to the VIPS repo root (the eval cd's there
# before running). They are derived from a single V2XREAL_ROOT, which is the
# assembled data root from the README "Data preparation" step.
import os

# Root holding the assembled V2X-Real eval data (data/, maps_final/, infos/).
V2XREAL_ROOT = "data/v2xreal"          # e.g. "/abs/path/to/V2X-Real-eval"

V2XREAL_PKL_PATH = os.path.join(V2XREAL_ROOT, "infos/v2xreal/cooperative/spd_infos_temporal_test.pkl")
V2XREAL_MAP_ROOT = os.path.join(V2XREAL_ROOT, "maps_final")
SENSOR_BLOB_PATH = os.path.join(V2XREAL_ROOT, "data")

# Generated outputs + metric cache (repo-relative, git-ignored).
NAVSIM_EXP_ROOT = "exp_v2xreal"
METRIC_CACHE_PATH = os.path.join(NAVSIM_EXP_ROOT, "metric_cache_v2xreal_stage2_coop_2hz_5s")
