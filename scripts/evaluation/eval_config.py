#!/usr/bin/env python3
"""Emit shell `export` lines from VIPS eval configs (configs/eval/*.py).

Loads the machine paths from the sibling ``configs/eval/paths.py`` (git-ignored)
and then the chosen experiment config, and prints each known key as

    export KEY="${KEY:-<value>}"

so a value set explicitly in the environment (e.g. on the command line) still
overrides the config, and the experiment config overrides paths.py.

Usage (from the eval entry / metric-caching script):
    eval "$(python3 scripts/evaluation/eval_config.py configs/eval/cos_v2x.py)"
"""
import importlib.util
import os
import sys

# Machine paths (live in configs/eval/paths.py, git-ignored).
PATH_KEYS = [
    "V2XREAL_PKL_PATH", "V2XREAL_MAP_ROOT", "SENSOR_BLOB_PATH",
    "NAVSIM_EXP_ROOT", "METRIC_CACHE_PATH", "OUTPUT_BASE_DIR",
    "COS_V2X_FOLDER", "COS_V2X_CONFIG_PATH", "COS_V2X_MODEL_CHECKPOINT_PATH",
    "ADMLP_V2_CKPT_PATH", "ADMLP_V2_STATS_PATH",
]
# Experiment-level knobs (live in the chosen configs/eval/<name>.py).
EXP_KEYS = [
    "AGENT", "AGENT_CLASS_PATH", "CONDA_ENV",
    "COS_V2X_MODE", "SPARSEDRIVE_MODE", "UNIV2X_MODE",
    "TRAFFIC_POLICY", "GPU_IDS", "NUM_WORKERS", "VISUALIZE",
    "EXPERIMENT_NAME", "STAGE1_ONLY", "MAX_VIZ", "MAX_TOKENS", "SCENE_FILTER_FILE",
]
KEYS = PATH_KEYS + EXP_KEYS


def _load(path: str):
    spec = importlib.util.spec_from_file_location("vips_cfg", path)
    if spec is None or spec.loader is None:
        sys.exit(f"eval_config: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: eval_config.py <config.py>")
    cfg_path = os.path.abspath(sys.argv[1])
    cfg_dir = os.path.dirname(cfg_path)

    merged: dict = {}
    # 1) machine paths from the sibling paths.py (if present)
    paths_py = os.path.join(cfg_dir, "paths.py")
    if os.path.exists(paths_py):
        m = _load(paths_py)
        for k in KEYS:
            if hasattr(m, k):
                merged[k] = getattr(m, k)
    # 2) the chosen experiment config (overrides paths.py on overlap)
    m = _load(cfg_path)
    for k in KEYS:
        if hasattr(m, k):
            merged[k] = getattr(m, k)

    for key, val in merged.items():
        value = str(val)
        if any(c in value for c in "\"'`$\n"):
            sys.exit(f"eval_config: unsafe value for {key}: {value!r}")
        print(f'export {key}="${{{key}:-{value}}}"')


if __name__ == "__main__":
    main()
