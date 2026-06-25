"""Single place that resolves VIPS machine paths / experiment knobs.

Values come from ``configs/eval/paths.py`` (the git-ignored, machine-specific
config created from ``paths.example.py``). The eval entry scripts also export
the same keys into the environment via ``scripts/evaluation/eval_config.py``,
so an explicitly-set environment variable takes precedence; ``paths.py`` is the
fallback.

Resolution order for ``get(name)``:
    1. environment variable ``name`` (set by the eval entry, or by the user)
    2. attribute ``name`` in ``configs/eval/paths.py``
    3. the supplied default

This keeps agents and the rest of the code free of scattered ``os.getenv``
calls — they import :func:`get` and read named keys from here instead.
"""

import importlib.util
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PATHS_PY = _REPO_ROOT / "configs" / "eval" / "paths.py"


def _load_module(path):
    """Import a standalone .py file (e.g. configs/eval/*.py) as a module."""
    spec = importlib.util.spec_from_file_location(f"vips_cfg_{Path(path).stem}", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_paths_module():
    """Load ``configs/eval/paths.py`` if present, else return ``None``."""
    if not _PATHS_PY.exists():
        return None
    return _load_module(_PATHS_PY)


_PATHS = _load_paths_module()


def load_eval_config(config_path=None) -> dict:
    """Merge ``configs/eval/paths.py`` with an optional experiment config.

    Returns a dict of the upper-case keys defined in those files. The machine
    paths come from ``paths.py`` (the sibling of ``config_path`` when given,
    else the repo default); the experiment config overrides on overlap.

    Environment variables are intentionally NOT applied here — callers layer
    those on top with higher precedence (env > experiment config > paths.py).
    """
    merged: dict = {}

    def _absorb(module):
        for key in dir(module):
            if key.isupper() and not key.startswith("_"):
                merged[key] = getattr(module, key)

    paths_py = (
        Path(config_path).resolve().parent / "paths.py" if config_path else _PATHS_PY
    )
    if paths_py.exists():
        _absorb(_load_module(paths_py))
    if config_path:
        _absorb(_load_module(config_path))
    return merged


def get(name: str, default=None):
    """Resolve a config value: environment > configs/eval/paths.py > default."""
    val = os.environ.get(name)
    if val is not None and val != "":
        return val
    if _PATHS is not None and hasattr(_PATHS, name):
        return getattr(_PATHS, name)
    return default


def get_first(*names: str, default=None):
    """Return the first of ``names`` that resolves to a non-empty value.

    Lets callers prefer a new key while still honouring a legacy alias, e.g.
    ``get_first("COS_V2X_FOLDER", "SPARSEDRIVE_FOLDER")``.
    """
    for name in names:
        val = get(name)
        if val is not None and val != "":
            return val
    return default
