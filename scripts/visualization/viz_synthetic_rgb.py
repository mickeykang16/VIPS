#!/usr/bin/env python3
"""Compare ORIGINAL V2X-Real camera images with SYNTHETIC novel-view images.

For one token, renders, per camera, the original image (from ``test/``) next to the
synthetic novel-view image (from ``test_novel/`` / ``test_novel_infra/``) for a chosen
ego-pose offset. The novel-view directory convention (mirrors the stage2 eval loader):

    original (vehicle): test/{log}/1/{frame}_{cam}.jpeg
    novel    (vehicle): test_novel/{log}/1_{offset}/{frame}_{cam}.jpeg
    original (infra)  : test/{log}/{agent}/{frame}_{cam}.jpeg
    novel    (infra)  : test_novel_infra/{log}/{agent}_{offset}/{frame}_{cam}.jpeg

where ``offset`` looks like ``x+5_y+0``. The script auto-discovers an offset that
exists on disk if the requested one is missing.

Run (vips env):
    source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate vips
    "$CONDA_PREFIX/bin/python" scripts/visualization/viz_synthetic_rgb.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from navsim.common.dataclasses import SensorConfig  # noqa: E402
from navsim.common.dataloader_v2xreal import SceneFilter, SceneLoaderV2XReal  # noqa: E402

# V2XReal PKL camera names -> human-readable titles.
_VEH_CAMS: Dict[str, str] = {"cam1": "Front", "cam2": "Left", "cam3": "Right", "cam4": "Back"}
_INFRA_CAMS: Dict[str, str] = {"cam1": "Infra 0", "cam2": "Infra 1"}
_NUM_HISTORY_FRAMES = 4  # current frame index == 3
_DEFAULT_OFFSET = "x+5_y+0"


def _default_paths() -> Dict[str, str]:
    """Read machine-specific default paths from configs/eval/paths.py if present."""
    paths_file = _REPO_ROOT / "configs" / "eval" / "paths.py"
    defaults = {"pkl": "", "map_root": "", "sensor_blob": ""}
    if paths_file.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("_eval_paths", paths_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        defaults["pkl"] = getattr(mod, "V2XREAL_PKL_PATH", "")
        defaults["map_root"] = getattr(mod, "V2XREAL_MAP_ROOT", "")
        defaults["sensor_blob"] = getattr(mod, "SENSOR_BLOB_PATH", "")
    return defaults


def _veh_novel_path(orig_data_path: str, offset: str) -> Optional[str]:
    """test/{log}/{agent}/{file} -> test_novel/{log}/{agent}_{offset}/{file}.

    The agent index (parts[2]) is NOT always "1": for some logs the ego vehicle is
    agent "2". test_novel holds a matching ``{agent}_{offset}`` dir per agent, so the
    novel path must reuse the original's agent index (mirrors ``_infra_novel_path``).
    """
    parts = orig_data_path.split("/")
    if len(parts) < 4:
        return None
    return f"test_novel/{parts[1]}/{parts[2]}_{offset}/{parts[3]}"


def _infra_novel_path(orig_data_path: str, offset: str) -> Optional[str]:
    """test/{log}/{agent}/{file} -> test_novel_infra/{log}/{agent}_{offset}/{file}."""
    parts = orig_data_path.split("/")
    if len(parts) < 4:
        return None
    return f"test_novel_infra/{parts[1]}/{parts[2]}_{offset}/{parts[3]}"


def _discover_offset(blob: Path, log: str, requested: str) -> str:
    """Return the requested offset if its veh novel dir exists, else any existing one."""
    if (blob / "test_novel" / log / f"1_{requested}").is_dir():
        return requested
    candidates = sorted((blob / "test_novel" / log).glob("1_x*_y*"))
    for cand in candidates:
        # prefer a non-zero offset so the synthetic view differs visibly from the original
        name = cand.name[len("1_"):]
        if name != "x+0_y+0":
            return name
    if candidates:
        return candidates[0].name[len("1_"):]
    return requested


def _load_image(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.array(Image.open(path))


def _gather_rows(info: Dict, blob: Path, offset: str) -> List[Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]]:
    """Build (title, original_image, novel_image) rows for vehicle + infra cameras."""
    rows: List[Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]] = []

    for cam_name, title in _VEH_CAMS.items():
        ci = info.get("cams", {}).get(cam_name)
        if ci is None:
            continue
        orig = _load_image(blob / ci["data_path"])
        novel_rel = _veh_novel_path(ci["data_path"], offset)
        novel = _load_image(blob / novel_rel) if novel_rel else None
        rows.append((f"Vehicle {title}", orig, novel))

    infra = info.get("other_agent_info_dict", {}).get("model_other_agent_inf")
    if infra and infra.get("cams"):
        for cam_name, title in _INFRA_CAMS.items():
            ci = infra["cams"].get(cam_name)
            if ci is None:
                continue
            orig = _load_image(blob / ci["data_path"])
            novel_rel = _infra_novel_path(ci["data_path"], offset)
            novel = _load_image(blob / novel_rel) if novel_rel else None
            rows.append((title, orig, novel))

    return rows


def main() -> None:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", type=str, default=None, help="scene token (default: first)")
    parser.add_argument("--pkl", type=str, default=defaults["pkl"])
    parser.add_argument("--map-root", type=str, default=defaults["map_root"])
    parser.add_argument("--sensor-blob", type=str, default=defaults["sensor_blob"])
    parser.add_argument("--offset", type=str, default=_DEFAULT_OFFSET, help="novel-view offset, e.g. x+5_y+0")
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(_REPO_ROOT / "exp" / "viz" / "viz4_synthetic_rgb.png"),
    )
    args = parser.parse_args()

    pkl_path = Path(args.pkl)
    blob = Path(args.sensor_blob)

    raw_infos = {it["token"]: it for it in pickle.load(open(pkl_path, "rb"))["infos"]}

    scene_filter = SceneFilter(num_history_frames=_NUM_HISTORY_FRAMES, num_future_frames=20, frame_interval=1)
    loader = SceneLoaderV2XReal(
        pkl_path=pkl_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
        sensor_blob_path=blob,
        map_root=Path(args.map_root),
    )
    token = args.token if args.token is not None else loader.tokens[0]
    current_dict = loader.scene_frames_dicts[token][_NUM_HISTORY_FRAMES - 1]
    pkl_token = current_dict["token"]
    info = raw_infos.get(pkl_token)
    if info is None:
        raise KeyError(f"Token {pkl_token} not found in raw PKL infos")

    # The log directory is the first path component of any camera data_path.
    sample_cam = next(iter(info.get("cams", {}).values()))
    log = sample_cam["data_path"].split("/")[1]
    offset = _discover_offset(blob, log, args.offset)
    print(f"[viz_synthetic_rgb] token: {token}")
    print(f"[viz_synthetic_rgb] log: {log}")
    print(f"[viz_synthetic_rgb] offset: {offset}" + (" (auto-discovered)" if offset != args.offset else ""))

    rows = _gather_rows(info, blob, offset)
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.4 * n_rows))
    axes = np.atleast_2d(axes)

    n_orig, n_novel = 0, 0
    for r, (title, orig, novel) in enumerate(rows):
        for c, (img, tag) in enumerate(((orig, "original"), (novel, "novel"))):
            ax = axes[r, c]
            ax.axis("off")
            if img is not None:
                ax.imshow(img)
                if tag == "original":
                    n_orig += 1
                else:
                    n_novel += 1
            else:
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{title} - {tag}", fontsize=10)

    print(f"[viz_synthetic_rgb] loaded {n_orig}/{n_rows} originals, {n_novel}/{n_rows} novel views")
    fig.suptitle(
        f"V2X-Real original vs. synthetic novel-view ({offset})  |  token: {token}", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"[viz_synthetic_rgb] saved {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
