#!/usr/bin/env python3
"""BEV visualization of a V2X-Real metric cache (stage 1 + linked stage 2) in ONE figure.

Renders, in a single ego-centric BEV (centered on the stage-1 base ego):
  - drivable-area / map, GT boxes (current + faint GT future), GT human trajectory
  - the stage-1 base ego (orange) and its route (bold blue)
  - every linked stage-2 novel-view offset cache (under <cache_root>/x+*_y*/) that
    shares the frame token: each offset ego pose (light-blue box) and its route
    (thin blue), so the two-stage structure is visible at once.

Usage:
    python scripts/visualization/viz_metric_cache.py <METRIC_CACHE_PATH> \
        [--token TOKEN] [--offsets x+0_y+0,...] [--no-stage2] \
        [--map-root MAP_ROOT] [--output-path out.png]
"""
from __future__ import annotations

import argparse
import lzma
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.visualization.bev import add_map_to_bev_ax, add_oriented_box_to_bev_ax
from navsim.visualization.config import AGENT_CONFIG, BEV_PLOT_CONFIG
from navsim.visualization.stage2_eval_viz import (
    add_tracks,
    add_trajectory_to_ax,
    get_map_api,
    transform_box_to_ego,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_paths() -> Dict[str, Optional[Path]]:
    """Read METRIC_CACHE_PATH / V2XREAL_MAP_ROOT from configs/eval/paths.py if present."""
    paths_file = _REPO_ROOT / "configs" / "eval" / "paths.py"
    out: Dict[str, Optional[Path]] = {"cache": None, "map_root": None}
    if paths_file.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("_eval_paths", paths_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        cache = getattr(mod, "METRIC_CACHE_PATH", None)
        map_root = getattr(mod, "V2XREAL_MAP_ROOT", None)
        out["cache"] = Path(cache) if cache else None
        out["map_root"] = Path(map_root) if map_root else None
    return out


_S2_COLOR = "#1f77b4"
_EGO_S2 = {**AGENT_CONFIG[TrackedObjectType.EGO],
           "fill_color_alpha": 0.0, "line_color": _S2_COLOR, "line_color_alpha": 0.6, "line_width": 1.0, "zorder": 5}


def load_metric_cache(path: Path) -> MetricCache:
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def _is_offset(name: str) -> bool:
    return name.startswith("x+") or name.startswith("x-")


def _path_has_offset(path: Path, cache_root: Path) -> bool:
    rel = path.relative_to(cache_root)
    return bool(rel.parts) and _is_offset(rel.parts[0])


def find_base_cache(cache_root: Path, token: Optional[str], need_stage2: bool) -> Path:
    caches = [c for c in sorted(cache_root.rglob("metric_cache.pkl")) if not _path_has_offset(c, cache_root)]
    if not caches:
        raise FileNotFoundError(f"No base metric_cache.pkl under {cache_root}")
    if token:
        caches = [c for c in caches if token in str(c)] or caches
    if need_stage2:
        for c in caches:
            if find_offset_caches(cache_root, c):
                return c
    return caches[0]


def find_offset_caches(cache_root: Path, base_path: Path) -> Dict[str, Path]:
    """For base <root>/<log>/unknown/<tok>/..., find <root>/<offset>/<log>/unknown/<tok>/..."""
    log = base_path.parents[2].name
    tok = base_path.parent.name
    out: Dict[str, Path] = {}
    for off_dir in sorted(cache_root.glob("x*")):
        if off_dir.is_dir() and _is_offset(off_dir.name):
            cand = off_dir / log / "unknown" / tok / "metric_cache.pkl"
            if cand.exists():
                out[off_dir.name] = cand
    return out


def render(base_mc: MetricCache, offset_items: List[Tuple[str, MetricCache]],
           output_path: Path, map_root: Optional[Path] = None) -> None:
    pose = base_mc.ego_state.rear_axle

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_facecolor(BEV_PLOT_CONFIG["background_color"])
    ax.set_aspect("equal")

    try:
        add_map_to_bev_ax(ax, get_map_api(base_mc, map_root), StateSE2(pose.x, pose.y, pose.heading))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] map load failed: {exc}")

    # ── Stage 2: every linked offset ego pose (light-blue outline) ──
    for _name, omc in offset_items:
        add_oriented_box_to_bev_ax(
            ax, transform_box_to_ego(omc.ego_state.car_footprint.oriented_box, pose), _EGO_S2, add_heading=False,
        )

    # ── GT objects ──
    if base_mc.current_tracked_objects:
        add_tracks(ax, base_mc.current_tracked_objects[0], pose, alpha=1.0)
    for future in (base_mc.future_tracked_objects or [])[::5]:
        add_tracks(ax, future, pose, alpha=0.12)

    # ── Stage 1: base ego (orange) ──
    add_oriented_box_to_bev_ax(
        ax, transform_box_to_ego(base_mc.ego_state.car_footprint.oriented_box, pose),
        AGENT_CONFIG[TrackedObjectType.EGO],
    )

    if base_mc.human_trajectory is not None and len(base_mc.human_trajectory.poses) > 0:
        add_trajectory_to_ax(ax, base_mc.human_trajectory.poses[:, :2],
                             color="#2ca02c", label="Human (GT)", linewidth=2.5, marker="s")

    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_xlim([-margin_y, margin_y])
    ax.set_ylim([-margin_x, margin_x])
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    handles = [
        Line2D([0], [0], color=AGENT_CONFIG[TrackedObjectType.EGO]["line_color"], lw=3, label="Stage 1 ego (base pose)"),
        Line2D([0], [0], color=_S2_COLOR, lw=1.5, label=f"Stage 2 ego ({len(offset_items)} novel-view offsets)"),
        Line2D([0], [0], color="#2ca02c", lw=2, marker="s", label="Human (GT)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(
        f"Metric cache (BEV): stage 1 (base) + {len(offset_items)} linked stage-2 offset poses",
        fontsize=11,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    defaults = _default_paths()
    ap = argparse.ArgumentParser(description="One-figure BEV of a V2X-Real metric cache (stage 1 + all linked stage 2).")
    ap.add_argument("cache_root", type=Path, nargs="?", default=defaults["cache"],
                    help="Metric cache root (default: METRIC_CACHE_PATH from configs/eval/paths.py)")
    ap.add_argument("--token", type=str, default=None, help="Substring of the frame token (default: first base cache with stage-2 offsets)")
    ap.add_argument("--no-stage2", action="store_true", help="Show the base (stage-1) cache only")
    ap.add_argument("--offsets", type=str, default=None, help="Comma-separated stage-2 offsets to show (default: ALL linked offsets)")
    ap.add_argument("--map-root", type=Path, default=defaults["map_root"], help="Map root (default: V2XREAL_MAP_ROOT from configs/eval/paths.py)")
    ap.add_argument("--output-path", type=Path, default=Path("exp/viz/metric_cache_bev.png"))
    args = ap.parse_args()
    if args.cache_root is None:
        ap.error("cache_root not given and METRIC_CACHE_PATH is missing from configs/eval/paths.py")

    base_path = find_base_cache(args.cache_root, args.token, need_stage2=not args.no_stage2)
    base_mc = load_metric_cache(base_path)

    offset_items: List[Tuple[str, MetricCache]] = []
    if not args.no_stage2:
        offsets = find_offset_caches(args.cache_root, base_path)
        names = sorted(offsets)
        if args.offsets:
            names = [o.strip() for o in args.offsets.split(",") if o.strip() in offsets]
        for name in names:
            try:
                offset_items.append((name, load_metric_cache(offsets[name])))
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] offset {name} load failed: {exc}")

    render(base_mc, offset_items, args.output_path, args.map_root)
    print(f"base cache: {base_path}")
    print(f"stage-2 offsets: {len(offset_items)}")
    print(f"saved: {args.output_path}")


if __name__ == "__main__":
    main()
