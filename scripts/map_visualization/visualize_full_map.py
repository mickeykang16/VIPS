#!/usr/bin/env python3
from __future__ import annotations

import argparse
import lzma
import pickle
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nuplan.common.maps.abstract_map import SemanticMapLayer
from shapely.geometry import LineString, Polygon

from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.visualization.config import BEV_PLOT_CONFIG, MAP_LAYER_CONFIG

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Read (METRIC_CACHE_PATH, V2XREAL_MAP_ROOT) from configs/eval/paths.py if present."""
    paths_file = _REPO_ROOT / "configs" / "eval" / "paths.py"
    if not paths_file.exists():
        return None, None
    import importlib.util

    spec = importlib.util.spec_from_file_location("_eval_paths", paths_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    cache = getattr(mod, "METRIC_CACHE_PATH", None)
    map_root = getattr(mod, "V2XREAL_MAP_ROOT", None)
    return (Path(cache) if cache else None, Path(map_root) if map_root else None)


def parse_args() -> argparse.Namespace:
    default_cache, default_map = _default_paths()
    parser = argparse.ArgumentParser(
        description="Render a full-map, map-only image from V2X metric cache metadata (no ego/agents/trajectories)."
    )
    parser.add_argument(
        "cache_root",
        type=Path,
        nargs="?",
        default=default_cache,
        help="Metric cache root (default: METRIC_CACHE_PATH from configs/eval/paths.py)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("exp/visualization_full_map_only/v2x_real_full_map.png"),
        help="Output PNG path",
    )
    parser.add_argument("--map-root", type=Path, default=default_map, help="Map root (default: V2XREAL_MAP_ROOT from configs/eval/paths.py)")
    parser.add_argument("--padding-m", type=float, default=10.0, help="Padding around global map bounds")
    parser.add_argument("--fig-width", type=float, default=14.0, help="Figure width in inches")
    parser.add_argument("--dpi", type=int, default=220, help="Output image DPI")
    args = parser.parse_args()
    if args.cache_root is None:
        parser.error("cache_root not given and METRIC_CACHE_PATH is missing from configs/eval/paths.py")
    return args


def load_metric_cache(cache_path: Path) -> MetricCache:
    with lzma.open(cache_path, "rb") as f:
        return pickle.load(f)


def find_reference_metric_cache(cache_root: Path) -> Path:
    caches = sorted(cache_root.rglob("metric_cache.pkl"))
    if not caches:
        raise FileNotFoundError(f"No metric_cache.pkl found under: {cache_root}")
    return caches[0]


def get_v2x_map_api(metric_cache: MetricCache, map_root: Optional[Path]):
    map_name = getattr(metric_cache.map_parameters, "map_name", None)
    if map_name != "v2x_real":
        raise RuntimeError(
            f"This full-map exporter currently supports map_name='v2x_real' only (got: {map_name})."
        )

    resolved_map_root = Path(map_root) if map_root else Path(metric_cache.map_parameters.map_root)
    from navsim.common.dataloader_v2xreal import V2XRealMapWrapper

    return V2XRealMapWrapper(resolved_map_root), resolved_map_root


def _iter_polygons(geom: Any) -> Iterable[Polygon]:
    if geom is None:
        return
    if isinstance(geom, Polygon):
        yield geom
        return
    geoms = getattr(geom, "geoms", None)
    if geoms is not None:
        for g in geoms:
            if isinstance(g, Polygon):
                yield g


def _draw_polygon(ax: plt.Axes, polygon_like: Any, config: dict) -> None:
    for poly in _iter_polygons(polygon_like):
        if poly.is_empty:
            continue

        x_ex, y_ex = poly.exterior.xy
        ax.fill(
            x_ex,
            y_ex,
            color=config["fill_color"],
            alpha=config["fill_color_alpha"],
            zorder=config["zorder"],
        )
        ax.plot(
            x_ex,
            y_ex,
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            zorder=config["zorder"],
        )

        for interior in poly.interiors:
            x_in, y_in = interior.xy
            ax.fill(x_in, y_in, color=BEV_PLOT_CONFIG["background_color"], zorder=config["zorder"])


def _draw_linestring(ax: plt.Axes, linestring: LineString, config: dict) -> None:
    if linestring is None or linestring.is_empty:
        return
    x, y = linestring.xy
    ax.plot(
        x,
        y,
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )


def _geometry_bounds(geometries: Sequence[Any]) -> Tuple[float, float, float, float]:
    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    for geom in geometries:
        if geom is None:
            continue

        for poly in _iter_polygons(geom):
            if poly.is_empty:
                continue
            gx0, gy0, gx1, gy1 = poly.bounds
            min_x = min(min_x, gx0)
            min_y = min(min_y, gy0)
            max_x = max(max_x, gx1)
            max_y = max(max_y, gy1)

        if isinstance(geom, LineString) and not geom.is_empty:
            gx0, gy0, gx1, gy1 = geom.bounds
            min_x = min(min_x, gx0)
            min_y = min(min_y, gy0)
            max_x = max(max_x, gx1)
            max_y = max(max_y, gy1)

    if min_x == float("inf"):
        raise RuntimeError("Failed to compute map bounds: no valid geometries found.")
    return min_x, min_y, max_x, max_y


def _extract_lane_centerline(lane_obj: Any) -> Optional[LineString]:
    try:
        path = lane_obj.baseline_path
        ls = path.linestring
        if ls is None or ls.is_empty:
            return None
        return ls
    except Exception:
        return None


def main() -> None:
    args = parse_args()

    cache_root = args.cache_root
    if not cache_root.exists():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")

    ref_cache_path = find_reference_metric_cache(cache_root)
    metric_cache = load_metric_cache(ref_cache_path)
    map_api, resolved_map_root = get_v2x_map_api(metric_cache, args.map_root)

    lanes = [lane for lane in map_api._get_lanes() if getattr(lane, "polygon", None) is not None]
    lane_polygons = [lane.polygon for lane in lanes if lane.polygon is not None and not lane.polygon.is_empty]
    junction_polygons = [poly for poly in map_api._get_junctions() if poly is not None and not poly.is_empty]
    crosswalk_polygons = [poly for poly in map_api._get_crosswalks() if poly is not None and not poly.is_empty]

    centerlines: List[LineString] = []
    for lane in lanes:
        centerline = _extract_lane_centerline(lane)
        if centerline is not None:
            centerlines.append(centerline)

    all_geometries: List[Any] = []
    all_geometries.extend(lane_polygons)
    all_geometries.extend(junction_polygons)
    all_geometries.extend(crosswalk_polygons)
    all_geometries.extend(centerlines)

    min_x, min_y, max_x, max_y = _geometry_bounds(all_geometries)

    span_x = max(1e-3, max_x - min_x)
    span_y = max(1e-3, max_y - min_y)
    fig_w = max(6.0, float(args.fig_width))
    fig_h = max(6.0, min(22.0, fig_w * (span_y / span_x)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor(BEV_PLOT_CONFIG["background_color"])

    for poly in lane_polygons:
        _draw_polygon(ax, poly, MAP_LAYER_CONFIG[SemanticMapLayer.LANE])
    for poly in junction_polygons:
        _draw_polygon(ax, poly, MAP_LAYER_CONFIG[SemanticMapLayer.INTERSECTION])
    for poly in crosswalk_polygons:
        _draw_polygon(ax, poly, MAP_LAYER_CONFIG[SemanticMapLayer.CROSSWALK])
    for line in centerlines:
        _draw_linestring(ax, line, MAP_LAYER_CONFIG[SemanticMapLayer.BASELINE_PATHS])

    pad = max(0.0, float(args.padding_m))
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xlabel("Global X [m]")
    ax.set_ylabel("Global Y [m]")
    ax.set_title(
        "V2X-Real Full Map (Map Only)\n"
        f"lanes={len(lane_polygons)}, junctions={len(junction_polygons)}, crosswalks={len(crosswalk_polygons)}, "
        f"centerlines={len(centerlines)}"
    )

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=max(50, int(args.dpi)), bbox_inches="tight")
    plt.close(fig)

    print(f"Reference cache: {ref_cache_path}")
    print(f"Map root: {resolved_map_root}")
    print(f"Bounds x=[{min_x:.3f}, {max_x:.3f}], y=[{min_y:.3f}, {max_y:.3f}]")
    print(
        "Map objects: "
        f"lanes={len(lane_polygons)}, junctions={len(junction_polygons)}, "
        f"crosswalks={len(crosswalk_polygons)}, centerlines={len(centerlines)}"
    )
    print(f"Saved full-map PNG: {output_path}")


if __name__ == "__main__":
    main()
