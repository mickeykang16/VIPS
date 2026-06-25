# navsim/visualization/vis_utils.py
from __future__ import annotations

import os
from typing import Optional, Sequence, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union

from nuplan.common.maps.maps_datatypes import SemanticMapLayer


def _plot_shapely_outline(ax: plt.Axes, geom, **kwargs) -> None:
    """Plot shapely Polygon / MultiPolygon outline."""
    if geom is None:
        return
    gt = getattr(geom, "geom_type", "")
    if gt == "Polygon":
        x, y = geom.exterior.xy
        ax.plot(x, y, **kwargs)
    elif gt == "MultiPolygon":
        for g in geom.geoms:
            x, y = g.exterior.xy
            ax.plot(x, y, **kwargs)
    else:
        # fallback: bounding box
        minx, miny, maxx, maxy = geom.bounds
        ax.plot([minx, maxx, maxx, minx, minx],
                [miny, miny, maxy, maxy, miny], **kwargs)


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _get_map_polygons_list(drivable_area_map):
    """
    Fetch polygon list from PDMDrivableMap.
    Your drivable_area_map has `_geometries` based on dir().
    """
    if hasattr(drivable_area_map, "geometries"):
        return drivable_area_map.geometries
    if hasattr(drivable_area_map, "_geometries"):
        return drivable_area_map._geometries
    if hasattr(drivable_area_map, "polygons"):
        return drivable_area_map.polygons
    if hasattr(drivable_area_map, "_polygons"):
        return drivable_area_map._polygons
    return None


def plot_drivable_area_and_ego_corners_global(
    *,
    drivable_area_map,
    ego_polygon_g,
    ego_coords_xy: np.ndarray,
    out_path: str,
    radius_m: float =120.0,
    show_center: bool = False,
    draw_drivable_outline: bool = True,
    drivable_layers: Optional[Sequence[SemanticMapLayer]] = None,
    title: Optional[str] = None,
    # ---- distance/debug options ----
    annotate_corner_distance: bool = True,
    union_subset_radius_m: float = 120.0,
    # ---- DAC mode ----
    dac_use_center: bool = False,  # True: center-based (relaxed), False: 4-corner-based (strict, original)
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """
    GLOBAL XY plot:
      - drivable polygons outlines (subset near ego)
      - ego polygon outline
      - ego corners (off-road corner colored red) [dac_use_center=False]
        OR ego center colored red/green [dac_use_center=True]
      - (debug) off-road distance to drivable union

    Parameters
    ----------
    ego_coords_xy : (5,2) expected (4 corners + center)
                   navsim state_array_to_coords_array output at [proposal,time]
                   ordering should match BBCoordsIndex:
                   [FRONT_LEFT, FRONT_RIGHT, REAR_LEFT, REAR_RIGHT, CENTER]
    dac_use_center : if True, check center point only (relaxed DAC); else check 4 corners (strict DAC)

    Returns
    -------
    point_in_any_drivable : (N,) bool  — (4,) for corners mode, (1,) for center mode
    off_road_indices : list[int]        — corner indices (0-3) or [0] if center is off-road
    distance_to_drivable_m : (N,) float — meters, NaN if unavailable
    """
    _ensure_dir(out_path)

    ego_coords_xy = np.asarray(ego_coords_xy, dtype=float)
    assert ego_coords_xy.ndim == 2 and ego_coords_xy.shape[1] == 2, ego_coords_xy.shape
    if ego_coords_xy.shape[0] < 5:
        raise ValueError(f"ego_coords_xy must have 5 points (4 corners + center). Got {ego_coords_xy.shape}")

    corner_xy = ego_coords_xy[:4]   # (4,2)
    center_xy = ego_coords_xy[4]    # (2,)
    ex, ey = float(center_xy[0]), float(center_xy[1])

    if drivable_layers is None:
        drivable_layers = [
            SemanticMapLayer.ROADBLOCK,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.DRIVABLE_AREA,
            SemanticMapLayer.CARPARK_AREA,
            SemanticMapLayer.LANE,
        ]

    drivable_idcs = drivable_area_map.get_indices_of_map_type(list(drivable_layers))

    # ===== 1) same as scorer: points_in_polygons input is (1,1,5,2) =====
    pts5 = ego_coords_xy[None, None, :, :]  # (1,1,5,2)
    in_polys = drivable_area_map.points_in_polygons(pts5)

    # navsim scorer uses in_polygons.transpose(1,2,0,3)
    # => (n_proposals, n_horizon, n_polygons, n_points)
    in_polys = np.asarray(in_polys).transpose(1, 2, 0, 3)  # (1,1,n_polys,5)

    corners_in_polygon = in_polys[..., :4]  # (1,1,n_polys,4)
    center_in_polygon  = in_polys[..., 4]   # (1,1,n_polys)

    if dac_use_center:
        # relaxed check: evaluate the center point only
        center_in_drivable = bool((center_in_polygon[:, :, drivable_idcs].sum(axis=-1) > 0)[0, 0])
        point_in_any_drivable = np.array([center_in_drivable])  # (1,) bool
        off_road_corner_indices = [] if center_in_drivable else [0]  # 0 = 'center'
        corner_distance_to_drivable_m = np.full((1,), np.nan, dtype=float)
    else:
        # original strict check: 4 corners
        corner_in_any_drivable = (corners_in_polygon[:, :, drivable_idcs].sum(axis=-2) > 0)[0, 0]  # (4,)
        point_in_any_drivable = corner_in_any_drivable
        off_road_corner_indices = np.where(~corner_in_any_drivable)[0].tolist()
        corner_distance_to_drivable_m = np.full((4,), np.nan, dtype=float)

    # ===== 2) distance: point(s) -> (union of nearby drivable polys) =====

    drivable_union = None
    map_polys = _get_map_polygons_list(drivable_area_map)
    if map_polys is not None and len(drivable_idcs) > 0:
        near_polys = []
        r2 = float(union_subset_radius_m) ** 2
        for idx in drivable_idcs:
            poly = map_polys[idx]
            if poly is None:
                continue
            c = poly.centroid
            if (c.x - ex) ** 2 + (c.y - ey) ** 2 <= r2:
                near_polys.append(poly)
        if near_polys:
            try:
                drivable_union = unary_union(near_polys)
            except Exception:
                drivable_union = None

    if drivable_union is not None:
        if dac_use_center:
            p = Point(ex, ey)
            try:
                corner_distance_to_drivable_m[0] = float(p.distance(drivable_union))
            except Exception:
                pass
        else:
            for i in range(4):
                p = Point(float(corner_xy[i, 0]), float(corner_xy[i, 1]))
                try:
                    corner_distance_to_drivable_m[i] = float(p.distance(drivable_union))
                except Exception:
                    pass

    # --- colors
    if dac_use_center:
        center_color_vis = "lime" if point_in_any_drivable[0] else "red"
    else:
        corner_colors = ["lime" if ok else "red" for ok in point_in_any_drivable]

    # --- plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # drivable polygons
    if draw_drivable_outline:
        if map_polys is not None:
            for idx in drivable_idcs:
                poly = map_polys[idx]
                if poly is None:
                    continue
                c = poly.centroid
                # if (c.x - ex) ** 2 + (c.y - ey) ** 2 > radius_m ** 2:
                #     continue
                _plot_shapely_outline(ax, poly, color="0.85", linewidth=1.0, alpha=0.9)
        else:
            ax.text(
                0.02, 0.02,
                "WARN: map polygons not found (polygons/geometries).",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"),
            )

    # ego polygon outline
    _plot_shapely_outline(ax, ego_polygon_g, color="tab:blue", linewidth=2.5, alpha=0.95)

    if dac_use_center:
        # center mode: show only a single large point (always shown)
        ax.scatter([ex], [ey], s=80, c=center_color_vis, zorder=11,
                   edgecolors="black", linewidths=0.8, label="ego center (DAC check)")
        # also show corners in gray for reference
        ax.scatter(corner_xy[:, 0], corner_xy[:, 1], s=12, c="0.6", zorder=9, label="ego corners (ref)")
        # distance annotation for center if off-road
        if annotate_corner_distance and not point_in_any_drivable[0]:
            d = corner_distance_to_drivable_m[0]
            if np.isfinite(d):
                ax.text(
                    ex + 0.3, ey + 0.3,
                    f"center: {d:.2f}m",
                    fontsize=9,
                    color="red",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"),
                    zorder=20,
                )
    else:
        # corners mode: original behavior
        ax.scatter(corner_xy[:, 0], corner_xy[:, 1], s=18, c=corner_colors, zorder=10, label="ego corners")

    # (debug) distance text for off-road corners (corners mode only)
    if not dac_use_center and annotate_corner_distance and np.isfinite(corner_distance_to_drivable_m).any():
        for i in range(4):
            if point_in_any_drivable[i]:
                continue
            d = corner_distance_to_drivable_m[i]
            if not np.isfinite(d):
                continue
            ax.text(
                float(corner_xy[i, 0]) + 0.3,
                float(corner_xy[i, 1]) + 0.3,
                f"{d:.2f}m",
                fontsize=9,
                color="red",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"),
                zorder=20,
            )

    # center (shown separately only when show_center=True and not in center mode)
    if show_center and not dac_use_center:
        ax.scatter([ex], [ey], s=20, c="tab:blue", zorder=11, label="ego center")

    # view
    ax.set_xlim(ex - radius_m, ex + radius_m)
    ax.set_ylim(ey - radius_m, ey + radius_m)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    mode_str = "center(relaxed)" if dac_use_center else "corners(strict)"
    if title is None:
        title = f"Drivable check [{mode_str}] | off-road={off_road_corner_indices} | dist={corner_distance_to_drivable_m}"
    ax.set_title(title)
    ax.legend(loc="upper right")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return point_in_any_drivable, off_road_corner_indices, corner_distance_to_drivable_m


import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely import affinity

from nuplan.common.maps.maps_datatypes import SemanticMapLayer

# -------------------------
# Helpers
# -------------------------

def _plot_polygon(ax: plt.Axes, geom, **kwargs):
    if geom is None:
        return
    gt = getattr(geom, "geom_type", "")
    if gt == "Polygon":
        x, y = geom.exterior.xy
        ax.plot(x, y, **kwargs)
    elif gt == "MultiPolygon":
        for g in geom.geoms:
            x, y = g.exterior.xy
            ax.plot(x, y, **kwargs)

def _to_ego_frame(geom, ego_pose_se2):
    """global shapely -> ego frame shapely (x=fwd,y=left in nuplan)"""
    if geom is None:
        return None
    # translate
    g = affinity.affine_transform(geom, [1, 0, 0, 1, -ego_pose_se2.x, -ego_pose_se2.y])
    # rotate by -heading
    c = float(np.cos(-ego_pose_se2.heading))
    s = float(np.sin(-ego_pose_se2.heading))
    return affinity.affine_transform(g, [c, -s, s, c, 0, 0])

def lane_direction_from_polygon(poly: Polygon):
    """
    Fallback: build an approximate heading vector from the polygon's 'longest axis'.
    Returns: (start_xy, end_xy) in same frame as polygon
    """
    if poly is None or poly.is_empty:
        return None
    # use the long-edge direction of the minimum rotated rectangle
    rect = poly.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    pts = np.stack([xs, ys], axis=1)[:-1]  # 4 corners
    # edges
    edges = pts[(np.arange(4)+1) % 4] - pts
    lens = np.linalg.norm(edges, axis=1)
    i = int(np.argmax(lens))
    v = edges[i] / (lens[i] + 1e-9)

    c = np.array([poly.centroid.x, poly.centroid.y], dtype=float)
    L = max(3.0, min(10.0, float(poly.area) ** 0.5))  # scale moderately
    p0 = c - v * (0.5 * L)
    p1 = c + v * (0.5 * L)
    return p0, p1

def _try_get_centerline_from_map_api(map_api, lane_token: str):
    """
    Try hard to find the lane/lane_connector centerline from map_api.
    Object types/field names vary by project, so try several candidates.
    Returns a shapely LineString on success, None on failure.
    """
    if map_api is None:
        return None

    # 1) nuPlan map api style: get_map_object(token, layer)
    for layer in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
        try:
            obj = map_api.get_map_object(lane_token, layer)
        except Exception:
            obj = None
        if obj is None:
            continue

        # try candidate fields
        for attr in ["baseline_path", "centerline", "reference_line", "polyline", "linestring"]:
            if hasattr(obj, attr):
                cl = getattr(obj, attr)
                # baseline_path may be a list/Path in nuPlan -> may need conversion to a linestring
                # if it is already a shapely LineString, use it as-is
                if isinstance(cl, LineString):
                    return cl
                # if baseline_path is a list of points, build a LineString
                try:
                    if hasattr(cl, "linestring") and isinstance(cl.linestring, LineString):
                        return cl.linestring
                except Exception:
                    pass
                try:
                    # iterable of (x,y)
                    pts = np.array([(p.x, p.y) for p in cl], dtype=float)  # possibly a StateSE2 list
                    if len(pts) >= 2:
                        return LineString(pts)
                except Exception:
                    pass

    # 2) the wrapper may have custom methods
    for fn in ["get_lane_centerline", "get_centerline", "get_baseline_path"]:
        if hasattr(map_api, fn):
            try:
                out = getattr(map_api, fn)(lane_token)
                if isinstance(out, LineString):
                    return out
            except Exception:
                pass
    return None


# -------------------------
# Main draw util
# -------------------------

def add_lane_polygons_with_ids_and_arrows(
    ax: plt.Axes,
    drivable_area_map,
    map_api=None,
    ego_pose_se2=None,
    ego_frame: bool = False,
    radius_m: float = 120.0,
    draw_lane: bool = True,
    draw_lane_connector: bool = True,
    draw_intersection: bool = True,
    max_items: int = 300,
    arrow_every: int = 0,
    id_fontsize: int = 7,
    line_alpha: float = 0.9,
):  
    """
    - draw lane/lane_connector polygons
    - label the token(id) at each polygon centroid
    - if ego_frame=True, draw after global->ego transform relative to ego_pose_se2
    - radius_m: show only the area around ego (requires ego_pose_se2)
    """

    # select which indices to draw
    layers = []
    if draw_lane:
        layers.append(SemanticMapLayer.LANE)
    if draw_lane_connector:
        layers.append(SemanticMapLayer.LANE_CONNECTOR)
    if draw_intersection:
        layers.append(SemanticMapLayer.INTERSECTION)

    # gather the indices for those layers from drivable_area_map
    idxs = []
    for lyr in layers:
        try:
            idxs.extend(drivable_area_map.get_indices_of_map_type([lyr]))
        except Exception:
            pass

    if len(idxs) == 0:
        return

    # ego-centered filter
    def _in_radius(geom):
        if ego_pose_se2 is None:
            return True
        c = geom.centroid
        dx = float(c.x) - float(ego_pose_se2.x)
        dy = float(c.y) - float(ego_pose_se2.y)
        return (dx*dx + dy*dy) <= (radius_m * radius_m)

    # color: consistent, based on a hash of the lane token
    def _color_for_token(tok: str):
        # matplotlib tab20 indexing
        cmap = plt.get_cmap("tab20")
        h = abs(hash(tok)) % 20
        return cmap(h)

    # limit the number of items
    chosen = []
    for i in idxs:
        tok = drivable_area_map.tokens[i]
        poly = drivable_area_map._geometries[i]  
        # poly = None
        if poly is None or poly.is_empty:
            continue
        if ego_pose_se2 is not None and not _in_radius(poly):
            continue
        chosen.append(i)
        if len(chosen) >= max_items:
            break

    for k, i in enumerate(chosen):
        tok = drivable_area_map.tokens[i]
        poly_g = drivable_area_map._geometries[i]
        color = _color_for_token(tok)

        # transform
        poly = _to_ego_frame(poly_g, ego_pose_se2) if (ego_frame and ego_pose_se2 is not None) else poly_g

        # polygon outline
        _plot_polygon(ax, poly, color=color, linewidth=1.8, alpha=line_alpha)

        # id text at centroid
        c = poly.centroid
        ax.text(
            c.x, c.y, str(tok),
            fontsize=id_fontsize,
            color=color,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6, edgecolor="none"),
            zorder=10,
        )

    