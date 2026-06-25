import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from shapely.geometry import Point as ShapelyPoint, LineString
from shapely.geometry import Polygon
from typing import Any, Generator, List, Optional, Set, Tuple, Type, cast, Dict
import os


def extract_ego_polyline_global(scene_dict_list: List[Dict[str, Any]]) -> np.ndarray:
    """
    Return ego polyline as [T,2] global xy from frames.
    """
    xy = []
    for fr in scene_dict_list:
        t = fr["ego2global_translation"]
        xy.append([float(t[0]), float(t[1])])
    return np.asarray(xy, dtype=np.float64)

def _parse_boundary_str_list(coord_str_list):
    """['(x, y)', ...] -> (N,2) float array"""
    pts = []
    for s in coord_str_list:
        try:
            s = s.strip().strip("()")
            x, y = map(float, s.split(","))
            pts.append((x, y))
        except Exception:
            continue
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)

def _downsample_for_arrows(xy, max_points=60):
    """Downsample so we don't draw too many arrows."""
    if xy.shape[0] <= max_points:
        return xy
    idx = np.linspace(0, xy.shape[0] - 1, max_points).astype(int)
    return xy[idx]

def _draw_arrows_along_polyline(ax, xy, color, every=6, head_scale=10.0, alpha=0.9, zorder=5):
    """
    xy: (N,2) polyline
    every: draw an arrow every N points
    """
    if xy is None or xy.shape[0] < 2:
        return
    xy2 = _downsample_for_arrows(xy, max_points=120)

    # segment vectors
    p = xy2[:-1]
    q = xy2[1:]
    v = q - p
    seg_len = np.linalg.norm(v, axis=1) + 1e-9
    vhat = v / seg_len[:, None]

    # sampling
    idx = np.arange(0, vhat.shape[0], every)
    p_sel = p[idx]
    v_sel = vhat[idx]

    ax.quiver(
        p_sel[:, 0], p_sel[:, 1],
        v_sel[:, 0], v_sel[:, 1],
        angles="xy", scale_units="xy", scale=head_scale,  # larger scale -> shorter arrows
        width=0.0035, headwidth=3.5, headlength=4.5,
        color=color, alpha=alpha, zorder=zorder
    )

def _poly_in_radius(poly: Polygon, cx: float, cy: float, r: float) -> bool:
    if poly is None or poly.is_empty:
        return False
    c = poly.centroid
    return (c.x - cx)**2 + (c.y - cy)**2 <= r**2

def _lane_centerline_xy_from_obj(lane_obj):
    """
    Based on V2XMapObject.baseline_path.
    """
    try:
        path = lane_obj.baseline_path
        states = path.discrete_path
        if states is None or len(states) < 2:
            return np.empty((0, 2), dtype=np.float64)
        xy = np.array([[float(s.x), float(s.y)] for s in states], dtype=np.float64)
        return xy
    except Exception:
        return np.empty((0, 2), dtype=np.float64)

def visualize_lane_boundaries_with_arrows_and_legend(
    map_api,
    scene_dict_list,
    radius_m=120.0,
    center_on_history_end=0,
    arrow_every=6,
    legend_mode="type_only",   # "type_only" | "full"
    max_legend_lanes=25,       # only meaningful when legend_mode="full"
    draw_centerline=False,
    show_intersection_polygons=True, 
    out_path=None,
):
    """
    - Draw only lanes within radius_m of the ego.
    - Use a distinct color per lane to show left/right boundaries + direction arrows.
    - Render lane ids as plot text; legend is selected by mode.
    """
    ego_xy = extract_ego_polyline_global(scene_dict_list)
    cx, cy = float(ego_xy[center_on_history_end, 0]), float(ego_xy[center_on_history_end, 1])
    ego_pt = ShapelyPoint(cx, cy)
    
    ################### visualize lanes

    lanes = list(map_api._get_lanes())
    lanes_by_id = {l.id: l for l in lanes if hasattr(l, "id")}

    # Filter lanes within radius: use "lane polygon distance to ego" or centerline min dist
    picked = []
    for lane in lanes:
        poly = lane.polygon
        if poly is None or poly.is_empty:
            continue
        d = float(poly.distance(ego_pt))
        if d <= radius_m:
            picked.append((d, lane))

    # Sort by nearest lane first
    picked.sort(key=lambda x: x[0])
    picked_lanes = [lane for _, lane in picked]

    # Color: use tab20 so colors repeat even with many lanes
    cmap = cm.get_cmap("tab20", 20)

    fig, ax = plt.subplots(figsize=(10, 10))

    # ego traj
    ax.plot(ego_xy[:, 0], ego_xy[:, 1], linewidth=2.0, alpha=0.9, color="k", label="Ego trajectory")

    ################## visualize intersections
    # junctions / intersections
    if show_intersection_polygons:
        for j_idx, jpoly in enumerate(map_api._get_junctions()):
            if jpoly is None or jpoly.is_empty:
                continue

            # radius gate (centered on ego)
            if not _poly_in_radius(jpoly, cx, cy, radius_m):
                continue

            # polygon draw
            x, y = jpoly.exterior.xy
            ax.plot(x, y, linewidth=1.8, alpha=0.85, color="tab:orange")

            # id label at centroid
            # if label_intersections:
            c = jpoly.centroid
            jid = f"junction_{j_idx}"
            ax.text(
                float(c.x), float(c.y), jid,
                fontsize=9, color="tab:orange",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65,
                        edgecolor="tab:orange", linewidth=1.0)
            )


    ############### draw per lane
    lane_handles = []
    lane_labels = []

    for i, lane in enumerate(picked_lanes):
        lane_id = lane.id
        color = cmap(i % 20)

        left_xy  = _parse_boundary_str_list(lane._lane_dict.get("left_boundary", []))
        right_xy = _parse_boundary_str_list(lane._lane_dict.get("right_boundary", []))

        if left_xy.shape[0] >= 2:
            ax.plot(left_xy[:, 0], left_xy[:, 1], linestyle="-", linewidth=1.6, alpha=0.9, color=color)
            _draw_arrows_along_polyline(ax, left_xy, color=color, every=arrow_every, head_scale=12.0, alpha=0.9)

        if right_xy.shape[0] >= 2:
            ax.plot(right_xy[:, 0], right_xy[:, 1], linestyle="--", linewidth=1.6, alpha=0.9, color=color)
            _draw_arrows_along_polyline(ax, right_xy, color=color, every=arrow_every, head_scale=12.0, alpha=0.9)

        # centerline (optional)
        if draw_centerline:
            cl = _lane_centerline_xy_from_obj(lane)
            if cl.shape[0] >= 2:
                ax.plot(cl[:, 0], cl[:, 1], linestyle=":", linewidth=1.0, alpha=0.5, color=color)

        # lane id text: centerline midpoint or polygon centroid
        try:
            cl = _lane_centerline_xy_from_obj(lane)
            if cl.shape[0] >= 2:
                mid = cl[len(cl)//2]
                tx, ty = float(mid[0]), float(mid[1])
            else:
                c = lane.polygon.centroid
                tx, ty = float(c.x), float(c.y)

            ax.text(
                tx, ty, str(lane_id),
                fontsize=8, color=color,  # same color as the lane
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65, edgecolor="none")
            )
        except Exception:
            pass

        # for the legend (per lane)
        if legend_mode == "full":
            # many lanes blow up the legend -> cap it
            if len(lane_handles) < max_legend_lanes:
                lane_handles.append(Line2D([0], [0], color=color, lw=2.5))
                lane_labels.append(str(lane_id))

    # build the legend
    if legend_mode == "type_only":
        type_handles = [
            Line2D([0],[0], color="k", lw=2.0, label="Ego trajectory"),
            Line2D([0],[0], color="0.2", lw=1.8, ls="-",  label="Left boundary (per-lane color)"),
            Line2D([0],[0], color="0.2", lw=1.8, ls="--", label="Right boundary (per-lane color)"),
        ]
        if draw_centerline:
            type_handles.append(Line2D([0],[0], color="0.2", lw=1.2, ls=":", label="Centerline (per-lane color)"))
        ax.legend(handles=type_handles, loc="upper right", framealpha=0.9)

    elif legend_mode == "full":
        # a long lane id list looks bad (hence the max_legend_lanes cap)
        base_handles = [
            Line2D([0],[0], color="k", lw=2.0, label="Ego trajectory"),
            Line2D([0],[0], color="0.2", lw=1.8, ls="-",  label="Left boundary"),
            Line2D([0],[0], color="0.2", lw=1.8, ls="--", label="Right boundary"),
        ]
        ax.legend(handles=base_handles, loc="upper right", framealpha=0.9)

        # per-lane legend goes separately at the bottom
        if lane_handles:
            ax.legend(
                handles=lane_handles,
                labels=lane_labels,
                loc="lower right",
                framealpha=0.9,
                title=f"Lane IDs (closest {len(lane_handles)})",
                fontsize=8,
                title_fontsize=9,
            )
            ax.add_artist(ax.legend(handles=base_handles, loc="upper right", framealpha=0.9))

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(cx - radius_m, cx + radius_m)
    ax.set_ylim(cy - radius_m, cy + radius_m)
    ax.set_title(f"Lane boundaries (arrows) + lane IDs within {radius_m:.0f}m of ego")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()