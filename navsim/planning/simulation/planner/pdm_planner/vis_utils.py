from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject


def _xy_from_state_se2_list(states) -> np.ndarray:
    if states is None or len(states) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([[s.x, s.y] for s in states], dtype=np.float64)


def _plot_shapely_polygon(ax, poly, fill_alpha=0.12, edge_alpha=0.45, lw=1.0) -> bool:
    """
    poly: shapely Polygon or MultiPolygon
    """
    if poly is None:
        return False

    # Shapely import inside to avoid hard dependency if not available
    try:
        from shapely.geometry import Polygon, MultiPolygon
    except Exception:
        return False

    def _plot_one(p: Polygon):
        x, y = p.exterior.xy
        ax.fill(x, y, alpha=fill_alpha)
        ax.plot(x, y, linewidth=lw, alpha=edge_alpha)
        # holes
        for interior in p.interiors:
            xi, yi = interior.xy
            ax.plot(xi, yi, linewidth=lw, alpha=edge_alpha)

    if isinstance(poly, MultiPolygon) or getattr(poly, "geom_type", None) == "MultiPolygon":
        for p in poly.geoms:
            _plot_one(p)
        return True

    if isinstance(poly, Polygon) or getattr(poly, "geom_type", None) == "Polygon":
        _plot_one(poly)
        return True

    return False


def plot_route_plan_with_polygons(
    route_plan: List[LaneGraphEdgeMapObject],
    out_path: Optional[str] = None,
    pad_m: float = 25.0,
    annotate: bool = True,
    draw_polygons: bool = True,
    polygon_fill_alpha: float = 0.10,
    polygon_edge_alpha: float = 0.35,
    polygon_lw: float = 1.0,
    baseline_lw: float = 3.0,
    concat_centerline_lw: float = 5.0,
    show: bool = True,
):
    """
    Visualize:
      - edge polygon (lane + lane connector)
      - per-edge baseline discrete path
      - concatenated centerline (optional)
      - annotations with index/class/id/roadblock/length

    Assumptions:
      - each edge has `polygon` attribute (shapely Polygon/MultiPolygon)
      - each edge has `baseline_path.discrete_path`
    """
    if route_plan is None or len(route_plan) == 0:
        print("route_plan is empty.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    all_xy = []
    concat_states = []

    for i, edge in enumerate(route_plan):
        # ---- polygon ----
        if draw_polygons:
            poly = getattr(edge, "polygon", None)
            _plot_shapely_polygon(
                ax,
                poly,
                fill_alpha=polygon_fill_alpha,
                edge_alpha=polygon_edge_alpha,
                lw=polygon_lw,
            )

        # ---- baseline discrete path ----
        try:
            dpath = edge.baseline_path.discrete_path
        except Exception:
            dpath = None

        xy = _xy_from_state_se2_list(dpath)
        if xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], linewidth=baseline_lw, alpha=0.9)
            all_xy.append(xy)
            concat_states.extend(dpath)

            if annotate:
                cls = type(edge).__name__
                eid = getattr(edge, "id", "<?>")
                rb = "<?>"
                if hasattr(edge, "get_roadblock_id"):
                    try:
                        rb = edge.get_roadblock_id()
                    except Exception:
                        pass
                length = None
                if hasattr(edge, "baseline_path") and hasattr(edge.baseline_path, "length"):
                    try:
                        length = float(edge.baseline_path.length)
                    except Exception:
                        length = None

                p = xy[0]
                label = f"{i}: {cls}\nid={eid}\nrb={rb}"
                if length is not None:
                    label += f"\nlen={length:.1f}m"
                ax.text(p[0], p[1], label, fontsize=8, alpha=0.9)

                # start-point marker
                ax.scatter([p[0]], [p[1]], s=18)

    # ---- concat centerline ----
    cxy = _xy_from_state_se2_list(concat_states)
    if cxy.shape[0] >= 2:
        ax.plot(cxy[:, 0], cxy[:, 1], linewidth=concat_centerline_lw, alpha=0.45, label="concat centerline")
        ax.scatter([cxy[0, 0]], [cxy[0, 1]], s=40, label="start")
        ax.scatter([cxy[-1, 0]], [cxy[-1, 1]], s=40, label="end")

    # ---- axis limits ----
    if len(all_xy) > 0:
        pts = np.vstack(all_xy)
    else:
        pts = cxy

    if pts.shape[0] > 0:
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        ax.set_xlim(xmin - pad_m, xmax + pad_m)
        ax.set_ylim(ymin - pad_m, ymax + pad_m)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title("route_plan visualization: polygons + per-edge baseline + concat centerline")
    ax.legend(loc="best")

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"Saved to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def debug_plot_lane_boundaries_and_baseline(
    current_lane,
    out_path: str = None,
    title: str = None,
    show_headings: bool = True,
    heading_every: int = 1,
    arrow_len: float = 3.0,
    annotate_idx: bool = True,
    annotate_step: int = 1,

    # NEW: zoom options
    zoom_mode: str = "bounds",   # "bounds" | "radius"
    radius_m: float = 30.0,      # used when zoom_mode="radius"
    pad_m: float = 5.0,          # padding when zoom_mode="bounds"
    equal_aspect: bool = True,
):
    """
    Plot lane boundaries (raw/resampled) and baseline(centerline).
    zoom_mode:
      - "bounds": auto-zoom based on the data bounding box (recommended)
      - "radius": zoom to radius_m around the baseline start
    """

    # ---------- raw boundaries ----------
    left_raw = current_lane._parse_boundary(current_lane._lane_dict.get("left_boundary", []))
    right_raw = current_lane._parse_boundary(current_lane._lane_dict.get("right_boundary", []))

    if left_raw is None: left_raw = np.zeros((0, 2), dtype=float)
    if right_raw is None: right_raw = np.zeros((0, 2), dtype=float)

    # ---------- resample like baseline_path does ----------
    left_rs = None
    right_rs = None
    if len(left_raw) >= 2 and len(right_raw) >= 2:
        n_pts = max(len(left_raw), len(right_raw))
        t_uniform = np.linspace(0, 1, n_pts)

        t_left = current_lane._arc_length_param(left_raw)
        left_rs = np.column_stack([
            np.interp(t_uniform, t_left, left_raw[:, 0]),
            np.interp(t_uniform, t_left, left_raw[:, 1]),
        ])

        t_right = current_lane._arc_length_param(right_raw)
        right_rs = np.column_stack([
            np.interp(t_uniform, t_right, right_raw[:, 0]),
            np.interp(t_uniform, t_right, right_raw[:, 1]),
        ])

    # ---------- baseline ----------
    base_states = current_lane.baseline_path.discrete_path
    base_xy = np.array([[float(s.x), float(s.y)] for s in base_states], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 8))

    # raw boundaries
    if len(left_raw) >= 2:
        ax.plot(left_raw[:, 0], left_raw[:, 1], linewidth=2.0, label="left_boundary raw")
        ax.scatter(left_raw[:, 0], left_raw[:, 1], s=18, marker="o")
    if len(right_raw) >= 2:
        ax.plot(right_raw[:, 0], right_raw[:, 1], linewidth=2.0, label="right_boundary raw")
        ax.scatter(right_raw[:, 0], right_raw[:, 1], s=18, marker="o")

    # resampled boundaries (thin)
    if left_rs is not None and len(left_rs) >= 2:
        ax.plot(left_rs[:, 0], left_rs[:, 1], linewidth=1.2, alpha=0.9, linestyle="--",
                label="left_boundary resampled")
    if right_rs is not None and len(right_rs) >= 2:
        ax.plot(right_rs[:, 0], right_rs[:, 1], linewidth=1.2, alpha=0.9, linestyle="--",
                label="right_boundary resampled")

    # baseline
    ax.plot(base_xy[:, 0], base_xy[:, 1], linewidth=3.0, marker="x", markersize=7,
            label="baseline(centerline)")
    ax.scatter([base_xy[0, 0]], [base_xy[0, 1]], s=120, marker="s", label="baseline start")
    ax.scatter([base_xy[-1, 0]], [base_xy[-1, 1]], s=120, marker="X", label="baseline end")

    # heading arrows
    if show_headings:
        for i in range(0, len(base_states), max(1, heading_every)):
            s = base_states[i]
            x, y, h = float(s.x), float(s.y), float(s.heading)
            dx = np.cos(h) * arrow_len
            dy = np.sin(h) * arrow_len
            ax.arrow(x, y, dx, dy, length_includes_head=True,
                     head_width=0.8, head_length=1.2, alpha=0.8)

    # annotate indices
    if annotate_idx:
        def _annot(arr, prefix, step):
            if arr is None: return
            for i in range(0, len(arr), max(1, step)):
                ax.text(arr[i, 0], arr[i, 1], f"{prefix}{i}", fontsize=9)
        _annot(left_raw, "L", annotate_step)
        _annot(right_raw, "R", annotate_step)
        for i in range(0, len(base_xy), max(1, annotate_step)):
            ax.text(base_xy[i, 0], base_xy[i, 1], f"C{i}", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    # ---------- zoom ----------
    if zoom_mode == "bounds":
        all_pts = []
        if len(left_raw) > 0: all_pts.append(left_raw)
        if len(right_raw) > 0: all_pts.append(right_raw)
        if base_xy is not None and len(base_xy) > 0: all_pts.append(base_xy)
        if left_rs is not None and len(left_rs) > 0: all_pts.append(left_rs)
        if right_rs is not None and len(right_rs) > 0: all_pts.append(right_rs)

        pts = np.vstack(all_pts) if len(all_pts) else base_xy
        minx, miny = np.min(pts[:, 0]), np.min(pts[:, 1])
        maxx, maxy = np.max(pts[:, 0]), np.max(pts[:, 1])

        # guard against very flat lanes (tall but narrow): enforce a minimum width/height
        w = maxx - minx
        h = maxy - miny
        min_span = 10.0  # meters
        if w < min_span:
            cx = 0.5 * (minx + maxx)
            minx, maxx = cx - 0.5 * min_span, cx + 0.5 * min_span
        if h < min_span:
            cy = 0.5 * (miny + maxy)
            miny, maxy = cy - 0.5 * min_span, cy + 0.5 * min_span

        ax.set_xlim(minx - pad_m, maxx + pad_m)
        ax.set_ylim(miny - pad_m, maxy + pad_m)

    elif zoom_mode == "radius":
        cx0, cy0 = base_xy[0, 0], base_xy[0, 1]
        ax.set_xlim(cx0 - radius_m, cx0 + radius_m)
        ax.set_ylim(cy0 - radius_m, cy0 + radius_m)

    if equal_aspect:
        ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if title is None:
        title = f"Lane boundaries + baseline | lane_id={getattr(current_lane,'id',None)}"
    ax.set_title(title)
    ax.legend(loc="best")

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def debug_plot_lane_and_proposals(
    current_lane,
    proposal_paths=None,          # List[PDMPath] (baseline only if None)
    out_path: str = None,
    title: str = None,

    # baseline heading/annotation
    show_headings: bool = False,
    heading_every: int = 1,
    arrow_len: float = 3.0,
    annotate_idx: bool = False,
    annotate_step: int = 1,

    # zoom
    zoom_mode: str = "bounds",    # "bounds" | "radius"
    radius_m: float = 30.0,
    pad_m: float = 5.0,
    equal_aspect: bool = True,

    # proposal plotting
    plot_proposal_points: bool = True,
    proposal_marker_size: float = 10.0,
):
    """
    Plot:
      - lane left/right boundary (raw + resampled)
      - lane baseline(centerline)
      - proposal paths (PDMPath list): each path.linestring or path.discrete_path

    Also prints/annotates ring/closed status for each proposal.
    """

    # ---------- raw boundaries ----------
    left_raw = current_lane._parse_boundary(current_lane._lane_dict.get("left_boundary", []))
    right_raw = current_lane._parse_boundary(current_lane._lane_dict.get("right_boundary", []))

    if left_raw is None: left_raw = np.zeros((0, 2), dtype=float)
    if right_raw is None: right_raw = np.zeros((0, 2), dtype=float)

    # ---------- resample like baseline_path does ----------
    left_rs = None
    right_rs = None
    if len(left_raw) >= 2 and len(right_raw) >= 2:
        n_pts = max(len(left_raw), len(right_raw))
        t_uniform = np.linspace(0, 1, n_pts)

        t_left = current_lane._arc_length_param(left_raw)
        left_rs = np.column_stack([
            np.interp(t_uniform, t_left, left_raw[:, 0]),
            np.interp(t_uniform, t_left, left_raw[:, 1]),
        ])

        t_right = current_lane._arc_length_param(right_raw)
        right_rs = np.column_stack([
            np.interp(t_uniform, t_right, right_raw[:, 0]),
            np.interp(t_uniform, t_right, right_raw[:, 1]),
        ])

    # ---------- baseline ----------
    base_states = current_lane.baseline_path.discrete_path
    base_xy = np.array([[float(s.x), float(s.y)] for s in base_states], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 9))

    # raw boundaries
    if len(left_raw) >= 2:
        ax.plot(left_raw[:, 0], left_raw[:, 1], linewidth=2.0, label="left_boundary raw")
        ax.scatter(left_raw[:, 0], left_raw[:, 1], s=18, marker="o")
    if len(right_raw) >= 2:
        ax.plot(right_raw[:, 0], right_raw[:, 1], linewidth=2.0, label="right_boundary raw")
        ax.scatter(right_raw[:, 0], right_raw[:, 1], s=18, marker="o")

    # resampled boundaries (thin)
    if left_rs is not None and len(left_rs) >= 2:
        ax.plot(left_rs[:, 0], left_rs[:, 1], linewidth=1.2, alpha=0.9, linestyle="--",
                label="left_boundary resampled")
    if right_rs is not None and len(right_rs) >= 2:
        ax.plot(right_rs[:, 0], right_rs[:, 1], linewidth=1.2, alpha=0.9, linestyle="--",
                label="right_boundary resampled")

    # baseline
    ax.plot(base_xy[:, 0], base_xy[:, 1], linewidth=3.0, marker="x", markersize=7,
            label="baseline(centerline)")
    ax.scatter([base_xy[0, 0]], [base_xy[0, 1]], s=120, marker="s", label="baseline start")
    ax.scatter([base_xy[-1, 0]], [base_xy[-1, 1]], s=120, marker="X", label="baseline end")

    # heading arrows
    if show_headings:
        for i in range(0, len(base_states), max(1, heading_every)):
            s = base_states[i]
            x, y, h = float(s.x), float(s.y), float(s.heading)
            dx = np.cos(h) * arrow_len
            dy = np.sin(h) * arrow_len
            ax.arrow(x, y, dx, dy, length_includes_head=True,
                     head_width=0.8, head_length=1.2, alpha=0.8)

    # annotate indices
    if annotate_idx:
        def _annot(arr, prefix, step):
            if arr is None or len(arr) == 0: 
                return
            for i in range(0, len(arr), max(1, step)):
                ax.text(arr[i, 0], arr[i, 1], f"{prefix}{i}", fontsize=9)
        _annot(left_raw, "L", annotate_step)
        _annot(right_raw, "R", annotate_step)
        for i in range(0, len(base_xy), max(1, annotate_step)):
            ax.text(base_xy[i, 0], base_xy[i, 1], f"C{i}", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    # ---------- proposal paths overlay ----------
    proposal_info_lines = []
    if proposal_paths is not None:
        for i, path in enumerate(proposal_paths):
            # try to get shapely linestring
            ls = getattr(path, "linestring", None)

            if ls is not None and getattr(ls, "geom_type", None) == "LineString":
                coords = np.array(ls.coords, dtype=float)  # [N,2]
                is_ring = getattr(ls, "is_ring", None)
                is_closed = getattr(ls, "is_closed", None) if hasattr(ls, "is_closed") else None

                proposal_info_lines.append(
                    f"proposal[{i}] coords={len(coords)} ring={is_ring} closed={is_closed}"
                )

                ax.plot(coords[:, 0], coords[:, 1], linewidth=2.2, alpha=0.9,
                        label=f"proposal[{i}] (ring={is_ring})")
                if plot_proposal_points:
                    ax.scatter(coords[:, 0], coords[:, 1], s=proposal_marker_size, alpha=0.7)

                # mark start/end of this proposal
                ax.scatter([coords[0, 0]], [coords[0, 1]], s=80, marker="s", alpha=0.9)
                ax.scatter([coords[-1, 0]], [coords[-1, 1]], s=80, marker="X", alpha=0.9)

            else:
                # fallback: try discrete_path
                dp = getattr(path, "discrete_path", None)
                if dp is None:
                    proposal_info_lines.append(f"proposal[{i}] (no linestring/discrete_path)")
                    continue
                coords = np.array([[float(s.x), float(s.y)] for s in dp], dtype=float)
                closed = (len(coords) >= 2 and np.allclose(coords[0], coords[-1]))
                proposal_info_lines.append(
                    f"proposal[{i}] dp={len(coords)} closed_by_points={closed}"
                )
                ax.plot(coords[:, 0], coords[:, 1], linewidth=2.2, alpha=0.9,
                        label=f"proposal[{i}] (closed={closed})")
                if plot_proposal_points:
                    ax.scatter(coords[:, 0], coords[:, 1], s=proposal_marker_size, alpha=0.7)

    # ---------- zoom ----------
    if zoom_mode == "bounds":
        all_pts = []
        if len(left_raw) > 0: all_pts.append(left_raw)
        if len(right_raw) > 0: all_pts.append(right_raw)
        if base_xy is not None and len(base_xy) > 0: all_pts.append(base_xy)
        if left_rs is not None and len(left_rs) > 0: all_pts.append(left_rs)
        if right_rs is not None and len(right_rs) > 0: all_pts.append(right_rs)

        # include proposals in bounds
        if proposal_paths is not None:
            for path in proposal_paths:
                ls = getattr(path, "linestring", None)
                if ls is not None and getattr(ls, "geom_type", None) == "LineString":
                    all_pts.append(np.array(ls.coords, dtype=float))
                else:
                    dp = getattr(path, "discrete_path", None)
                    if dp is not None:
                        all_pts.append(np.array([[float(s.x), float(s.y)] for s in dp], dtype=float))

        pts = np.vstack(all_pts) if len(all_pts) else base_xy
        minx, miny = np.min(pts[:, 0]), np.min(pts[:, 1])
        maxx, maxy = np.max(pts[:, 0]), np.max(pts[:, 1])

        # enforce a minimum span
        w = maxx - minx
        h = maxy - miny
        min_span = 10.0
        if w < min_span:
            cx = 0.5 * (minx + maxx)
            minx, maxx = cx - 0.5 * min_span, cx + 0.5 * min_span
        if h < min_span:
            cy = 0.5 * (miny + maxy)
            miny, maxy = cy - 0.5 * min_span, cy + 0.5 * min_span

        ax.set_xlim(minx - pad_m, maxx + pad_m)
        ax.set_ylim(miny - pad_m, maxy + pad_m)

    elif zoom_mode == "radius":
        cx0, cy0 = base_xy[0, 0], base_xy[0, 1]
        ax.set_xlim(cx0 - radius_m, cx0 + radius_m)
        ax.set_ylim(cy0 - radius_m, cy0 + radius_m)

    if equal_aspect:
        ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if title is None:
        lane_id = getattr(current_lane, "id", None)
        title = f"Lane + baseline + proposals | lane_id={lane_id}"
    if proposal_info_lines:
        title += "\n" + " | ".join(proposal_info_lines[:3])  # keep only the first 3 if too long
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_centerline_discrete_path(centerline_discrete_path, out_path=None, pad_m=20.0):
    xy = np.array([[p.x, p.y] for p in centerline_discrete_path], dtype=np.float64)
    if len(xy) == 0:
        print("Empty centerline_discrete_path")
        return

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(xy[:, 0], xy[:, 1], linewidth=3.0, alpha=0.9, label="centerline_discrete_path")

    # mark start / end
    ax.scatter([xy[0, 0]], [xy[0, 1]], s=40, label="start")
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=40, label="end")

    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    ax.set_xlim(xmin - pad_m, xmax + pad_m)
    ax.set_ylim(ymin - pad_m, ymax + pad_m)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_title("Centerline (discrete StateSE2 path)")

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"Saved to: {out_path}")
        

def plot_proposal_paths(
    proposal_paths,
    centerline_discrete_path=None,
    ego_state=None,
    out_path: str = "exp_debug/proposal_paths.png",
    title: str = "PDM Proposal Paths",
    figsize: tuple = (14, 10),
):
    """
    Visualize a list of PDMPath objects.
    proposal_paths[0] is the centerline; the rest are lateral-offset paths.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(proposal_paths), 1)))

    for idx, path in enumerate(proposal_paths):
        try:
            total_length = path._progress[-1] if hasattr(path, '_progress') and len(path._progress) > 0 else 0.0
            if total_length < 1e-3:
                continue
            num_samples = max(int(total_length / 0.3), 10)
            sample_progress = np.linspace(0.0, total_length, num_samples)
            states = path.interpolate(sample_progress)
            xs, ys, headings = states[:, 0], states[:, 1], states[:, 2]
        except Exception:
            try:
                line = path._linestring
                xs = np.array(line.coords.xy[0])
                ys = np.array(line.coords.xy[1])
                headings = None
            except Exception:
                continue

        color = colors[idx % len(colors)]

        if idx == 0:
            ax.plot(xs, ys, color='black', linewidth=2.5, linestyle='--', label='Centerline', zorder=5)
            if headings is not None:
                arrow_every = max(len(xs) // 15, 1)
                for i in range(0, len(xs), arrow_every):
                    dx = np.cos(headings[i]) * 0.8
                    dy = np.sin(headings[i]) * 0.8
                    ax.annotate('', xy=(xs[i]+dx, ys[i]+dy), xytext=(xs[i], ys[i]),
                                arrowprops=dict(arrowstyle='->', color='black', lw=1.0))
        else:
            ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.8, label=f'Offset path {idx}', zorder=3)

        ax.plot(xs[0], ys[0], 'o', color=color, markersize=5, zorder=6)
        ax.plot(xs[-1], ys[-1], 's', color=color, markersize=5, zorder=6)

    # centerline discrete points (for comparing pre/post interpolation)
    if centerline_discrete_path is not None:
        cxs = [s.x for s in centerline_discrete_path]
        cys = [s.y for s in centerline_discrete_path]
        ax.scatter(cxs, cys, c='gray', s=8, alpha=0.4, zorder=2, label='Centerline pts')

    if ego_state is not None:
        try:
            ex, ey = ego_state.rear_axle.x, ego_state.rear_axle.y
            eh = ego_state.rear_axle.heading
            ax.plot(ex, ey, 'r*', markersize=15, zorder=10, label='Ego')
            dx = np.cos(eh) * 2.0
            dy = np.sin(eh) * 2.0
            ax.annotate('', xy=(ex+dx, ey+dy), xytext=(ex, ey),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2.0))
        except Exception:
            pass

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[vis] Proposal paths saved to {out_path}")


def plot_proposal_trajectories(
    proposal_states,
    ego_state=None,
    out_path: str = "exp_debug/proposal_trajectories.png",
    title: str = "PDM Proposal Trajectories",
    figsize: tuple = (14, 10),
    max_proposals: int = 50,
    show_sample_points: bool = True,   # mark 0.5s sample points with 'x'
):
    """
    Visualize the proposal trajectories rolled out by PDMGenerator.
    proposal_states: (N_proposals, T_steps, state_dim) numpy array
    show_sample_points: if True, mark each 0.5s sample position with 'x'
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    n_proposals = proposal_states.shape[0]
    n_draw = min(n_proposals, max_proposals)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_draw))

    for idx in range(n_draw):
        xs = proposal_states[idx, :, 0]
        ys = proposal_states[idx, :, 1]
        color = colors[idx]

        # drop trailing zeros (keep up to the last valid timestep)
        valid = np.any(proposal_states[idx] != 0, axis=-1)
        valid_end = int(np.where(valid)[0][-1]) + 1 if valid.any() else len(xs)
        xs_v = xs[:valid_end]
        ys_v = ys[:valid_end]

        # connect with a line
        ax.plot(xs_v, ys_v, color=color, linewidth=1.0, alpha=0.5, zorder=3)

        if show_sample_points:
            # intermediate points (excluding start/end): 'x' marker
            ax.scatter(
                xs_v[1:-1], ys_v[1:-1],
                marker='x', s=30, color=color, alpha=0.8,
                linewidths=1.0, zorder=5,
            )

        # start point: circle, end point: square
        ax.plot(xs_v[0], ys_v[0], 'o', color=color, markersize=4, zorder=6)
        ax.plot(xs_v[-1], ys_v[-1], 's', color=color, markersize=4, zorder=6)

    if ego_state is not None:
        try:
            ex, ey = ego_state.rear_axle.x, ego_state.rear_axle.y
            eh = ego_state.rear_axle.heading
            ax.plot(ex, ey, 'r*', markersize=15, zorder=10, label='Ego')
            ax.annotate('', xy=(ex + np.cos(eh)*2, ey + np.sin(eh)*2), xytext=(ex, ey),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2.0))
        except Exception:
            pass

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=5, label='Start'),
        Line2D([0], [0], marker='s', color='gray', linestyle='None', markersize=5, label='End'),
        Line2D([0], [0], marker='x', color='gray', linestyle='None', markersize=6, label='0.5s sample pts'),
    ]
    if ego_state:
        legend_elements.append(Line2D([0], [0], marker='*', color='red', linestyle='None', markersize=10, label='Ego'))
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"{title} ({n_draw}/{n_proposals} shown)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[vis] Proposal trajectories saved to {out_path}")
    
def plot_proposal_refined_trajectories(
    proposal_states,
    ego_state=None,
    out_path: str = "exp_debug/proposal_trajectories.png",
    title: str = "PDM Proposal Trajectories",
    figsize: tuple = (14, 10),
    max_proposals: int = 50,
    upsample_factor: int = 10,   # added: upsample each segment 10x
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.interpolate import CubicSpline

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    n_proposals = proposal_states.shape[0]
    n_draw = min(n_proposals, max_proposals)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_draw))

    T = proposal_states.shape[1]
    t_orig = np.arange(T, dtype=np.float64)
    t_fine = np.linspace(0, T - 1, (T - 1) * upsample_factor + 1)

    for idx in range(n_draw):
        xs_raw = proposal_states[idx, :, 0]
        ys_raw = proposal_states[idx, :, 1]

        # use valid points only (drop all-zero trailing entries)
        valid = np.any(proposal_states[idx] != 0, axis=-1)
        valid_end = int(np.where(valid)[0][-1]) + 1 if valid.any() else T
        if valid_end < 2:
            continue

        t_v = t_orig[:valid_end]
        t_f = np.linspace(0, valid_end - 1, (valid_end - 1) * upsample_factor + 1)

        try:
            cs_x = CubicSpline(t_v, xs_raw[:valid_end], bc_type='natural')
            cs_y = CubicSpline(t_v, ys_raw[:valid_end], bc_type='natural')
            xs = cs_x(t_f)
            ys = cs_y(t_f)
        except Exception:
            xs, ys = xs_raw[:valid_end], ys_raw[:valid_end]

        ax.plot(xs, ys, color=colors[idx], linewidth=1.0, alpha=0.6)
        ax.plot(xs[0], ys[0], 'o', color=colors[idx], markersize=3, zorder=4)
        ax.plot(xs[-1], ys[-1], 's', color=colors[idx], markersize=3, zorder=4)

    if ego_state is not None:
        try:
            ex, ey = ego_state.rear_axle.x, ego_state.rear_axle.y
            eh = ego_state.rear_axle.heading
            ax.plot(ex, ey, 'r*', markersize=15, zorder=10, label='Ego')
            ax.annotate('', xy=(ex + np.cos(eh)*2, ey + np.sin(eh)*2), xytext=(ex, ey),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2.0))
        except Exception:
            pass

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"{title} ({n_draw}/{n_proposals} shown)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    if ego_state:
        ax.legend(loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[vis] Proposal trajectories saved to {out_path}")
    
