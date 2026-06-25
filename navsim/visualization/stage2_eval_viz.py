import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import affinity
from shapely.geometry import LineString

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.common.maps.abstract_map import SemanticMapLayer

from navsim.common.dataclasses import Trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from navsim.visualization.bev import add_linestring_to_bev_ax, add_map_to_bev_ax, add_oriented_box_to_bev_ax
from navsim.visualization.config import AGENT_CONFIG, BEV_PLOT_CONFIG, MAP_LAYER_CONFIG

logger = logging.getLogger(__name__)


def get_map_api(metric_cache: MetricCache, map_root: Optional[Path]):
    map_root = Path(map_root) if map_root else Path(metric_cache.map_parameters.map_root)
    if metric_cache.map_parameters.map_name == "v2x_real":
        from navsim.common.dataloader_v2xreal import V2XRealMapWrapper

        return V2XRealMapWrapper(map_root)

    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    return get_maps_api(map_root, metric_cache.map_parameters.map_version, metric_cache.map_parameters.map_name)


def transform_box_to_ego(box: OrientedBox, ego_pose: StateSE2) -> OrientedBox:
    dx = box.center.x - ego_pose.x
    dy = box.center.y - ego_pose.y
    cos_h = np.cos(-ego_pose.heading)
    sin_h = np.sin(-ego_pose.heading)
    x_local = dx * cos_h - dy * sin_h
    y_local = dx * sin_h + dy * cos_h
    heading_local = normalize_angle(box.center.heading - ego_pose.heading)
    return OrientedBox(StateSE2(x_local, y_local, heading_local), box.length, box.width, box.height)


def transform_linestring_to_ego(linestring: LineString, ego_pose: StateSE2) -> LineString:
    a = np.cos(ego_pose.heading)
    b = np.sin(ego_pose.heading)
    d = -np.sin(ego_pose.heading)
    e = np.cos(ego_pose.heading)
    xoff = -ego_pose.x
    yoff = -ego_pose.y
    translated = affinity.affine_transform(linestring, [1, 0, 0, 1, xoff, yoff])
    return affinity.affine_transform(translated, [a, b, d, e, 0, 0])


def iter_tracked_objects(detections_tracks):
    if detections_tracks is None:
        return []
    if isinstance(detections_tracks, TrackedObjects):
        return detections_tracks.tracked_objects
    if hasattr(detections_tracks, "tracked_objects"):
        return detections_tracks.tracked_objects.tracked_objects
    return []


def add_tracks(ax: plt.Axes, detections_tracks, ego_pose: StateSE2, alpha: float = 1.0) -> None:
    for tracked_object in iter_tracked_objects(detections_tracks):
        agent_type = tracked_object.tracked_object_type
        config = dict(AGENT_CONFIG.get(agent_type, AGENT_CONFIG[TrackedObjectType.GENERIC_OBJECT]))
        config["fill_color_alpha"] = config.get("fill_color_alpha", 1.0) * alpha
        config["line_color_alpha"] = config.get("line_color_alpha", 1.0) * alpha
        local_box = transform_box_to_ego(tracked_object.box, ego_pose)
        add_oriented_box_to_bev_ax(ax, local_box, config)


def add_trajectory_to_ax(
    ax: plt.Axes,
    poses: np.ndarray,
    color: str,
    label: str,
    linewidth: float = 2.0,
    marker: str = "o",
) -> None:
    if poses is None or len(poses) == 0:
        return
    ax.plot(poses[:, 1], poses[:, 0], color=color, linewidth=linewidth, label=label, marker=marker, markersize=4)


def _format_metric_block(
    prefix: str,
    metric_values: Dict[str, float],
    metric_rows: List[List[Tuple[str, str]]],
    metric_precision: Dict[str, int],
) -> List[str]:
    """Format metric summary as multiple fixed-width rows."""

    def _fmt_pair(label: str, key: str) -> str:
        value = float(metric_values.get(key, float("nan")))
        precision = metric_precision.get(key, 3)
        return f"{label}={value:.{precision}f}"

    lines: List[str] = []
    indent = " " * (len(prefix) + 1)
    for row_idx, row_keys in enumerate(metric_rows):
        row_text = "  ".join(_fmt_pair(label, key) for label, key in row_keys)
        lines.append(f"{prefix} {row_text}" if row_idx == 0 else f"{indent}{row_text}")
    return lines


def _global_to_ego(x_global, y_global, ego_pose: StateSE2):
    """Transform global (x, y) array to ego-local frame."""
    cos_h = np.cos(-ego_pose.heading)
    sin_h = np.sin(-ego_pose.heading)
    dx = np.asarray(x_global) - ego_pose.x
    dy = np.asarray(y_global) - ego_pose.y
    return dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h


def visualize_prediction_two_stage(
    metric_cache: MetricCache,
    pred_trajectory: Trajectory,
    stage1_row: pd.Series,
    stage2_data: Optional[Dict],  # {offset_name: {"simulated_states", "trajectory", "start_x", "start_y", "weight", "metrics"}}
    combined_row: Optional[pd.Series],
    output_path: Path,
    map_root: Optional[Path] = None,
    simulated_states: Optional[np.ndarray] = None,
    simulated_tracks: Optional[list] = None,
) -> None:
    ego_state = metric_cache.ego_state
    ego_pose = ego_state.rear_axle

    # ── figure layout: BEV (left 60%) + metrics panel (right 40%) ──────────
    fig = plt.figure(figsize=(22, 10))
    ax = fig.add_axes([0.01, 0.04, 0.56, 0.92])  # BEV
    ax_m = fig.add_axes([0.59, 0.04, 0.40, 0.92])  # metrics

    # ── BEV ────────────────────────────────────────────────────────────────
    ax.set_facecolor(BEV_PLOT_CONFIG["background_color"])
    ax.set_aspect("equal")

    try:
        map_api = get_map_api(metric_cache, map_root)
        add_map_to_bev_ax(ax, map_api, StateSE2(ego_pose.x, ego_pose.y, ego_pose.heading))
    except Exception as exc:
        logger.warning(f"Map load failed: {exc}")

    if metric_cache.centerline is not None:
        centerline_local = transform_linestring_to_ego(metric_cache.centerline.linestring, ego_pose)
        add_linestring_to_bev_ax(ax, centerline_local, MAP_LAYER_CONFIG[SemanticMapLayer.ROADBLOCK])

    if metric_cache.current_tracked_objects:
        add_tracks(ax, metric_cache.current_tracked_objects[0], ego_pose, alpha=1.0)
    # Future background traffic: use the policy-propagated tracks when provided
    # (so an idm run actually shows idm-reactive agents, not the GT/log_replay
    # tracks from the cache); otherwise fall back to the cache's GT future.
    if simulated_tracks is not None:
        _future_tracks = list(simulated_tracks)[1:]  # [0] is the t=0 state drawn above
    else:
        _future_tracks = metric_cache.future_tracked_objects
    for future in _future_tracks[::5]:
        add_tracks(ax, future, ego_pose, alpha=0.15)

    ego_box_local = transform_box_to_ego(ego_state.car_footprint.oriented_box, ego_pose)
    add_oriented_box_to_bev_ax(ax, ego_box_local, AGENT_CONFIG[TrackedObjectType.EGO])

    if metric_cache.human_trajectory is not None:
        human_poses = metric_cache.human_trajectory.poses
        if len(human_poses) > 0:
            add_trajectory_to_ax(
                ax,
                human_poses[:, :2],
                color="#2ca02c",
                label="Human (GT)",
                linewidth=2.5,
                marker="s",
            )

    # Stage1 trajectory
    if pred_trajectory is not None and pred_trajectory.poses is not None:
        pred_poses = pred_trajectory.poses
        if len(pred_poses) > 0:
            add_trajectory_to_ax(ax, pred_poses[:, :2], color="#d62728", label="S1 Agent", linewidth=2.5, marker="o")

    # LQR simulated states
    if simulated_states is not None and len(simulated_states) > 0:
        sim_x_ego, sim_y_ego = _global_to_ego(simulated_states[:, StateIndex.X], simulated_states[:, StateIndex.Y], ego_pose)
        ax.plot(sim_y_ego, sim_x_ego, color="#ff7f0e", linewidth=2.0, label="LQR Simulated", marker="D", markersize=3, zorder=5)

    # Stage2 per-offset start points + trajectories
    if stage2_data:
        offsets_sorted = sorted(stage2_data.items(), key=lambda kv: kv[1]["weight"], reverse=True)
        weights = np.array([v["weight"] for _, v in offsets_sorted])
        pdm_scores = np.array([float(v["metrics"].get("pdm_score", 0.0)) for _, v in offsets_sorted])

        # colormap: pdm_score → color (red=0, green=1)
        cmap = plt.get_cmap("RdYlGn")

        # Draw trajectories (thin, alpha by weight) + collect for weighted avg
        wavg_xy_ego = None
        for k, (offset_name, od) in enumerate(offsets_sorted):
            traj = od.get("trajectory")
            sim_states = od.get("simulated_states")
            sx_ego, sy_ego = _global_to_ego(od["start_x"], od["start_y"], ego_pose)
            pdm_c = np.clip(pdm_scores[k], 0.0, 1.0)
            color = cmap(pdm_c)
            alpha = min(1.0, max(0.2, weights[k] * 4.0))
            heading = float(od.get("heading", ego_pose.heading))

            # draw start point
            ax.scatter(sy_ego, sx_ego, s=80 + weights[k] * 300, color=color, edgecolors="k", linewidths=0.5, zorder=6, alpha=0.85)
            # draw heading direction at start point
            heading_len = 1.0
            hx_global = od["start_x"] + heading_len * np.cos(heading)
            hy_global = od["start_y"] + heading_len * np.sin(heading)
            hx_ego, hy_ego = _global_to_ego(hx_global, hy_global, ego_pose)
            ax.plot([sy_ego, hy_ego], [sx_ego, hx_ego], color=color, linewidth=1.4, zorder=7)

            if sim_states is not None and len(sim_states) > 0:
                tx = np.asarray(sim_states[:, StateIndex.X], dtype=np.float64)
                ty = np.asarray(sim_states[:, StateIndex.Y], dtype=np.float64)
                lx, ly = _global_to_ego(tx, ty, ego_pose)
                ax.plot(ly, lx, color=color, linewidth=1.0, alpha=alpha, zorder=4)

                # accumulate for weighted average
                xy = np.stack([lx, ly], axis=1) * weights[k]
                wavg_xy_ego = xy if wavg_xy_ego is None else wavg_xy_ego + xy
            elif traj is not None and traj.poses is not None and len(traj.poses) > 0:
                # transform stage2 local trajectory → global → stage1 ego
                s2_ego_pose = StateSE2(od["start_x"], od["start_y"], heading)
                abs_poses = relative_to_absolute_poses(s2_ego_pose, [StateSE2(*p) for p in traj.poses])
                tx = np.array([p.x for p in abs_poses])
                ty = np.array([p.y for p in abs_poses])
                lx, ly = _global_to_ego(tx, ty, ego_pose)
                ax.plot(ly, lx, color=color, linewidth=1.0, alpha=alpha, zorder=4)

                # accumulate for weighted average
                xy = np.stack([lx, ly], axis=1) * weights[k]
                wavg_xy_ego = xy if wavg_xy_ego is None else wavg_xy_ego + xy

        # colorbar legend for pdm_score
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, shrink=0.4, aspect=15)
        cbar.set_label("S2 PDM", fontsize=7)

    margin_x, margin_y = BEV_PLOT_CONFIG["figure_margin"]
    ax.set_xlim([-margin_y, margin_y])
    ax.set_ylim([-margin_x, margin_x])
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Token: {output_path.stem}", fontsize=8)

    # ── Metrics panel ───────────────────────────────────────────────────────
    ax_m.axis("off")

    metric_keys = [
        ("NC", "no_at_fault_collisions"),
        ("DAC", "drivable_area_compliance"),
        ("DDC", "driving_direction_compliance"),
        ("EP", "ego_progress"),
        ("TTC", "time_to_collision_within_bound"),
        ("LK", "lane_keeping"),
        ("HC", "history_comfort"),
        ("PDM", "pdm_score"),
    ]
    metric_rows = [
        [metric_keys[0], metric_keys[1], metric_keys[2], metric_keys[7]],
        [metric_keys[3], metric_keys[4], metric_keys[5], metric_keys[6]],
    ]
    metric_precision = {
        "no_at_fault_collisions": 2,
        "drivable_area_compliance": 2,
        "driving_direction_compliance": 2,
        "lane_keeping": 2,
        "history_comfort": 2,
        "ego_progress": 3,
        "time_to_collision_within_bound": 3,
        "pdm_score": 3,
    }

    def _rv(row, key):
        return float(row.get(key, float("nan"))) if row is not None else float("nan")

    # Driving command header
    _cmd_map = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT", 3: "UNKNOWN"}
    _cmd_int = metric_cache.driving_command
    _cmd_str = _cmd_map.get(int(_cmd_int) if _cmd_int is not None else -1, "N/A")

    # Summary block (Stage1, Stage2-weighted, Combined)
    summary_lines = [f"CMD: {_cmd_str}", "── Summary ──────────────────────────────"]
    summary_lines.extend(_format_metric_block("S1  ", stage1_row, metric_rows, metric_precision))
    if combined_row is not None:
        s2_dict = {b: float(combined_row.get(f"stage2_{b}", float("nan"))) for _, b in metric_keys}
        cb_dict = {b: float(combined_row.get(f"combined_{b}", float("nan"))) for _, b in metric_keys}
        summary_lines.extend(_format_metric_block("S2w ", s2_dict, metric_rows, metric_precision))
        summary_lines.extend(_format_metric_block("Comb", cb_dict, metric_rows, metric_precision))

    # Per-offset table
    if stage2_data:
        summary_lines.append("")
        summary_lines.append("── Per-offset (sorted by weight) ─────────")

        def _col_width(key: str) -> int:
            return 4 if metric_precision.get(key, 3) == 2 else 5

        metric_hdr = " ".join(f"{label:>{_col_width(key)}}" for label, key in metric_keys)
        hdr = f"{'Offset':<14} {'w':>5}  {metric_hdr}"
        summary_lines.append(hdr)
        summary_lines.append("─" * len(hdr))
        for offset_name, od in offsets_sorted:
            m = od["metrics"]
            metric_cells = []
            for _, key in metric_keys:
                precision = metric_precision.get(key, 3)
                width = _col_width(key)
                metric_cells.append(f"{_rv(m, key):>{width}.{precision}f}")
            w = od["weight"]
            summary_lines.append(f"{offset_name:<14} {w:>5.3f}  {' '.join(metric_cells)}")

    ax_m.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax_m.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        fontfamily="monospace",
        wrap=False,
    )
    print(f"saved to : {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
