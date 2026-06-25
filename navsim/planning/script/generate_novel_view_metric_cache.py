#!/usr/bin/env python
"""
Novel View Metric Cache Generator for V2X-Real

For each (x, y) ego offset in ego coordinate frame:
1. Shift ego pose (ego2global_translation) by the rotated offset
2. Check collision between shifted ego footprint and GT bounding boxes
3. Check if shifted ego is within drivable area
4. If valid for ALL frames in a sub-scene, generate metric cache

Usage:
    python generate_novel_view_metric_cache.py \
        --v2xreal_pkl_path /path/to/spd_infos_temporal_test.pkl \
        --map_root /path/to/maps/expansion \
        --output_dir /path/to/output/metric_cache_novel
"""

import argparse
import copy
import json
import logging
import os
import pickle
import shutil
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import math
import yaml
import matplotlib.pyplot as plt

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import Polygon, Point
from tqdm import tqdm

from navsim.common.dataloader_v2xreal import (
    SceneLoaderV2XReal,
    SceneFilter,
    V2XRealMapWrapper,
    load_v2xreal_pkl,
    convert_v2xreal_to_navsim_format,
    compute_body_frame_velocity,
    compute_body_frame_velocity_from_positions,
    group_frames_by_scene,
    split_scene_into_subscenes,
    select_lane_id_for_ego_point,
)
from navsim.common.dataclasses import SensorConfig
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============== Constants ==============

# Ego vehicle dimensions (meters) - typical sedan
EGO_LENGTH = 4.5
EGO_WIDTH = 2.0

# Offsets in ego coordinate frame: x=forward, y=left
# X_OFFSETS = [-10, -5, 0, 5, 10]
X_OFFSETS = [-15, -10, -5, 0, 5, 10, 15]
# X_OFFSETS = [10]
Y_OFFSETS = [-2, -1, 0, 1, 2]

# V2X-Real raw PKL frame interval (seconds). Used for Hermite parameterization.
DATABASE_INTERVAL = 0.1  # 10Hz

# NOTE: Hermite interpolation is always used regardless of speed.
# Zero speed → zero tangent → linear interpolation from start to shifted current.


# driving command
CMD_STRAIGHT = 0
CMD_LEFT = 1
CMD_RIGHT = 2
CMD_UNKNOWN = 3

def compute_driving_command_from_gt_endpoint_y(
    s2_frames: list[dict],
    num_history: int,
    *,
    lookahead_idx: int = None,   # None means use the last frame as goal
    y_thresh_m: float = 2.0,            # body-frame y threshold (meters)
    require_forward_progress_m: float = 1.0,  # goal must be at least this far forward (x) to be valid
) -> int:
    """
    Transform the GT ego future trajectory into the current ego frame, then
    classify left/right/straight from the goal endpoint's body-frame y value.

    Body frame convention:
      x: forward, y: left  (nuPlan / EgoState convention)

    Returns CMD_*.
    """
    cur = num_history - 1
    if cur < 0 or cur >= len(s2_frames):
        return CMD_UNKNOWN

    # current pose (global)
    fr0 = s2_frames[cur]
    p0 = np.array(fr0["ego2global_translation"][:2], dtype=np.float64)
    yaw0 = Quaternion(*fr0["ego2global_rotation"]).yaw_pitch_roll[0]
    cos0, sin0 = math.cos(yaw0), math.sin(yaw0)

    # choose goal on ORIGINAL GT traj (global)
    if lookahead_idx is None:
        j = len(s2_frames) - 1
    else:
        j = min(max(0, lookahead_idx), len(s2_frames) - 1)

    pg = np.array(s2_frames[j]["ego2global_translation"][:2], dtype=np.float64)

    # global delta -> body delta (R(-yaw))
    dG = pg - p0
    dx_e =  dG[0] * cos0 + dG[1] * sin0
    dy_e = -dG[0] * sin0 + dG[1] * cos0

    # too little / behind -> unknown (optional safety)
    if dx_e < require_forward_progress_m and (dx_e * dx_e + dy_e * dy_e) < 1e-4:
        return CMD_STRAIGHT

    # classify by endpoint lateral offset
    if dy_e >= y_thresh_m:
        return CMD_LEFT
    elif dy_e <= -y_thresh_m:
        return CMD_RIGHT
    else:
        return CMD_STRAIGHT

def visualize_gt_endpoint_y_command(
    s2_frames: list[dict],
    num_history: int,
    *,
    lookahead_idx: int = None,   # None means the last frame
    y_thresh_m: float = 2.0,
    require_forward_progress_m: float = 1.0,
    cmd: int = None,             # pass in if already computed
    cmd_to_name: dict[int, str] = None,
    out_path: str = None,        # save path; None means plt.show()
    figsize: tuple[int, int] = (12, 5),
    title: str = None,
    show_global_view: bool = False,      # left: global traj
    show_ego_view: bool = True,         # right: ego(frame) traj
    ego_view_samples: int = None # None means cur..end (all); otherwise cur..cur+N
):
    """
    Visualize command decision based on GT endpoint y in ego/body frame.

    Plots:
      (Left) Global: original ego trajectory + current point + goal point
      (Right) Ego/body frame: trajectory relative to current, with y-threshold bands,
              and endpoint highlight.

    Body frame convention:
      x forward, y left.
    """
    assert len(s2_frames) > 0, "s2_frames empty"
    cur = num_history - 1
    if cur < 0 or cur >= len(s2_frames):
        raise ValueError(f"Invalid cur index cur={cur}, len={len(s2_frames)}")

    # ---- Extract trajectory (global) ----
    trajG = np.array([fr["ego2global_translation"][:2] for fr in s2_frames], dtype=np.float64)

    # current pose
    fr0 = s2_frames[cur]
    p0 = trajG[cur].copy()
    yaw0 = Quaternion(*fr0["ego2global_rotation"]).yaw_pitch_roll[0]
    cos0, sin0 = math.cos(yaw0), math.sin(yaw0)

    # choose goal index
    if lookahead_idx is None:
        j = len(s2_frames) - 1
    else:
        j = min(max(0, lookahead_idx), len(s2_frames) - 1)

    pg = trajG[j].copy()

    # ---- Build ego/body-frame trajectory from cur..end (or limited samples) ----
    end_idx = len(s2_frames) - 1
    if ego_view_samples is not None:
        end_idx = min(end_idx, cur + int(ego_view_samples))

    relG = trajG[cur : end_idx + 1] - p0[None, :]  # global delta to current

    # rotate global->ego: R(-yaw0)
    # [dx_e] = [ cos  sin] [dx_g]
    # [dy_e]   [-sin  cos] [dy_g]
    dx_e = relG[:, 0] * cos0 + relG[:, 1] * sin0
    dy_e = -relG[:, 0] * sin0 + relG[:, 1] * cos0
    trajE = np.stack([dx_e, dy_e], axis=1)

    # endpoint in ego frame (goal relative to current)
    dG = pg - p0
    goal_ex = dG[0] * cos0 + dG[1] * sin0
    goal_ey = -dG[0] * sin0 + dG[1] * cos0

    # compute cmd if not given (expects your compute fn in scope)
    if cmd is None:
        cmd = compute_driving_command_from_gt_endpoint_y(
            s2_frames=s2_frames,
            num_history=num_history,
            lookahead_idx=lookahead_idx,
            y_thresh_m=y_thresh_m,
            require_forward_progress_m=require_forward_progress_m,
        )

    if cmd_to_name is None:
        cmd_to_name = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT", 3: "UNKNOWN"}
    cmd_name = cmd_to_name.get(cmd, str(cmd))

    # ---- Make figure ----
    ncols = (1 if (not show_global_view and show_ego_view) or (show_global_view and not show_ego_view) else 2)
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    ax_idx = 0

    # (A) Global view
    if show_global_view:
        ax = axes[ax_idx]
        ax_idx += 1

        ax.plot(trajG[:, 0], trajG[:, 1], linewidth=2.0, marker="o", markersize=2.5, alpha=0.9, label="GT ego traj (global)")
        ax.scatter([p0[0]], [p0[1]], s=90, marker="s", alpha=0.95, label=f"Current (idx={cur})")
        ax.scatter([pg[0]], [pg[1]], s=120, marker="X", alpha=0.95, label=f"Goal (idx={j})")
        ax.plot([p0[0], pg[0]], [p0[1], pg[1]], linewidth=2.0, alpha=0.8, label="Current→Goal")

        # heading arrow at current
        fwd_len = 8.0
        ax.quiver(
            [p0[0]], [p0[1]],
            [fwd_len * cos0], [fwd_len * sin0],
            angles="xy", scale_units="xy", scale=1,
            width=0.006, headwidth=4.0, headlength=6.0,
            alpha=0.9,
            label="Heading @ current",
        )

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)
        ax.set_title("Global view")

    # (B) Ego/body view
    if show_ego_view:
        ax = axes[ax_idx]
        ax_idx += 1

        ax.plot(trajE[:, 0], trajE[:, 1], linewidth=2.0, marker="o", markersize=2.5, alpha=0.9, label="GT future traj in ego frame")
        ax.scatter([0.0], [0.0], s=90, marker="s", alpha=0.95, label="Current (origin)")
        ax.scatter([goal_ex], [goal_ey], s=140, marker="X", alpha=0.95, label=f"Goal in ego frame (idx={j})")

        # thresholds bands
        ax.axhline(+y_thresh_m, linewidth=2.0, alpha=0.7, label=f"+y_thresh={y_thresh_m:.1f}m")
        ax.axhline(-y_thresh_m, linewidth=2.0, alpha=0.7, label=f"-y_thresh={y_thresh_m:.1f}m")

        # show x=0 axis too
        ax.axvline(0.0, linewidth=1.5, alpha=0.4)

        # annotate endpoint y
        

        leg = ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.00, 1.02),   # (x, y) in axes coords, y>1 => above the axes
            borderaxespad=0.0,
            framealpha=0.9,
            ncol=1,                        # set to 2 if there are many entries
        )

        # 2) place the text above and to the "right" (x=1.0, ha=right so it does not overlap the legend)
        text = (
            f"cmd = {cmd_name}\n"
            f"goal_y = {goal_ey:.2f} m\n"
            f"goal_x = {goal_ex:.2f} m\n"
            f"y_thresh = ±{y_thresh_m:.2f} m\n"
            f"lookahead idx = {j}"
        )
        ax.text(
            1.00, 1.02, text,
            transform=ax.transAxes,
            va="bottom", ha="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.90, edgecolor="none"),
        )

        # reserve layout margin so the outside text is not clipped
        fig.subplots_adjust(right=0.80)

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        # ax.legend(loc="best", framealpha=0.9)
        # ax.legend(loc="lower left", bbox_to_anchor=(1.02, 0.60), framealpha=0.9)
        # fig.subplots_adjust(right=0.78)
        ax.set_xlabel("x (forward) [m]")
        ax.set_ylabel("y (left) [m]")
        # ax.set_title("Ego/body frame view")

        # optional view limits: include thresholds + endpoint
        xs = np.concatenate([trajE[:, 0], [goal_ex, 0.0]])
        ys = np.concatenate([trajE[:, 1], [goal_ey, 0.0, +y_thresh_m, -y_thresh_m]])
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        pad = max(5.0, 0.15 * max(xmax - xmin, ymax - ymin))
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

    if title is None:
        pass
        # title = "Driving Command via GT endpoint y (ego frame)"
    fig.suptitle(title)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        out_path = f"exp_debug/driving_cmd_stage2/{fr0['token']}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

def load_scene_tokens_from_file(path: Path) -> List[str]:
    """Load scene tokens from a text file (one token per line)."""
    if not path.exists():
        raise FileNotFoundError(f"scene_tokens_file not found: {path}")

    tokens: List[str] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens.append(line)

    if not tokens:
        raise ValueError(f"scene_tokens_file is empty after parsing: {path}")
    return tokens


def get_v2xreal_ego_parameters() -> VehicleParameters:
    """
    Vehicle parameters for V2X-Real ego vehicle.

    V2X-Real's ego2global_translation represents the vehicle center (not the rear axle),
    so we set front_length = rear_length = EGO_LENGTH/2 to make rear_axle_to_center = 0.
      rear_axle_to_center = half_length - rear_length
                          = (front_length + rear_length)/2 - rear_length
                          = (EGO_LENGTH/2 + EGO_LENGTH/2)/2 - EGO_LENGTH/2 = 0
    """
    half_len = EGO_LENGTH / 2.0  # 2.25 m
    return VehicleParameters(
        vehicle_name="v2xreal_ego",
        vehicle_type="v2xreal",
        width=EGO_WIDTH,
        front_length=half_len,
        rear_length=half_len,
        wheel_base=2.0,
        cog_position_from_rear_axle=1.0,
        height=1.8,
    )


# ============== Utility Functions ==============


def get_ego_footprint_polygon(
    ego_x: float, ego_y: float, ego_heading: float,
    length: float = EGO_LENGTH, width: float = EGO_WIDTH,
) -> Polygon:
    """
    Create an oriented bounding box polygon for the ego vehicle.
    :param ego_x: ego x in global frame
    :param ego_y: ego y in global frame
    :param ego_heading: ego heading in global frame (radians)
    :param length: vehicle length
    :param width: vehicle width
    :return: Shapely Polygon of ego footprint
    """
    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)

    # Half dimensions
    hl = length / 2.0
    hw = width / 2.0

    # 4 corners relative to center, then rotate + translate
    corners_local = np.array([
        [ hl,  hw],
        [ hl, -hw],
        [-hl, -hw],
        [-hl,  hw],
    ])

    corners_global = np.zeros_like(corners_local)
    for i, (lx, ly) in enumerate(corners_local):
        corners_global[i, 0] = ego_x + lx * cos_h - ly * sin_h
        corners_global[i, 1] = ego_y + lx * sin_h + ly * cos_h

    return Polygon(corners_global)


def get_box_polygon(
    x: float, y: float, heading: float, length: float, width: float,
) -> Polygon:
    """
    Create an oriented bounding box polygon for a GT object.
    :param x: center x in global frame
    :param y: center y in global frame
    :param heading: heading in global frame (radians)
    :param length: box length
    :param width: box width
    :return: Shapely Polygon
    """
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    hl = length / 2.0
    hw = width / 2.0

    corners_local = np.array([
        [ hl,  hw],
        [ hl, -hw],
        [-hl, -hw],
        [-hl,  hw],
    ])

    corners_global = np.zeros_like(corners_local)
    for i, (lx, ly) in enumerate(corners_local):
        corners_global[i, 0] = x + lx * cos_h - ly * sin_h
        corners_global[i, 1] = y + lx * sin_h + ly * cos_h

    return Polygon(corners_global)


def apply_ego_offset(
    ego_translation: np.ndarray,
    ego_heading: float,
    dx_ego: float,
    dy_ego: float,
) -> np.ndarray:
    """
    Apply offset in ego coordinate frame to global translation.
    :param ego_translation: original [x, y, z] in global frame
    :param ego_heading: ego heading in global frame (radians)
    :param dx_ego: forward offset in ego frame (meters)
    :param dy_ego: leftward offset in ego frame (meters)
    :return: shifted [x, y, z] in global frame
    """
    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)

    shifted = ego_translation.copy()
    shifted[0] += dx_ego * cos_h - dy_ego * sin_h
    shifted[1] += dx_ego * sin_h + dy_ego * cos_h
    return shifted


def cubic_hermite_interpolate_past_poses(
    original_frames: List[Dict[str, Any]],
    dx_ego: float,
    dy_ego: float,
    num_history_frames: int = 4,
    stage1_start_frame: Optional[Dict[str, Any]] = None,
    frame_dt: float = DATABASE_INTERVAL,
    lookback_seconds: float = 4.0,
) -> List[np.ndarray]:
    """
    Interpolate past ego positions using Cubic Hermite spline.

    Start point (P0): stage1 start position (4s before current, original pose).
                      If not available, falls back to frame 0 (1.5s ago).
    End point   (P1): current frame (shifted pose, original heading).

    The Hermite curve spans the full time window (P0 → P1).
    Frames 0..num_history-1 are sampled at their corresponding t-parameters.

    With stage1_start (4s ago, dt_total=4.0s):
      frame 0 (t=-1.5s): t_param = 2.5 / 4.0 = 0.625
      frame 1 (t=-1.0s): t_param = 3.0 / 4.0 = 0.75
      frame 2 (t=-0.5s): t_param = 3.5 / 4.0 = 0.875
      frame 3 (t= 0.0s): t_param = 4.0 / 4.0 = 1.0

    When speed is zero, tangent magnitude falls back to chord speed
    (|P1-P0|/dt_total), so the vehicle heading always shapes the curve.

    :param original_frames: list of original frame dicts (sub-scene)
    :param dx_ego: forward offset in ego frame
    :param dy_ego: leftward offset in ego frame
    :param num_history_frames: number of history frames
    :param stage1_start_frame: frame dict at stage1 start (4s before current), or None
    :return: list of interpolated [x, y, z] translations for frames 0..(num_history-1).
    """
    current_idx = num_history_frames - 1  # frame 3

    # --- Start point (P0): stage1 start or frame 0 fallback ---
    if stage1_start_frame is not None:
        start_frame = stage1_start_frame
        # stage1_start is lookback_seconds before current
        dt_total = lookback_seconds
        # Time from stage1_start to subscene frame 0
        dt_start_to_frame0 = dt_total - current_idx * frame_dt
    else:
        start_frame = original_frames[0]
        dt_total = current_idx * frame_dt  # time span within subscene only
        dt_start_to_frame0 = 0.0

    trans_start = np.array(start_frame["ego2global_translation"], dtype=np.float64)
    quat_start = Quaternion(*start_frame["ego2global_rotation"])
    heading_start = quat_start.yaw_pitch_roll[0]

    # --- End point (P1): shifted current frame ---
    trans_curr = np.array(original_frames[current_idx]["ego2global_translation"], dtype=np.float64)
    quat_curr = Quaternion(*original_frames[current_idx]["ego2global_rotation"])
    heading_curr = quat_curr.yaw_pitch_roll[0]

    shifted_curr = apply_ego_offset(trans_curr, heading_curr, dx_ego, dy_ego)

    P0 = trans_start[:2]
    P1 = shifted_curr[:2]

    # --- Speeds for tangent magnitude ---
    # Use max(actual_speed, chord_speed) so that heading always influences
    # the curve shape, even for stationary vehicles (speed ≈ 0).
    chord_length = np.linalg.norm(P1 - P0)
    chord_speed = chord_length / dt_total if dt_total > 0 else 0.0

    dyn_start = start_frame["ego_dynamic_state"]
    speed_start = max(np.sqrt(dyn_start[0] ** 2 + dyn_start[1] ** 2), chord_speed)

    dyn_curr = original_frames[current_idx]["ego_dynamic_state"]
    speed_curr = max(np.sqrt(dyn_curr[0] ** 2 + dyn_curr[1] ** 2), chord_speed)

    # --- Tangent vectors (position derivative scaled by dt_total) ---
    # M = speed * dt_total * (cos(heading), sin(heading))
    # Heading always determines direction; magnitude >= chord_speed * dt_total.
    M0 = np.array([
        speed_start * dt_total * np.cos(heading_start),
        speed_start * dt_total * np.sin(heading_start),
    ])
    M1 = np.array([
        speed_curr * dt_total * np.cos(heading_curr),
        speed_curr * dt_total * np.sin(heading_curr),
    ])

    # --- Hermite basis functions ---
    # P(t) = h00*P0 + h10*M0 + h01*P1 + h11*M1, t in [0,1]
    # t=0 → P0 (stage1 start), t=1 → P1 (shifted current)
    interpolated_translations = []
    for i in range(num_history_frames):
        # Time from stage1 start to frame i
        time_from_start = dt_start_to_frame0 + i * frame_dt
        t = time_from_start / dt_total
        t2 = t * t
        t3 = t2 * t

        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        xy = h00 * P0 + h10 * M0 + h01 * P1 + h11 * M1

        # Keep original z
        z = original_frames[i]["ego2global_translation"][2]
        new_trans = np.array([xy[0], xy[1], z], dtype=np.float64)
        interpolated_translations.append(new_trans)

    return interpolated_translations


def check_collision_with_gt_boxes(
    ego_polygon: Polygon,
    gt_boxes_9d: np.ndarray,
) -> bool:
    """
    Check if ego footprint collides with any GT bounding box.
    :param ego_polygon: Shapely Polygon of ego footprint in global frame
    :param gt_boxes_9d: (N, 9) array [x, y, z, l, w, h, vx, vy, heading] in global frame
    :return: True if collision exists
    """
    if len(gt_boxes_9d) == 0:
        return False

    for box in gt_boxes_9d:
        box_polygon = get_box_polygon(
            x=box[0], y=box[1], heading=box[8],
            length=box[3], width=box[4],
        )
        if ego_polygon.intersects(box_polygon):
            return True
    return False


def check_in_route_or_intersection(
    map_wrapper: V2XRealMapWrapper,
    ego_x: float,
    ego_y: float,
    route_lane_ids: List[str],
) -> bool:
    """
    Check if ego position is within route lane polygons or intersection (junction) polygons.
    Unlike check_in_drivable_area which accepts ANY lane, this restricts to the
    ego's planned route lanes + intersections only.

    :param map_wrapper: V2X-Real map wrapper
    :param ego_x: ego x in global frame
    :param ego_y: ego y in global frame
    :param route_lane_ids: list of lane IDs on the ego's route
    :return: True if in route lane or intersection
    """
    point = Point(ego_x, ego_y)

    # Check route lane polygons
    lane_id_to_obj = map_wrapper._lane_id_to_obj()
    for lid in route_lane_ids:
        lane_obj = lane_id_to_obj.get(lid)
        if lane_obj is not None and lane_obj.polygon is not None:
            if lane_obj.polygon.contains(point):
                return True

    # Check junction (intersection) polygons
    for junction_poly in map_wrapper._get_junctions():
        if junction_poly.contains(point):
            return True

    return False


def check_in_drivable_area(
    map_wrapper: V2XRealMapWrapper,
    ego_x: float,
    ego_y: float,
) -> bool:
    """
    Check if ego position is within drivable area.
    Uses is_in_layer which checks lanes + crosswalks + junctions.
    :param map_wrapper: V2X-Real map wrapper
    :param ego_x: ego x in global frame
    :param ego_y: ego y in global frame
    :return: True if in drivable area
    """
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    return map_wrapper.is_in_layer(ego_x, ego_y, SemanticMapLayer.ROADBLOCK)


def validate_offset_for_subscene(
    subscene_frames: List[Dict[str, Any]],
    dx_ego: float,
    dy_ego: float,
    map_wrapper: V2XRealMapWrapper,
    num_history_frames: int = 4,
    stage1_start_frame: Optional[Dict[str, Any]] = None,
    s1_frames: Optional[List[Dict[str, Any]]] = None,
    frame_time_s: float = 0.5,
    collision_window_s: float = 0.5,
    route_lane_ids: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Validate an offset for a sub-scene.
    Checks:
      1. Shifted position is in route lanes or intersection (if route_lane_ids provided),
         otherwise falls back to drivable area check.
      2. No collision along Hermite trajectory from Stage1 current (t=0) to Stage2 current+offset (t=+4s),
         checking GT boxes within ±collision_window_s of each sample time.
      3. Shifted position is not behind the stage1 start position.

    :param subscene_frames: list of converted NavSim frame dicts for Stage2 sub-scene
    :param dx_ego: forward offset in ego frame
    :param dy_ego: leftward offset in ego frame
    :param map_wrapper: V2X-Real map wrapper
    :param num_history_frames: number of history frames (current frame is at this index - 1)
    :param stage1_start_frame: Stage1 current frame dict (t=0), used as Hermite start
    :param s1_frames: Stage1 full subscene frames (covers t=-1.5s to t=+4s);
                      used for Hermite collision sampling
    :param frame_time_s: time between consecutive s1_frames (seconds)
    :param collision_window_s: half-width of GT box time window for collision check (seconds)
    :param route_lane_ids: list of route lane IDs; if provided, restricts area check to
                           route lanes + intersections instead of all drivable area
    :return: (valid, reason)
    """
    # (0, 0) offset is always valid (original ego pose)
    if dx_ego == 0 and dy_ego == 0:
        return True, "valid"

    # Stage2 current frame (decision point)
    current_frame = subscene_frames[num_history_frames - 1]

    ego_trans = np.array(current_frame["ego2global_translation"])
    ego_quat = Quaternion(*current_frame["ego2global_rotation"])
    ego_heading = ego_quat.yaw_pitch_roll[0]

    shifted_trans = apply_ego_offset(ego_trans, ego_heading, dx_ego, dy_ego)
    shifted_x, shifted_y = shifted_trans[0], shifted_trans[1]

    # --- Check 1: route lane / intersection check at Stage2 current ---
    if route_lane_ids is not None and len(route_lane_ids) > 0:
        if not check_in_route_or_intersection(map_wrapper, shifted_x, shifted_y, route_lane_ids):
            return False, f"frame {current_frame['frame_idx']}: outside route lanes / intersection"
    else:
        if not check_in_drivable_area(map_wrapper, shifted_x, shifted_y):
            return False, f"frame {current_frame['frame_idx']}: outside drivable area"

    # --- Check 2: Hermite trajectory collision (Stage1 current t=0 → Stage2 current+offset t=+4s) ---
    if s1_frames is not None and stage1_start_frame is not None:
    # if False:
        # Hermite endpoints
        P0 = np.array(stage1_start_frame["ego2global_translation"][:2], dtype=np.float64)
        P1 = np.array([shifted_x, shifted_y], dtype=np.float64)

        quat_start = Quaternion(*stage1_start_frame["ego2global_rotation"])
        heading_start = quat_start.yaw_pitch_roll[0]

        # Total time span of the Hermite curve
        dt_total = (len(s1_frames) - 1 - (num_history_frames - 1)) * frame_time_s
        if dt_total <= 0.0:
            dt_total = 1.0  # safety fallback

        chord_length = np.linalg.norm(P1 - P0)
        chord_speed = chord_length / dt_total

        dyn_start = stage1_start_frame["ego_dynamic_state"]
        speed_start = max(np.sqrt(dyn_start[0] ** 2 + dyn_start[1] ** 2), chord_speed)

        dyn_curr = current_frame["ego_dynamic_state"]
        speed_curr = max(np.sqrt(dyn_curr[0] ** 2 + dyn_curr[1] ** 2), chord_speed)

        M0 = np.array([
            speed_start * dt_total * np.cos(heading_start),
            speed_start * dt_total * np.sin(heading_start),
        ])
        M1 = np.array([
            speed_curr * dt_total * np.cos(ego_heading),
            speed_curr * dt_total * np.sin(ego_heading),
        ])

        # Number of adjacent frames within ±collision_window_s
        lookback_frames = max(1, int(round(collision_window_s / frame_time_s)))

        # Iterate from Stage1 current (index num_history_frames-1) to last s1_frame (Stage2 current)
        s1_start_idx = num_history_frames - 1
        for i in range(s1_start_idx, len(s1_frames)):
            t_param = min((i - s1_start_idx) * frame_time_s / dt_total, 1.0)
            t2, t3 = t_param ** 2, t_param ** 3
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t_param
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            pos_xy = h00 * P0 + h10 * M0 + h01 * P1 + h11 * M1

            frame_i = s1_frames[i]
            quat_i = Quaternion(*frame_i["ego2global_rotation"])
            heading_i = quat_i.yaw_pitch_roll[0]

            # Collect GT boxes from frames within ±collision_window_s
            lo = max(0, i - lookback_frames)
            hi = min(len(s1_frames) - 1, i + lookback_frames)
            gt_list = [
                s1_frames[j]["anns"]["gt_boxes"]
                for j in range(lo, hi + 1)
                if len(s1_frames[j]["anns"]["gt_boxes"]) > 0
            ]
            if gt_list:
                combined_gt = np.concatenate(gt_list, axis=0)
                polygon = get_ego_footprint_polygon(pos_xy[0], pos_xy[1], heading_i)
                if check_collision_with_gt_boxes(polygon, combined_gt):
                    return False, f"frame {frame_i['frame_idx']}: collision along Hermite path"
    else:
        # Fallback: simple collision check at Stage2 start only
        gt_boxes = current_frame["anns"]["gt_boxes"]
        if len(gt_boxes) > 0:
            polygon = get_ego_footprint_polygon(shifted_x, shifted_y, ego_heading)
            if check_collision_with_gt_boxes(polygon, gt_boxes):
                return False, f"frame {current_frame['frame_idx']}: collision at stage2 start"

    # --- Check 3: shifted position must not be behind stage1 start position ---
    ref_frame = stage1_start_frame
    trans_ref = np.array(ref_frame["ego2global_translation"])
    quat_ref = Quaternion(*ref_frame["ego2global_rotation"])
    heading_ref = quat_ref.yaw_pitch_roll[0]

    forward_dir = np.array([np.cos(heading_ref), np.sin(heading_ref)])
    diff = np.array([shifted_x - trans_ref[0], shifted_y - trans_ref[1]])
    forward_proj = np.dot(diff, forward_dir)

    if forward_proj < 2.0:
        return False, f"frame {current_frame['frame_idx']}: shifted position behind stage1 start (forward_proj={forward_proj:.2f}m < 2.0m)"

    return True, "valid"


def offset_to_dirname(dx: int, dy: int) -> str:
    """Convert offset values to directory name matching novel view naming convention."""
    x_str = f"x+{dx}" if dx >= 0 else f"x{dx}"
    y_str = f"y+{dy}" if dy >= 0 else f"y{dy}"
    return f"{x_str}_{y_str}"


def create_shifted_pkl_frames(
    original_frames: List[Dict[str, Any]],
    dx_ego: float,
    dy_ego: float,
    num_history_frames: int = 4,
    stage1_start_frame: Optional[Dict[str, Any]] = None,
    frame_dt: float = DATABASE_INTERVAL,
    lookback_seconds: float = 4.0,
    cmd = None,
) -> List[Dict[str, Any]]:
    """
    Create a deep copy of frames with shifted ego pose.

    Past frames (0 ~ num_history-2):
      - Cubic Hermite interpolation between stage1 start (4s ago, original)
        and current frame (shifted), using heading-aware tangents.
      - Zero speed → zero tangent → linear interpolation (no fallback).
      - Velocity recalculated from interpolated positions.

    Current frame (num_history-1) and future frames:
      - Full offset applied (same as before).

    Heading (ego2global_rotation) is kept original for all frames.
    GT boxes remain in their original global positions.

    :param original_frames: list of original frame dicts (already in NavSim format)
    :param dx_ego: forward offset in ego frame
    :param dy_ego: leftward offset in ego frame
    :param num_history_frames: number of history frames (current frame index = this - 1)
    :param stage1_start_frame: frame dict at stage1 start (4s before current), or None
    :return: list of shifted frame dicts
    """
    current_frame_idx = num_history_frames - 1

    # --- Cubic Hermite interpolation for past frames (always applied) ---
    hermite_translations = cubic_hermite_interpolate_past_poses(
        original_frames, dx_ego, dy_ego, num_history_frames,
        stage1_start_frame=stage1_start_frame,
        frame_dt=frame_dt,
        lookback_seconds=lookback_seconds,
    )

    shifted_frames = []
    for i, frame in enumerate(original_frames):
        shifted_frame = copy.deepcopy(frame)

        if i < current_frame_idx:
            # Past frame: use Hermite-interpolated position
            shifted_frame["ego2global_translation"] = hermite_translations[i]
        else:
            # Current + future frames: full offset
            ego_trans = np.array(frame["ego2global_translation"])
            ego_quat = Quaternion(*frame["ego2global_rotation"])
            ego_heading = ego_quat.yaw_pitch_roll[0]
            shifted_frame["ego2global_translation"] = apply_ego_offset(
                ego_trans, ego_heading, dx_ego, dy_ego,
            )
        if cmd is not None:
            shifted_frame['driving_command'] = cmd

        shifted_frames.append(shifted_frame)

    # --- Recalculate velocity for all frames (past + current + future) from shifted positions ---
    compute_body_frame_velocity_from_positions(shifted_frames, start_idx=0)

    return shifted_frames


def _compute_route_lane_ids(
    map_wrapper: V2XRealMapWrapper,
    s1_frames: List[Dict[str, Any]],
    num_history: int,
    output_dir: str,
) -> Optional[List[str]]:
    """
    Compute route_lane_ids from connector cache (Approach 1) or ego-position fallback (Approach 2).
    This is a lightweight computation that does NOT require running PDMClosedPlanner.

    :param map_wrapper: V2X-Real map wrapper (freshly created, no route info yet)
    :param s1_frames: Stage1 subscene frame dicts
    :param num_history: number of history frames
    :param output_dir: output directory (connector cache is stored under output_dir/lane_connectors/)
    :return: list of route lane IDs, or None if computation fails
    """
    current_frame = s1_frames[num_history - 1]
    scene_token = current_frame.get("scene_token")

    # Approach 1: Try connector cache
    connectors = []
    if scene_token is not None:
        connector_path = Path(output_dir) / "lane_connectors" / f"{scene_token}.pkl"
        if connector_path.exists():
            try:
                with open(connector_path, "rb") as f:
                    connectors = pickle.load(f)
            except Exception:
                connectors = []

    if connectors:
        connector = connectors[0]
        incoming_lane_id = connector.incoming_lane_ids[0]
        outgoing_lane_id = connector.outgoing_lane_ids[0]

        route_lane_ids = [connector.id]
        for rb_id, lane_id_list in map_wrapper.road_block_ids_dict.items():
            if incoming_lane_id in lane_id_list:
                route_lane_ids.extend(lane_id_list)
            if outgoing_lane_id in lane_id_list:
                route_lane_ids.extend(lane_id_list)
        return route_lane_ids

    # Approach 2: Fallback — find lane by ego position
    ego_trans = current_frame["ego2global_translation"]
    ego_xy = np.array([ego_trans[0], ego_trans[1]])
    lane_id = select_lane_id_for_ego_point(map_wrapper, ego_xy)
    if lane_id is None:
        return None

    route_lane_ids = []
    for rb_id, lane_id_list in map_wrapper.road_block_ids_dict.items():
        if lane_id in lane_id_list:
            route_lane_ids.extend(lane_id_list)
    return route_lane_ids if route_lane_ids else None


def process_single_item(item, output_dir, map_root, num_history, num_future, force,
                        ps_num_poses=40, ps_interval=0.1):
    """Process a single (offset, subscene) pair. Returns (dx, dy, status_str).

    Stage2 metric cache is built from s2_frames (t+4s subscene) with offset,
    using s2_hermite_start (Stage1 current, t=0) as the Hermite start point.
    The cache file is named after the Stage1 token so it can be matched during evaluation.

    Top-level function so multiprocessing can pickle it.
    """
    dx, dy, token, s1_frames, _stage1_start_frame, s2_frames, s2_hermite_start = item
    offset_name = offset_to_dirname(dx, dy)
    cache_dir = Path(output_dir) / offset_name

    # Cache file is named after Stage1 token for Stage1↔Stage2 matching
    log_name = s1_frames[num_history - 1]["log_name"]
    frame_token = s1_frames[num_history - 1]["token"]
    cache_file = cache_dir / log_name / "unknown" / frame_token / "metric_cache.pkl"

    # Derived timing
    time_horizon = ps_num_poses * ps_interval  # e.g. 8 * 0.5 = 4.0s
    # Time between consecutive s1_frames (Stage1 future frames span time_horizon over num_future steps)
    frame_time_s = time_horizon / num_future if num_future > 0 else 0.5

    # Validate offset at Stage2 current frame (t+4s),
    # using Stage1 current (t=0, = s2_hermite_start) as the "not behind" reference.
    map_root_path = Path(map_root)
    map_wrapper_local = V2XRealMapWrapper(map_root_path)

    # Compute route_lane_ids from connector cache or ego-position fallback
    route_lane_ids = _compute_route_lane_ids(
        map_wrapper_local, s1_frames, num_history, output_dir,
    )

    is_valid, reason = validate_offset_for_subscene(
        s2_frames, dx, dy, map_wrapper_local,
        num_history_frames=num_history,
        stage1_start_frame=s2_hermite_start,
        s1_frames=s1_frames,
        frame_time_s=frame_time_s,
        collision_window_s=0.5,
        route_lane_ids=route_lane_ids,
    )
    if not is_valid:
        if cache_file.exists():
            cache_file.unlink()
        if "collision" in reason:
            return (dx, dy, "collision")
        elif "behind" in reason:
            return (dx, dy, "behind")
        else:
            return (dx, dy, "off_road")

    if cache_file.exists() and not force:
        return (dx, dy, "skipped")

    try:
        # Build shifted Stage2 frames:
        # 1) compute command from ORIGINAL s2_frames, but using shifted start pose
        cmd = compute_driving_command_from_gt_endpoint_y(
            s2_frames=s2_frames,
            num_history=num_history,
            lookahead_idx=None,
            y_thresh_m=2.0,
        )
        # visualize_gt_endpoint_y_command(
        #     s2_frames=s2_frames,
        #     num_history=num_history,
        #     lookahead_idx=None,
        #     y_thresh_m=2.0,
        #     cmd=cmd,
        #     out_path=None,  # display on screen
        # )
     

        #   Hermite curve: s2_hermite_start (t=0) → s2_current+offset (t=+4s)
        shifted_s2_frames = create_shifted_pkl_frames(
            s2_frames, dx, dy,
            num_history_frames=num_history,
            stage1_start_frame=s2_hermite_start,
            frame_dt=DATABASE_INTERVAL,
            lookback_seconds=time_horizon,
            cmd = cmd,
        )
        # Force cache path token to Stage1 current token so Stage1↔Stage2 can be matched by token.
        shifted_s2_frames[num_history - 1]["token"] = frame_token
        
        scene = SceneLoaderV2XReal._create_scene_v2xreal(
            scene_dict_list=shifted_s2_frames,
            sensor_blobs_path=None,
            num_history_frames=num_history,
            num_future_frames=num_future,
            sensor_config=SensorConfig.build_no_sensors(),
            map_root=map_root_path,
            connector_cache_dir=Path(output_dir),
        )

        scenario = NavSimScenario(
            scene,
            map_root=str(map_root_path),
            map_version='nuplan-maps-v1.0',
            ego_vehicle_parameters=get_v2xreal_ego_parameters(),
        )

        proposal_sampling_local = TrajectorySampling(num_poses=ps_num_poses, interval_length=ps_interval)
        if os.getenv('PDMS_V2', 'true').lower() == 'true':
            use_pdms_v1 = False
            # print("Use pdms v2")
        else:
            use_pdms_v1 = True
            print("Use pdms v1")

        
        processor_local = MetricCacheProcessor(
            cache_path=str(cache_dir),
            force_feature_computation=force,
            proposal_sampling=proposal_sampling_local,
            use_pdms_v1=use_pdms_v1,
        )
    
        metric_cache = processor_local.compute_metric_cache(scenario)
        metric_cache.dump()
        return (dx, dy, "valid")

    except Exception as e:
        logger.warning(f"Error processing offset ({dx:+d},{dy:+d}) token={frame_token}: {e}")
        return (dx, dy, f"error:{e}")


# ============== Main ==============


def main():
    parser = argparse.ArgumentParser(description="Generate Novel View Metric Caches for V2X-Real")
    parser.add_argument(
        "--v2xreal_pkl_path",
        type=str,
        required=True,
        help="Path to V2X-Real pkl file",
    )
    parser.add_argument(
        "--map_root",
        type=str,
        required=True,
        help="Path to V2X-Real map files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for novel view metric caches",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of existing caches",
    )
    parser.add_argument(
        "--skip_original",
        action="store_true",
        help="Skip (0,0) offset (original ego pose)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: min(cpu_count, 16)). Set to 1 for sequential/debug mode.",
    )
    parser.add_argument(
        "--scene_tokens_file",
        type=str,
        default=None,
        help="Optional path to scene-token text file (one scene token per line).",
    )
    _script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--scene_filter_yaml",
        type=str,
        default=str(_script_dir / "config" / "common" / "train_test_split" / "scene_filter" / "v2xreal_scenes_short_5s.yaml"),
        help="Path to scene filter yaml config (defaults to the 5s / 2Hz setup).",
    )
    parser.add_argument(
        "--proposal_yaml",
        type=str,
        default=str(_script_dir / "config" / "metric_caching" / "default_metric_caching_v2xreal_5s.yaml"),
        help="Path to metric caching yaml config for proposal_sampling (defaults to the 5s / 2Hz setup).",
    )
    args = parser.parse_args()

    os.environ['NUPLAN_MAPS_ROOT'] = args.map_root

    pkl_path = Path(args.v2xreal_pkl_path)
    map_root = Path(args.map_root)
    output_dir = Path(args.output_dir)

    logger.info("=" * 70)
    logger.info("Novel View Metric Cache Generator for V2X-Real")
    logger.info("=" * 70)
    logger.info(f"PKL: {pkl_path}")
    logger.info(f"Map: {map_root}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"X offsets: {X_OFFSETS}")
    logger.info(f"Y offsets: {Y_OFFSETS}")
    logger.info(f"Total offset combos: {len(X_OFFSETS) * len(Y_OFFSETS)}")
    logger.info("=" * 70)

    # Load map
    logger.info("Loading V2X-Real map...")
    map_wrapper = V2XRealMapWrapper(map_root)
    logger.info("Map loaded.")

    # Load and convert frames
    logger.info("Loading V2X-Real pkl...")
    raw_frames = load_v2xreal_pkl(pkl_path)
    logger.info(f"Loaded {len(raw_frames)} raw frames.")

    logger.info("Converting frames to NavSim format...")
    all_frames = [convert_v2xreal_to_navsim_format(f) for f in tqdm(raw_frames, desc="Converting")]

    # Calculate velocity from position differences (global → body frame)
    logger.info("Calculating velocities from position differences...")
    compute_body_frame_velocity(all_frames)

    # Group by scene
    scenes_dict = group_frames_by_scene(all_frames)
    scenes_dict_before_filter = scenes_dict
    selected_scene_tokens: Optional[Set[str]] = None

    logger.info(f"Grouped into {len(scenes_dict)} scenes")

    if args.scene_tokens_file:
        scene_tokens_path = Path(args.scene_tokens_file)
        selected_scene_tokens = set(load_scene_tokens_from_file(scene_tokens_path))
        before_scene_count = len(scenes_dict)
        scenes_dict = {k: v for k, v in scenes_dict.items() if k in selected_scene_tokens}
        after_scene_count = len(scenes_dict)
        logger.info(
            f"Applied scene-token filter ({scene_tokens_path}): "
            f"requested={len(selected_scene_tokens)}, before={before_scene_count}, after={after_scene_count}"
        )
        missing = sorted(selected_scene_tokens - set(scenes_dict.keys()))
        if missing:
            logger.warning(f"Scene-token filter unmatched ({len(missing)}): {missing}")
        if not scenes_dict:
            raise ValueError("scene_tokens_file filter removed all scenes; nothing to process.")

    # Load SceneFilter from stage1 yaml config (single source of truth)
    scene_filter_yaml = Path(args.scene_filter_yaml)
    proposal_yaml = Path(args.proposal_yaml)

    logger.info(f"Loading SceneFilter config from: {scene_filter_yaml}")
    with open(scene_filter_yaml) as f:
        sf_cfg = yaml.safe_load(f)

    scene_filter = SceneFilter(
        num_history_frames=sf_cfg["num_history_frames"],
        num_future_frames=sf_cfg["num_future_frames"],
        frame_interval=sf_cfg.get("frame_interval", 1),
    )
    num_frames = scene_filter.num_frames
    num_history = scene_filter.num_history_frames
    num_future = scene_filter.num_future_frames
    frame_interval = scene_filter.frame_interval

    logger.info(f"Loading proposal_sampling from: {proposal_yaml}")
    with open(proposal_yaml) as f:
        ps_cfg = yaml.safe_load(f)
    ps_num_poses = ps_cfg["proposal_sampling"]["num_poses"]
    ps_interval = ps_cfg["proposal_sampling"]["interval_length"]
    logger.info(f"proposal_sampling: num_poses={ps_num_poses}, interval_length={ps_interval}")

    logger.info(f"Sub-scene config: {num_history} history + {num_future} future = {num_frames} total, interval={frame_interval}")

    if selected_scene_tokens is not None:
        subscene_before = sum(
            len(split_scene_into_subscenes(frame_list, num_frames, frame_interval))
            for frame_list in scenes_dict_before_filter.values()
        )
        subscene_after = sum(
            len(split_scene_into_subscenes(frame_list, num_frames, frame_interval))
            for frame_list in scenes_dict.values()
        )
        logger.info(
            f"Scene-token filter sub-scenes: before={subscene_before}, after={subscene_after}"
        )

    # Build all sub-scenes using shared utility
    # Stage1 start frame: time_horizon / DATABASE_INTERVAL frames before current
    time_horizon = ps_num_poses * ps_interval
    STAGE1_LOOKBACK_FRAMES = int(round(time_horizon / DATABASE_INTERVAL))
    logger.info(f"STAGE1_LOOKBACK_FRAMES={STAGE1_LOOKBACK_FRAMES} ({time_horizon}s / {DATABASE_INTERVAL}s)")
    logger.info(f"Stage2 offset in frame_list = num_future * frame_interval = {num_future} * {frame_interval} = {num_future * frame_interval} frames")

    # each entry: (s1_frames, _unused_stage1_start, s2_frames, s2_hermite_start)
    # s2_frames   : Stage2 subscene (t+4s to t+8s), same window size as Stage1
    # s2_hermite_start: Stage1 current frame (t=0) used as Hermite start for Stage2
    all_subscenes: Dict[str, Tuple[
        List[Dict[str, Any]],   # s1_frames
        Optional[Dict[str, Any]],  # stage1_start_frame (kept for reference, unused in cache gen)
        List[Dict[str, Any]],   # s2_frames
        Dict[str, Any],         # s2_hermite_start
    ]] = {}
    for scene_token, frame_list in scenes_dict.items():
        if False:
            if scene_token != '2023-04-04-14-27-53_44_0_folder_1_-1': continue
        for sub in split_scene_into_subscenes(frame_list, num_frames, frame_interval):
            token = sub[num_history - 1]["token"]
            unique_key = f"{scene_token}_{token}"

            start = frame_list.index(sub[0])

            # Stage2 subscene: same num_frames window, starting num_future frames later
            # Stage2 current = Stage1's last future frame (t=+4s relative to Stage1 current)
            s2_start_idx = start + num_future * frame_interval
            s2_end_idx = s2_start_idx + (num_frames - 1) * frame_interval
            if s2_end_idx >= len(frame_list):
                continue  # not enough future frames for Stage2
            s2_frames = [frame_list[i] for i in range(s2_start_idx, s2_end_idx + 1, frame_interval)]

            # Stage2 Hermite start = Stage1 current frame (t=0)
            s2_hermite_start = sub[num_history - 1]

            all_subscenes[unique_key] = (sub, None, s2_frames, s2_hermite_start)

    logger.info(f"Total sub-scenes: {len(all_subscenes)}")
    if args.scene_tokens_file:
        logger.info("Sub-scenes were generated after scene-token filtering.")

    # ===== Parallel processing =====
    num_workers = args.num_workers if args.num_workers is not None else min(cpu_count(), 16)
    logger.info(f"Using {num_workers} workers for parallel processing")

    # Build list of all (offset, subscene) work items
    # Processing order: subscene outer loop -> offsets inner loop
    work_items = []
    for token, (s1_frames, stage1_start_frame, s2_frames, s2_hermite_start) in all_subscenes.items():
        for dx in X_OFFSETS:
            for dy in Y_OFFSETS:
                if args.skip_original and dx == 0 and dy == 0:
                    continue
                work_items.append((dx, dy, token, s1_frames, stage1_start_frame, s2_frames, s2_hermite_start))

    logger.info(f"Total work items: {len(work_items)} ({len(X_OFFSETS)*len(Y_OFFSETS) - (1 if args.skip_original else 0)} offsets x {len(all_subscenes)} sub-scenes)")

    # Create output dirs for all offsets
    for dx in X_OFFSETS:
        for dy in Y_OFFSETS:
            if args.skip_original and dx == 0 and dy == 0:
                continue
            (output_dir / offset_to_dirname(dx, dy)).mkdir(parents=True, exist_ok=True)

    # Run in parallel
    process_fn = partial(
        process_single_item,
        output_dir=str(output_dir),
        map_root=str(map_root),
        num_history=num_history,
        num_future=num_future,
        force=args.force,
        ps_num_poses=ps_num_poses,
        ps_interval=ps_interval,
    )

    total_valid = 0
    total_invalid_collision = 0
    total_invalid_drivable = 0
    total_invalid_behind = 0
    total_errors = 0
    offset_summary = {}

    results = []

    if num_workers == 1:
        for item in tqdm(work_items, desc="Processing all offsets"):
            if False:
                print(f'offset: {item[0]}, {item[1]}')
                print(item[2])
            results.append(process_fn(item))
    else:
        with Pool(num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(process_fn, work_items, chunksize=8),
                total=len(work_items),
                desc="Processing all offsets",
            ):
                results.append(result)

    # Aggregate results by offset
    for dx, dy, status in results:
        key = (dx, dy)
        if key not in offset_summary:
            offset_summary[key] = {"valid": 0, "collision": 0, "off_road": 0, "behind": 0, "errors": 0}
        if status == "valid" or status == "skipped":
            offset_summary[key]["valid"] += 1
            total_valid += 1
        elif status == "collision":
            offset_summary[key]["collision"] += 1
            total_invalid_collision += 1
        elif status == "off_road":
            offset_summary[key]["off_road"] += 1
            total_invalid_drivable += 1
        elif status == "behind":
            offset_summary[key]["behind"] += 1
            total_invalid_behind += 1
        elif status.startswith("error:"):
            offset_summary[key]["errors"] += 1
            total_errors += 1

    # Print per-offset summary
    for (dx, dy), stats in sorted(offset_summary.items()):
        logger.info(f"Offset ({dx:+d},{dy:+d}): valid={stats['valid']}, collision={stats['collision']}, off_road={stats['off_road']}, behind={stats['behind']}, errors={stats['errors']}")

    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total valid caches: {total_valid}")
    logger.info(f"Total collision rejections: {total_invalid_collision}")
    logger.info(f"Total off-road rejections: {total_invalid_drivable}")
    logger.info(f"Total behind rejections: {total_invalid_behind}")
    logger.info(f"Total errors: {total_errors}")
    logger.info("")

    # Print table
    header = f"{'Offset':>12} | {'Valid':>6} | {'Collision':>10} | {'Off-road':>10} | {'Behind':>8} | {'Errors':>6}"
    logger.info(header)
    logger.info("-" * len(header))
    for (dx, dy), stats in sorted(offset_summary.items()):
        offset_str = f"({dx:+d}, {dy:+d})"
        logger.info(
            f"{offset_str:>12} | {stats['valid']:>6} | {stats['collision']:>10} | {stats['off_road']:>10} | {stats['behind']:>8} | {stats['errors']:>6}"
        )

    # Save summary JSON
    summary_path = output_dir / "generation_summary.json"
    summary_data = {
        "x_offsets": X_OFFSETS,
        "y_offsets": Y_OFFSETS,
        "ego_length": EGO_LENGTH,
        "ego_width": EGO_WIDTH,
        "total_subscenes": len(all_subscenes),
        "offset_results": {
            offset_to_dirname(dx, dy): stats
            for (dx, dy), stats in offset_summary.items()
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
