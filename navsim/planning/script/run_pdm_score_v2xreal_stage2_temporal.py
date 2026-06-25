#!/usr/bin/env python
"""
V2X-Real Two-Stage PDM Score Evaluation Script

Two-stage evaluation for novel view robustness:
  Stage 1: Evaluate agent on original ego pose → score against original metric cache
  Stage 2: Evaluate agent on original ego pose → score against shifted metric caches
  
Combined score per token: stage1_score × mean(stage2_scores across offsets)
Final score: mean over all tokens

Usage:
    python run_pdm_score_v2xreal_stage2.py \
        --v2xreal_pkl_path /path/to/spd_infos_temporal_test.pkl \
        --map_root /path/to/maps/expansion \
        --metric_cache_path /path/to/metric_cache_v2xreal
"""

import argparse
import importlib
import logging
import pickle
import traceback
import lzma
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

import numpy as np
import pandas as pd

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
import os
    
from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.agents.stop_agent import StopAgent
from navsim.agents.human_agent import HumanAgent
from navsim.common.dataloader_v2xreal import SceneLoaderV2XReal, SceneFilter
from navsim.common.dataclasses import AgentInput, EgoStatus, PDMResults, SensorConfig, Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer, PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from navsim.traffic_agents_policies.constant_velocity_traffic_agents import (
    ConstantVelocityTrafficAgents,
)
from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy
from navsim.visualization.stage2_eval_viz import visualize_prediction_two_stage


def _build_traffic_agents_policy(
    policy_name: str,
    proposal_sampling: "TrajectorySampling",
    map_root: Path,
) -> AbstractTrafficAgentsPolicy:
    """Construct the background traffic-agents policy by name.

    log_replay   : ground-truth replay (default; matches stock V2X-Real eval).
    constant_velocity : extrapolate detected vehicles at constant velocity.
    idm          : reactive Intelligent Driver Model on V2X-Real lanes.
    """
    if policy_name == "log_replay":
        return LogReplayTrafficAgents(future_trajectory_sampling=proposal_sampling)
    if policy_name == "constant_velocity":
        return ConstantVelocityTrafficAgents(proposal_sampling)
    if policy_name == "idm":
        from navsim.common.dataloader_v2xreal import V2XRealMapWrapper
        from navsim.planning.simulation.observation.navsim_idm.v2xreal_idm_map_adapter import (
            V2XRealIDMMapAdapter,
        )
        from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
        from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents

        v2x_map = V2XRealIDMMapAdapter(V2XRealMapWrapper(map_root=Path(map_root)))
        idm_obs = NavsimIDMAgents(
            target_velocity=10.0,
            min_gap_to_lead_agent=1.0,
            headway_time=1.5,
            accel_max=1.0,
            decel_max=2.0,
            open_loop_detections_types=[],
            minimum_path_length=20,
            planned_trajectory_samples=None,
            planned_trajectory_sample_interval=None,
            radius=100,
            add_open_loop_parked_vehicles=True,
            idm_snap_threshold=3.0,
        )
        return NavsimIDMTrafficAgents(
            future_trajectory_sampling=proposal_sampling,
            idm_agents_observation=idm_obs,
            map_api=v2x_map,
        )
    raise ValueError(f"Unknown traffic_policy: {policy_name}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# V2XReal cam names (PKL) → navsim Cameras field names
V2XREAL_CAM_MAPPING: Dict[str, str] = {
    "cam1": "cam_f0",  # front
    "cam2": "cam_l0",  # left
    "cam3": "cam_r0",  # right
    "cam4": "cam_b0",  # rear
}
V2XREAL_INFRA_CAM_MAPPING: Dict[str, str] = {
    "cam1": "cam_infra0",  # infrastructure camera 1
    "cam2": "cam_infra1",  # infrastructure camera 2
}
_ALL_NS_CAM_NAMES = ["cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0", "cam_infra0", "cam_infra1"]
NUM_HISTORY_FRAMES = 4  # current + 3 past frames (standard navsim history)

# ── Stage1 novel-view mode ────────────────────────────────────────────────────
# Set _STAGE1_NOVEL_OFFSET = "x+0_y+0" to use test_novel images for Stage1.
# To revert to original behavior (test/ images), set to None.
# _STAGE1_NOVEL_OFFSET = "x+0_y+0"   # STAGE1_NOVEL: comment/set None to revert to original
_STAGE1_NOVEL_OFFSET = None        # ORIGINAL: uncomment to revert
# ─────────────────────────────────────────────────────────────────────────────


def _parse_offset_name(offset_name: str) -> Tuple[float, float]:
    """Parse offset directory name 'x+5_y-1' → (dx=5.0, dy=-1.0).

    :param offset_name: offset directory name string
    :return: (dx_ego, dy_ego) in meters; (0.0, 0.0) if unparseable
    """
    import re
    m = re.fullmatch(r"x([+-]?\d+)_y([+-]?\d+)", offset_name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


def _load_agent(
    agent_name: str,
    trajectory_sampling: TrajectorySampling,
    agent_class_path: Optional[str] = None,
    output_dir: Optional[Path] = None,
):
    if agent_class_path:
        module_path, class_name = agent_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)(trajectory_sampling=trajectory_sampling), class_name
    if agent_name == "constant_velocity":
        return ConstantVelocityAgent(trajectory_sampling=trajectory_sampling), agent_name
    if agent_name == "stop":
        return StopAgent(trajectory_sampling=trajectory_sampling), agent_name
    if agent_name == "human":
        return HumanAgent(trajectory_sampling=trajectory_sampling), agent_name
    if agent_name in ("cos_v2x", "sparsedrive_navsim"):  # sparsedrive_navsim = legacy alias
        from navsim.agents.cos_v2x_agent import CoSV2XAgent

        return CoSV2XAgent(trajectory_sampling=trajectory_sampling), agent_name
    raise ValueError(f"Unknown agent: {agent_name}")

def _load_scene_filter_config(cache_root: Path) -> dict:
    """Load num_history_frames, num_future_frames, frame_interval from metric cache's hydra config.yaml."""
    cache_cfg_path = Path(cache_root) / "metadata" / "code" / "hydra" / "config.yaml"
    with open(cache_cfg_path) as f:
        cache_cfg = yaml.safe_load(f)
    sf = cache_cfg.get("scene_filter", {})
    result = {
        "num_history": sf.get("num_history_frames", 10),
        "num_future": sf.get("num_future_frames", 40),
        "frame_interval": sf.get("frame_interval", 1),
    }
    logger.info(f"scene_filter config from cache: {result}")
    return result


def _build_future_token_map(pkl_infos: list, num_future: int, frame_interval: int) -> dict:
    """Build {pkl_token → future_pkl_token} where future is num_future*frame_interval frames later.

    Groups PKL infos by scene_token (preserving temporal order within each scene),
    then maps each token to the token that is num_future*frame_interval positions ahead.
    This corresponds to the t+4s frame (GT trajectory endpoint for Stage2).
    """
    from collections import defaultdict
    scene_tokens: dict = defaultdict(list)
    for info in pkl_infos:
        scene_tokens[info.get("scene_token", "unknown")].append(info["token"])

    step = num_future * frame_interval
    token_to_future: dict = {}
    for tokens in scene_tokens.values():
        for k, tok in enumerate(tokens):
            future_k = k + step
            if future_k < len(tokens):
                token_to_future[tok] = tokens[future_k]
    return token_to_future


def _load_metric_cache(path: Path) -> MetricCache:
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def _load_proposal_sampling_from_cache(cache_root: Path) -> TrajectorySampling:
    """
    Read proposal_sampling (num_poses, interval_length) from the metric cache's
    metadata/code/hydra/config.yaml. Falls back to (40, 0.1) if not found.
    """
    config_path = cache_root / "metadata" / "code" / "hydra" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        ps = cfg.get("proposal_sampling", {})
        num_poses = ps.get("num_poses", 40)
        interval_length = ps.get("interval_length", 0.1)
        logger.info(f"Loaded proposal_sampling from {config_path}")
    else:
        logger.warning(f"config.yaml not found at {config_path}, using default (num_poses=40, interval_length=0.1)")
        num_poses, interval_length = 40, 0.1
    logger.info(f"proposal_sampling: num_poses={num_poses}, interval_length={interval_length}")
    return TrajectorySampling(num_poses=num_poses, interval_length=interval_length)


def _make_ego_status(es, cmd=None, reference_pose: Optional[StateSE2] = None) -> EgoStatus:
    """Convert a nuplan EgoState to navsim EgoStatus.

    Args:
        es: nuplan EgoState.
        reference_pose: if None → global frame (in_global_frame=True).
                        if StateSE2 → ego-local frame relative to that pose
                        (in_global_frame=False). Typically pass mc.ego_state.rear_axle
                        to get poses relative to the current frame.
    """
    x = es.rear_axle.x
    y = es.rear_axle.y
    h = es.rear_axle.heading
    vx = es.dynamic_car_state.rear_axle_velocity_2d.x
    vy = es.dynamic_car_state.rear_axle_velocity_2d.y
    ax = es.dynamic_car_state.rear_axle_acceleration_2d.x
    ay = es.dynamic_car_state.rear_axle_acceleration_2d.y
    yaw_rate = es.dynamic_car_state.angular_velocity

    if reference_pose is not None:
        # Position: global → ego-local (rotate global offset by -current_heading)
        dx = x - reference_pose.x
        dy = y - reference_pose.y
        cos_pos = np.cos(-reference_pose.heading)
        sin_pos = np.sin(-reference_pose.heading)
        x = dx * cos_pos - dy * sin_pos
        y = dx * sin_pos + dy * cos_pos
        # Heading delta: history_heading - current_heading
        delta_h = normalize_angle(h - reference_pose.heading)
        h = delta_h
        # Velocity/acceleration: rear_axle_velocity_2d is in body frame.
        # Rotate from history body frame to current body frame by delta_h.
        cos_v = np.cos(delta_h)
        sin_v = np.sin(delta_h)
        vx, vy = vx * cos_v - vy * sin_v, vx * sin_v + vy * cos_v
        ax, ay = ax * cos_v - ay * sin_v, ax * sin_v + ay * cos_v
        # yaw_rate is frame-invariant (scalar angular velocity), no rotation needed
        in_global = False
    else:
        in_global = True
    
    if cmd is None:
        return EgoStatus(
            ego_pose=np.array([x, y, h], dtype=np.float64),
            ego_velocity=np.array([vx, vy], dtype=np.float32),
            ego_acceleration=np.array([ax, ay], dtype=np.float32),
            driving_command=np.array([0]),
            in_global_frame=in_global,
            ego_yaw_rate=float(yaw_rate),
        )
    else:
        return EgoStatus(
            ego_pose=np.array([x, y, h], dtype=np.float64),
            ego_velocity=np.array([vx, vy], dtype=np.float32),
            ego_acceleration=np.array([ax, ay], dtype=np.float32),
            driving_command=np.array([cmd]),
            in_global_frame=in_global,
            ego_yaw_rate=float(yaw_rate),
        )
        


def _agent_input_from_metric_cache(mc: MetricCache, ego_local: bool = False) -> AgentInput:
    """Create AgentInput with 4-frame ego history from metric cache (no sensor data).

    Args:
        mc: MetricCache for the current token.
        ego_local: if True, all ego_poses are expressed in the current frame's ego
                   coordinate system (in_global_frame=False).
                   if False (default), poses are in global frame (in_global_frame=True).
    """
    # past_human_trajectory is at 10Hz; pick states near t=-1.5s,-1.0s,-0.5s
    past_states = mc.past_human_trajectory.get_sampled_trajectory()
    n = len(past_states)
    # Indices counting from the end: 15 = ~1.5s ago, 10 = ~1.0s ago, 5 = ~0.5s ago
    history_idx = [max(0, n - 1 - 15), max(0, n - 1 - 10), max(0, n - 1 - 5)]
    ref = mc.ego_state.rear_axle if ego_local else None
    cmd = mc.driving_command
    
    ego_statuses = [_make_ego_status(past_states[i], cmd, reference_pose=ref) for i in history_idx]
    ego_statuses.append(_make_ego_status(mc.ego_state, cmd, reference_pose=ref))  # index 3 = current
    return AgentInput(ego_statuses=ego_statuses, cameras=[], lidars=[])


def _compute_infra_sensor_to_ego_lidar(
    info: Dict, infra_cam: Dict, infra: Dict,
    dx_ego: float = 0.0, dy_ego: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sensor2lidar transform for infrastructure camera w.r.t. ego LiDAR.

    Transform chain: sensor → infra_LiDAR → infra_ego → global → ego_ego → ego_LiDAR
    Since infra lidar2ego is typically identity, infra_LiDAR ≈ infra_ego.

    For Stage2, ego is shifted by (dx_ego, dy_ego) in ego frame. The shifted global
    translation is: x += dx*cos(h) - dy*sin(h), y += dx*sin(h) + dy*cos(h).

    :param info: ego vehicle frame info dict from PKL
    :param infra_cam: single infra camera dict (with sensor2lidar_rotation/translation, cam_intrinsic)
    :param infra: model_other_agent_inf dict (with lidar2ego, ego2global)
    :param dx_ego: forward offset in ego frame for Stage2 (0.0 for Stage1)
    :param dy_ego: leftward offset in ego frame for Stage2 (0.0 for Stage1)
    :return: (rotation 3x3, translation 3,) of sensor → ego_LiDAR transform
    """
    from pyquaternion import Quaternion as _Quaternion

    def make_T(rot, trans):
        T = np.eye(4)
        rot = np.array(rot)
        if rot.ndim == 1:  # quaternion [w, x, y, z]
            T[:3, :3] = _Quaternion(rot).rotation_matrix
        else:  # 3x3 rotation matrix
            T[:3, :3] = rot
        T[:3, 3] = np.array(trans)
        return T

    # sensor → infra LiDAR (rotation already 3x3 in PKL)
    T_s2il = make_T(infra_cam["sensor2lidar_rotation"], infra_cam["sensor2lidar_translation"])
    # infra LiDAR → infra ego (usually identity)
    T_il2ie = make_T(infra["lidar2ego_rotation"], infra["lidar2ego_translation"])
    # infra ego → global
    T_ie2g = make_T(infra["ego2global_rotation"], infra["ego2global_translation"])
    # global → (shifted) ego ego
    ego_rot = info["ego2global_rotation"]
    ego_xyz = np.array(info["ego2global_translation"])
    if dx_ego != 0.0 or dy_ego != 0.0:
        heading = _Quaternion(*ego_rot).yaw_pitch_roll[0]
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        ego_xyz = ego_xyz.copy()
        ego_xyz[0] += dx_ego * cos_h - dy_ego * sin_h
        ego_xyz[1] += dx_ego * sin_h + dy_ego * cos_h
    T_g2ee = np.linalg.inv(make_T(ego_rot, ego_xyz))
    # ego ego → ego LiDAR
    T_ee2el = np.linalg.inv(make_T(info["lidar2ego_rotation"], info["lidar2ego_translation"]))

    T = T_ee2el @ T_g2ee @ T_ie2g @ T_il2ie @ T_s2il
    return T[:3, :3], T[:3, 3]


def _build_v2xreal_agent_input(
    mc: MetricCache,
    token: str,
    info_dict: Dict,
    sensor_blob_path: Path,
    sensor_config: SensorConfig,
    offset_name: Optional[str] = None,
    ego_local: bool = True,
    future_info: Optional[Dict] = None,
) -> AgentInput:
    """
    Build 4-frame AgentInput with real sensor data for V2XReal.
    - Stage1: offset_name=None → images + LiDAR from test/
    - Stage2: offset_name="x+5_y+0" → images from test_novel/ (using future_info file paths)
    - future_info: raw PKL info for the t+4s frame; if provided, camera file paths are
      taken from this frame (the actual GT endpoint) and remapped to test_novel/.
    - ego_local: if True, ego_poses expressed in current ego frame (in_global_frame=False).
    """
    from navsim.common.dataclasses import Camera, Cameras, Lidar

    base = _agent_input_from_metric_cache(mc, ego_local=ego_local)  # 4-frame ego_statuses, no sensors

    # cache token = "{log_name}_{pkl_token}" (from _discover_metric_caches),
    # but PKL info_dict is keyed by pkl_token = "{scene}_{frame_idx}".
    # Recover pkl_token by stripping the log_name prefix.
    log_name = getattr(mc, "log_name", "")
    if log_name and token.startswith(log_name + "_"):
        pkl_token = token[len(log_name) + 1:]
    else:
        pkl_token = token
    info = info_dict.get(pkl_token) or info_dict.get(token)
    if info is None:
        return base  # fallback: ego history only

    _eff_info = info

    # For Stage2 (or Stage1 novel mode), use future_info (t+4s frame) for camera file paths;
    # for Stage1 original, use info (t=0 frame).
    _eff_offset = offset_name if offset_name is not None else _STAGE1_NOVEL_OFFSET  # STAGE1_NOVEL
    # _eff_offset = offset_name  # ORIGINAL: uncomment to revert
    cam_source_info = future_info if (_eff_offset is not None and future_info is not None) else info  # STAGE1_NOVEL
    # cam_source_info = future_info if (offset_name is not None and future_info is not None) else info  # ORIGINAL

    # Determine which sensors are needed for the current frame (index NUM_HISTORY_FRAMES-1)
    sensor_names = sensor_config.get_sensors_at_iteration(NUM_HISTORY_FRAMES - 1)
    # Build per-camera info with remapped names and (optionally) remapped paths
    cam_data: Dict[str, Dict] = {}
    for v2x_name, ns_name in V2XREAL_CAM_MAPPING.items():
        if v2x_name not in cam_source_info.get("cams", {}):
            continue
        ci = dict(cam_source_info["cams"][v2x_name])
        if _eff_offset is not None:  # STAGE1_NOVEL (was: if offset_name is not None)
            # Stage1 novel / Stage2: file path from t+4s frame → remap dir to test_novel/{offset}
            # Original path: test/{scene}/1/{file}
            parts = ci["data_path"].split("/")
            if len(parts) >= 4:
                ci["data_path"] = f"test_novel/{parts[1]}/1_{_eff_offset}/{parts[3]}"  # STAGE1_NOVEL
                # ci["data_path"] = f"test_novel/{parts[1]}/1_{offset_name}/{parts[3]}"  # ORIGINAL
        ci.setdefault("distortion", None)
        cam_data[ns_name] = ci

    # Load infra cameras (cooperative PKL only)
    # Extrinsic is computed relative to info (t=0 for Stage1, t=0 for Stage2 w/ shifted pose)
    infra = info.get("other_agent_info_dict", {}).get("model_other_agent_inf")
    infra_cam_source = (
        future_info.get("other_agent_info_dict", {}).get("model_other_agent_inf")
        if (_eff_offset is not None and future_info is not None)  # STAGE1_NOVEL (was: offset_name is not None)
        # if (offset_name is not None and future_info is not None)  # ORIGINAL
        else infra
    )
    if infra and infra_cam_source:
        dx_ego, dy_ego = _parse_offset_name(offset_name) if offset_name else (0.0, 0.0)
        for v2x_name, ns_name in V2XREAL_INFRA_CAM_MAPPING.items():
            ci = infra_cam_source.get("cams", {}).get(v2x_name)
            if ci is None:
                continue
            ci = dict(ci)
            if _eff_offset is not None:  # STAGE1_NOVEL (was: if offset_name is not None)
                # Stage1 novel / Stage2: file path from t+4s infra frame → remap to test_novel_infra/{offset}
                # Original path: test/{scene}/-2/{file}  (parts[2] = agent_id, e.g. "-2")
                parts = ci["data_path"].split("/")
                if len(parts) >= 4:
                    ci["data_path"] = f"test_novel_infra/{parts[1]}/{parts[2]}_{_eff_offset}/{parts[3]}"  # STAGE1_NOVEL
                    # ci["data_path"] = f"test_novel_infra/{parts[1]}/{parts[2]}_{offset_name}/{parts[3]}"  # ORIGINAL
            ci.setdefault("distortion", None)
            # Recompute extrinsic using ego info at t=0 (original ego), shifted by offset
            try:
                R, t = _compute_infra_sensor_to_ego_lidar(_eff_info, ci, infra, dx_ego=dx_ego, dy_ego=dy_ego)
                ci["sensor2lidar_rotation"] = R
                ci["sensor2lidar_translation"] = t
            except Exception as exc:
                logger.debug(f"Infra cam extrinsic computation failed for {ns_name}/{token}: {exc}")
                continue
            cam_data[ns_name] = ci
    # Load Camera objects for current frame
    cam_objects: Dict[str, "Camera"] = {}
    from PIL import Image as PILImage
    for ns_name in _ALL_NS_CAM_NAMES:
        ci = cam_data.get(ns_name)
        if ci is not None and ns_name in sensor_names:
            img_path = sensor_blob_path / ci["data_path"]
            # ###############################
            # if 'infra' not in ns_name:
            #     ci["data_path"][-16:-10] = ci["data_path"][-16:-10]
            # ###############################

            try:
                cam_objects[ns_name] = Camera(
                    image=np.array(PILImage.open(img_path)),
                    sensor2lidar_rotation=ci.get("sensor2lidar_rotation"),
                    sensor2lidar_translation=ci.get("sensor2lidar_translation"),
                    intrinsics=ci.get("cam_intrinsic"),
                    distortion=ci.get("distortion"),
                )
            except Exception as exc:
                logger.debug(f"Camera load failed for {ns_name}/{token}: {exc}")
                cam_objects[ns_name] = Camera()
        else:
            cam_objects[ns_name] = Camera()

    cameras_current = Cameras(**cam_objects)
    empty_cameras = Cameras(**{k: Camera() for k in _ALL_NS_CAM_NAMES})
    cameras_list = [empty_cameras] * (NUM_HISTORY_FRAMES - 1) + [cameras_current]

    # Load LiDAR (stage1 original only — no lidar for novel view)
    lidar_current = Lidar()
    if _eff_offset is None and "lidar_pc" in sensor_names:  # STAGE1_NOVEL (was: offset_name is None)
    # if offset_name is None and "lidar_pc" in sensor_names:  # ORIGINAL
        lidar_current = Lidar.from_lidar_dict(
            sensor_blob_path, info, "lidar", sensor_names
        )
    lidars_list = [Lidar()] * (NUM_HISTORY_FRAMES - 1) + [lidar_current]
    result = AgentInput(
        ego_statuses=base.ego_statuses,
        cameras=cameras_list,
        lidars=lidars_list,
    )

    # ── Populate global transform metadata for navsim-only agents ─────
    if info is not None:
        result.ego2global_translation = np.array(_eff_info["ego2global_translation"], dtype=np.float64)
        result.ego2global_rotation = np.array(_eff_info["ego2global_rotation"], dtype=np.float64)
        _l2e_r = info.get("lidar2ego_rotation")
        _l2e_t = info.get("lidar2ego_translation")
        if _l2e_r is not None:
            result.lidar2ego_rotation = np.array(_l2e_r, dtype=np.float64)
        if _l2e_t is not None:
            result.lidar2ego_translation = np.array(_l2e_t, dtype=np.float64)
        result.timestamp = float(info.get("timestamp", 0))
        result.token = pkl_token
        result.scene_token = info.get("scene_token")

    return result


def _is_offset_dir_name(name: str) -> bool:
    return name.startswith("x") and "_y" in name


def _path_has_offset_dir(path: Path) -> bool:
    return any(_is_offset_dir_name(part) for part in path.parts)


def _discover_metric_caches(cache_root: Path, include_offset_dirs: bool = True) -> Dict[str, Path]:
    """
    Build token → path mapping from cache directory.
    Expected structure: <cache_root>/<log_name>/unknown/<frame_token>/metric_cache.pkl
    Token format: "{log_name}_{frame_token}"
    """
    result: Dict[str, Path] = {}
    for mc in sorted(cache_root.rglob("metric_cache.pkl")):
        if not include_offset_dirs and _path_has_offset_dir(mc):
            continue
        frame_token = mc.parent.name
        log_name = mc.parent.parent.parent.name
        token = f"{log_name}_{frame_token}"
        result[token] = mc
    return result


def _build_temporal_schedule(
    pkl_infos: list,
    stage1_caches: Dict[str, Path],
    stage2_caches: Dict[str, Dict[str, Path]],
    num_future: int,
    frame_interval: int,
) -> "OrderedDict[str, List[Tuple[int, str, str, Optional[List[str]]]]]":
    """Build per-scene temporally ordered evaluation schedule.

    s2[tok] is placed at (frame_idx_of_tok + step) where step = num_future * frame_interval.
    When s1 and s2 land at the same time index, s2 is scheduled first.

    Returns:
        OrderedDict { scene_token: [(time_idx, stage, cache_token, offset_names_or_None), ...] }
        sorted by (time_idx, stage_priority) where s2 (priority=0) comes before s1 (priority=1).
    """
    from collections import defaultdict, OrderedDict

    step = num_future * frame_interval

    # 1. pkl_token → (scene_token, frame_idx) using actual frame_idx from PKL info
    pkl_token_info: Dict[str, Tuple[str, int]] = {}
    for info in pkl_infos:
        scene_token = info.get("scene_token", "unknown")
        pkl_token_info[info["token"]] = (scene_token, info["frame_idx"])

    # 3. cache_token → pkl_token (from metric cache path structure)
    cache_to_pkl: Dict[str, str] = {}
    for cache_token, path in stage1_caches.items():
        frame_token = path.parent.name  # = pkl_token
        cache_to_pkl[cache_token] = frame_token

    # 4. Build schedule: list of (time_idx, priority, stage, cache_token, offsets)
    scene_schedules: Dict[str, List] = defaultdict(list)
    for cache_token in stage1_caches:
        pkl_tok = cache_to_pkl.get(cache_token)
        if pkl_tok is None or pkl_tok not in pkl_token_info:
            continue
        scene_token, frame_idx = pkl_token_info[pkl_tok]

        # s1 task at frame_idx
        scene_schedules[scene_token].append((frame_idx, 1, "s1", cache_token, None))

        # s2 task at frame_idx + step (if any offset caches exist)
        s2_offsets = [on for on, oc in stage2_caches.items() if cache_token in oc]
        if s2_offsets:
            scene_schedules[scene_token].append((frame_idx + step, 0, "s2", cache_token, s2_offsets))

    # 5. Sort each scene: (time_idx, priority) → s2 first at same time
    result = OrderedDict()
    for scene_token in sorted(scene_schedules.keys()):
        tasks = scene_schedules[scene_token]
        tasks.sort(key=lambda x: (x[0], x[1]))
        # Flatten to (time_idx, stage, cache_token, offsets)
        result[scene_token] = [(t[0], t[2], t[3], t[4]) for t in tasks]

    return result


def _discover_offset_dirs(cache_root: Path) -> List[Tuple[str, Path]]:
    """
    Find all offset directories (x*_y*/) under cache_root.
    Returns list of (offset_name, offset_dir_path).
    """
    offsets = []
    for d in sorted(cache_root.iterdir()):
        if d.is_dir() and _is_offset_dir_name(d.name):
            offsets.append((d.name, d))
    return offsets


def _score_single(
    token: str,
    trajectory: Trajectory,
    metric_cache: MetricCache,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run pdm_score and attach metadata columns."""
    expected_steps = simulator.proposal_sampling.num_poses
    cache_steps = len(metric_cache.future_tracked_objects)
    if cache_steps != expected_steps:
        raise ValueError(
            "Metric cache sampling mismatch: "
            f"scorer expects {expected_steps} steps "
            f"(num_poses={simulator.proposal_sampling.num_poses}, "
            f"interval_length={simulator.proposal_sampling.interval_length}), "
            f"but cache has {cache_steps} future steps. "
            "Regenerate metric caches with matching proposal sampling."
        )

    score_row, simulated_states = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=trajectory,
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=traffic_agents_policy,
    )
    score_row["valid"] = True
    score_row["token"] = token
    score_row["log_name"] = metric_cache.log_name
    score_row["start_time"] = metric_cache.timepoint.time_s

    end_pose = StateSE2(
        x=trajectory.poses[-1, 0],
        y=trajectory.poses[-1, 1],
        heading=trajectory.poses[-1, 2],
    )
    abs_end = relative_to_absolute_poses(metric_cache.ego_state.rear_axle, [end_pose])[0]
    score_row["endpoint_x"] = abs_end.x
    score_row["endpoint_y"] = abs_end.y
    score_row["start_point_x"] = metric_cache.ego_state.rear_axle.x
    score_row["start_point_y"] = metric_cache.ego_state.rear_axle.y
    score_row["start_heading"] = metric_cache.ego_state.rear_axle.heading
    return score_row, simulated_states


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="V2X-Real Two-Stage PDM Score Evaluation")
    parser.add_argument("--v2xreal_pkl_path", type=str, required=True)
    parser.add_argument("--map_root", type=str, required=True)
    parser.add_argument("--metric_cache_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--agent", type=str, default="constant_velocity",
                        choices=["constant_velocity", "stop", "human", "cos_v2x", "sparsedrive_navsim"])
    parser.add_argument("--agent_class_path", type=str, default=None)
    parser.add_argument("--sensor_blob_path", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Limit number of tokens for debugging")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualizations with stage1/stage2/combined scores")
    parser.add_argument("--viz_output_dir", type=str, default=None,
                        help="Output directory for visualizations (default: <output_dir>/viz)")
    parser.add_argument("--max_viz", type=int, default=None,
                        help="Maximum number of visualizations to save (default: all)")
    parser.add_argument("--stage1_only", action="store_true",
                        help="Run Stage 1 evaluation only; skip Stage 2 offsets and combined score")
    parser.add_argument("--traffic_policy", type=str, default="log_replay",
                        choices=["log_replay", "idm", "constant_velocity"],
                        help="Background traffic agents policy: log_replay (GT replay, default), "
                             "idm (reactive Intelligent Driver Model), or constant_velocity")
    args = parser.parse_args()

    cache_root = Path(args.metric_cache_path)
    map_root = Path(args.map_root)

    logger.info("=" * 70)
    logger.info("V2X-Real Two-Stage PDM Score Evaluation")
    logger.info("=" * 70)

    # Setup components
    proposal_sampling = _load_proposal_sampling_from_cache(cache_root)
    # for more accurate DAC evaluation on V2XReal vehicles
    mondeo_params = VehicleParameters(
        width=1.700,
        front_length=3.603,
        rear_length=0.997,
        cog_position_from_rear_axle=1.620,
        wheel_base=2.850,
        vehicle_name="mondeo",
        vehicle_type="gen1",
        height=1.495,
    )
    scorer_stage1 = PDMScorer(
        proposal_sampling=proposal_sampling,
        config=PDMScorerConfig(use_pdms_v1=False, human_penalty_filter=False),
        vehicle_parameters=mondeo_params,
    )
    scorer_stage2 = PDMScorer(
        proposal_sampling=proposal_sampling,
        config=PDMScorerConfig(use_pdms_v1=False, human_penalty_filter=False),
        vehicle_parameters=mondeo_params,
    )
    simulator = PDMSimulator(proposal_sampling=proposal_sampling)
    traffic_agents_policy = _build_traffic_agents_policy(
        args.traffic_policy, proposal_sampling, map_root,
    )
    logger.info(f"Traffic agents policy: {args.traffic_policy}")

    agent, resolved_agent_name = _load_agent(args.agent, proposal_sampling, args.agent_class_path)
    if hasattr(agent, "initialize"):
        agent.initialize()

    output_agent_name = f"{resolved_agent_name}_stage1" if args.stage1_only else resolved_agent_name
    # Suffix with traffic policy so different policies don't overwrite each other
    output_agent_name = f"{output_agent_name}_{args.traffic_policy}"
    output_dir = Path(args.output_dir) / output_agent_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(agent, "output_dir"):
        agent.output_dir = output_dir / "camera_viz"
        agent.output_dir.mkdir(parents=True, exist_ok=True)
    viz_output_dir = None
    if args.visualize:
        viz_output_dir = Path(args.viz_output_dir) if args.viz_output_dir else output_dir / "viz"
        viz_output_dir.mkdir(parents=True, exist_ok=True)

    # Detect whether the agent needs real sensor data (cameras/lidar)
    import dataclasses as _dc
    sensor_cfg = agent.get_sensor_config()
    needs_sensors = any(
        v is True or (isinstance(v, list) and len(v) > 0)
        for v in _dc.asdict(sensor_cfg).values()
    )
    ego_local = getattr(agent, 'wants_ego_local', False)
    logger.info(f"Agent needs sensors: {needs_sensors}, ego_local: {ego_local}")

    # Always load PKL for temporal ordering
    with open(args.v2xreal_pkl_path, "rb") as _f:
        _pkl = pickle.load(_f)
    pkl_infos = _pkl["infos"]
    logger.info(f"Loaded PKL: {len(pkl_infos)} frames")

    sf_cfg = _load_scene_filter_config(cache_root)
    s2_frame_offset = sf_cfg["num_future"] * sf_cfg["frame_interval"]
    logger.info(f"s2 temporal offset: {s2_frame_offset} frames "
                f"(num_future={sf_cfg['num_future']} * frame_interval={sf_cfg['frame_interval']})")

    # Sensor-specific loading (conditional)
    info_dict: Dict = {}
    token_to_future_pkl_token: Dict[str, str] = {}
    sensor_blob_path: Optional[Path] = None
    if args.sensor_blob_path:
        sensor_blob_path = Path(args.sensor_blob_path)
        info_dict = {info["token"]: info for info in pkl_infos}
        logger.info(f"Loaded PKL info dict: {len(info_dict)} tokens (sensor_blob_path={sensor_blob_path})")

        # Build t=0 token → t+Ns token mapping for Stage2 sensor data
        token_to_future_pkl_token = _build_future_token_map(
            pkl_infos,
            num_future=sf_cfg["num_future"],
            frame_interval=sf_cfg["frame_interval"],
        )
        logger.info(f"Built future token map: {len(token_to_future_pkl_token)} entries "
                    f"(step={s2_frame_offset} frames)")

    elif needs_sensors:
        logger.warning("Agent needs sensors but --sensor_blob_path not provided; will return empty cameras/lidar")

    # Load scene loader (used only for requires_scene=True agents)
    scene_filter = SceneFilter(
        num_history_frames=sf_cfg["num_history"],
        num_future_frames=sf_cfg["num_future"],
        frame_interval=sf_cfg["frame_interval"],
    )
    scene_loader = SceneLoaderV2XReal(
        pkl_path=Path(args.v2xreal_pkl_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        sensor_blob_path=sensor_blob_path,
        map_root=map_root,
    )
    logger.info(f"Scene loader: {len(scene_loader)} scenes")

    # Discover stage1 (original) metric caches only from non-offset paths
    stage1_caches = _discover_metric_caches(cache_root, include_offset_dirs=False)
    logger.info(f"Stage 1 caches: {len(stage1_caches)}")

    # Discover stage2 (offset) metric caches
    stage2_caches: Dict[str, Dict[str, Path]] = {}  # offset_name → {token → path}
    if not args.stage1_only:
        offset_dirs = _discover_offset_dirs(cache_root)
        for offset_name, offset_dir in offset_dirs:
            caches = _discover_metric_caches(offset_dir)
            stage2_caches[offset_name] = caches
            logger.info(f"  {offset_name}: {len(caches)} caches")
    logger.info(f"Stage 2 offsets: {len(stage2_caches)}")

    # Build temporal schedule: scene → [(time_idx, stage, token, offsets), ...]
    temporal_schedule = _build_temporal_schedule(
        pkl_infos, stage1_caches, stage2_caches,
        num_future=sf_cfg["num_future"],
        frame_interval=sf_cfg["frame_interval"],
    )
    total_tasks = sum(len(v) for v in temporal_schedule.values())
    total_tokens = sum(1 for v in temporal_schedule.values() for _, st, _, _ in v if st == "s1")
    logger.info(f"Temporal schedule: {total_tasks} tasks ({total_tokens} s1 tokens) "
                f"across {len(temporal_schedule)} scenes")

    if args.max_tokens:
        # Limit total s1 tokens evaluated (across all scenes)
        limited_schedule = {}
        s1_count = 0
        for scene_tok, tasks in temporal_schedule.items():
            kept = []
            for task in tasks:
                if task[1] == "s1":
                    if s1_count >= args.max_tokens:
                        break
                    s1_count += 1
                kept.append(task)
            if kept:
                limited_schedule[scene_tok] = kept
            if s1_count >= args.max_tokens:
                break
        temporal_schedule = limited_schedule
        total_tasks = sum(len(v) for v in temporal_schedule.values())
        logger.info(f"  (limited to {args.max_tokens} s1 tokens → {total_tasks} tasks)")

    # ===== Evaluate (temporal order: s2 before s1 at same time) =====
    SIGMA_SQUARED = 1.0
    sub_metric_cols = [
        "no_at_fault_collisions", "drivable_area_compliance",
        "driving_direction_compliance",
        "ego_progress", "time_to_collision_within_bound",
        "lane_keeping", "history_comfort", "pdm_score",
    ]

    stage1_results: List[pd.DataFrame] = []
    stage2_results: List[pd.DataFrame] = []
    agent_trajectories: Dict[str, Trajectory] = {}
    stage1_simulated_states: Dict[str, np.ndarray] = {}
    stage1_rows_by_token: Dict[str, pd.DataFrame] = {}  # for combined score after s1
    stage2_rows_by_token: Dict[str, List[pd.DataFrame]] = {}  # s2 results accumulated per token
    stage2_trajectories: Dict[str, Dict[str, Trajectory]] = {}
    stage2_simulated_states: Dict[str, Dict[str, np.ndarray]] = {}
    combined_rows_inline: List[Dict] = []

    max_viz = args.max_viz if args.max_viz is not None else total_tokens
    viz_count = 0
    viz_eligible_count = 0

    logger.info("")
    logger.info("=" * 70)
    _mode_str = "Stage1 only" if args.stage1_only else "Temporal (s2 before s1)"
    logger.info(f"Evaluating tokens ({_mode_str})")
    logger.info("=" * 70)

    global_idx = 0
    for scene_token, schedule in temporal_schedule.items():
        logger.info(f"  Scene: {scene_token} ({len(schedule)} tasks)")
        # Reset temporal state at scene boundary to prevent cross-scene bank contamination
        if hasattr(agent, '_saved_temporal_state'):
            agent._saved_temporal_state = None
        # if scene_token != "2023-04-05-16-25-26_22_0_folder_2_-1": 
            # continue
        for time_idx, stage, token, offset_names in schedule:
            # if token != '2023-04-04-14-27-53_44_0_folder_1_-1_2023-04-04-14-27-53_44_0_folder_1_-1_000016_folder_1_-1': continue

            global_idx += 1

            # ── Stage 1 (processed first: s1 at frame_idx F) ─────────────────
            if stage == "s1":
                logger.info(f"    [{global_idx}/{total_tasks}] t={time_idx} S1 {token}")
                mc_s1 = None
                s1_simulated = None
                try:
                    mc_s1 = _load_metric_cache(stage1_caches[token])
                    # pass current stage info to the agent
                    if hasattr(agent, 'current_stage'):
                        agent.current_stage = 's1'
                        agent.current_offset = None
                    if agent.requires_scene:
                        agent_input = scene_loader.get_agent_input_from_token(token)
                        scene = scene_loader.get_scene_from_token(token)
                        traj_s1 = agent.compute_trajectory(agent_input, scene)
                    elif needs_sensors and sensor_blob_path is not None:
                        agent_input = _build_v2xreal_agent_input(
                            mc_s1, token, info_dict, sensor_blob_path, sensor_cfg,
                            offset_name=None,
                        )
                        traj_s1 = agent.compute_trajectory(agent_input)
                    else:
                        _ai_s1 = _agent_input_from_metric_cache(mc_s1, ego_local=ego_local)
                        _log = getattr(mc_s1, "log_name", "")
                        _ai_s1.token = token[len(_log) + 1:] if (_log and token.startswith(_log + "_")) else token
                        traj_s1 = agent.compute_trajectory(_ai_s1)
                    agent_trajectories[token] = traj_s1

                    s1_row, s1_simulated = _score_single(
                        token, traj_s1, mc_s1, simulator, scorer_stage1, traffic_agents_policy
                    )
                    s1_row["stage"] = "stage1"
                    stage1_simulated_states[token] = s1_simulated
                except Exception:
                    logger.warning(f"      Stage1 FAILED: {traceback.format_exc()}")
                    s1_row = pd.DataFrame([PDMResults.get_empty_results()])
                    s1_row["valid"] = False
                    s1_row["token"] = token
                    s1_row["stage"] = "stage1"
                stage1_results.append(s1_row)
                stage1_rows_by_token[token] = s1_row

                # stage1_only visualization
                s1_ser = s1_row.iloc[0] if isinstance(s1_row, pd.DataFrame) else s1_row
                s1_valid = bool(s1_ser.get("valid", False))
                if args.stage1_only and args.visualize and viz_output_dir is not None \
                        and viz_count < max_viz and s1_valid:
                    try:
                        if mc_s1 is None:
                            mc_s1 = _load_metric_cache(stage1_caches[token])
                        output_path = viz_output_dir / f"{token}.png"
                        visualize_prediction_two_stage(
                            metric_cache=mc_s1,
                            pred_trajectory=agent_trajectories.get(token),
                            stage1_row=s1_ser,
                            stage2_data=None,
                            combined_row=None,
                            output_path=output_path,
                            map_root=map_root,
                            simulated_states=s1_simulated,
                        )
                        viz_count += 1
                    except Exception:
                        logger.warning(f"      Viz FAILED {token}: {traceback.format_exc()}")

            # ── Stage 2 (processed later: s2 at frame_idx F+step) ─────────
            elif stage == "s2":
                logger.info(f"    [{global_idx}/{total_tasks}] t={time_idx} S2 {token} (offsets={len(offset_names)})")
                for offset_name in sorted(offset_names):
                    offset_caches = stage2_caches[offset_name]
                    if token not in offset_caches:
                        continue
                    try:
                        mc_s2 = _load_metric_cache(offset_caches[token])
                        # pass current stage info to the agent
                        if hasattr(agent, 'current_stage'):
                            agent.current_stage = 's2'
                            agent.current_offset = offset_name
                        if agent.requires_scene:
                            traj_s2 = agent_trajectories.get(token)
                            if traj_s2 is None:
                                traj_s2 = agent.compute_trajectory(
                                    _agent_input_from_metric_cache(mc_s2, ego_local=ego_local))
                        elif needs_sensors and sensor_blob_path is not None:
                            _log = getattr(mc_s2, "log_name", "")
                            _pkl_tok = token[len(_log) + 1:] if (_log and token.startswith(_log + "_")) else token
                            _future_pkl_tok = token_to_future_pkl_token.get(_pkl_tok)
                            future_info_s2 = info_dict.get(_future_pkl_tok) if _future_pkl_tok else None
                            agent_input_s2 = _build_v2xreal_agent_input(
                                mc_s2, token, info_dict, sensor_blob_path, sensor_cfg,
                                offset_name=offset_name,
                                future_info=future_info_s2,
                            )
                            traj_s2 = agent.compute_trajectory(agent_input_s2)
                        else:
                            _ai_s2 = _agent_input_from_metric_cache(mc_s2, ego_local=ego_local)
                            _log_s2 = getattr(mc_s2, "log_name", "")
                            _ai_s2.token = token[len(_log_s2) + 1:] if (_log_s2 and token.startswith(_log_s2 + "_")) else token
                            traj_s2 = agent.compute_trajectory(_ai_s2)
                        s2_row, s2_simulated = _score_single(
                            token, traj_s2, mc_s2, simulator, scorer_stage2, traffic_agents_policy
                        )
                        s2_row["stage"] = "stage2"
                        s2_row["offset"] = offset_name
                        stage2_trajectories.setdefault(token, {})[offset_name] = traj_s2
                        stage2_simulated_states.setdefault(token, {})[offset_name] = s2_simulated
                    except Exception:
                        logger.warning(f"      Stage2 FAILED {offset_name}/{token}: {traceback.format_exc()}")
                        s2_row = pd.DataFrame([PDMResults.get_empty_results()])
                        s2_row["valid"] = False
                        s2_row["token"] = token
                        s2_row["stage"] = "stage2"
                        s2_row["offset"] = offset_name
                    stage2_results.append(s2_row)
                    stage2_rows_by_token.setdefault(token, []).append(s2_row)

                # ── Combined score (s1 was already processed at t=frame_idx) ──
                s1_row = stage1_rows_by_token.get(token)
                if s1_row is None:
                    continue
                s1_ser = s1_row.iloc[0] if isinstance(s1_row, pd.DataFrame) else s1_row
                s1_valid = bool(s1_ser.get("valid", False))
                s1_simulated = stage1_simulated_states.get(token)

                s2_rows_this_token = stage2_rows_by_token.get(token, [])
                if s2_rows_this_token:
                    s2_all = pd.concat(s2_rows_this_token, ignore_index=True)
                    s2_valid_tok = s2_all[s2_all["valid"] == True].reset_index(drop=True)
                else:
                    s2_valid_tok = pd.DataFrame()

                if s1_valid and len(s2_valid_tok) > 0:
                    if s1_simulated is not None and len(s1_simulated) > 0:
                        s1_endpoint_x = float(s1_simulated[-1, StateIndex.X])
                        s1_endpoint_y = float(s1_simulated[-1, StateIndex.Y])
                    else:
                        s1_endpoint_x = float(s1_ser["endpoint_x"])
                        s1_endpoint_y = float(s1_ser["endpoint_y"])

                    d_sq = (s1_endpoint_x - s2_valid_tok["start_point_x"]) ** 2 \
                         + (s1_endpoint_y - s2_valid_tok["start_point_y"]) ** 2
                    weights = np.exp(-d_sq.values / (2 * SIGMA_SQUARED))
                    weight_sum = weights.sum()
                    if np.isclose(weight_sum, 0.0) or np.isnan(weight_sum):
                        weights = np.ones(len(s2_valid_tok)) / len(s2_valid_tok)
                    else:
                        weights = weights / weight_sum

                    row_data = {"token": token}
                    for col in sub_metric_cols:
                        if col in s1_ser.index and col in s2_valid_tok.columns:
                            s1_val = float(s1_ser[col])
                            s2_val = float((s2_valid_tok[col].values * weights).sum())
                            row_data[f"stage1_{col}"] = s1_val
                            row_data[f"stage2_{col}"] = s2_val
                            row_data[f"combined_{col}"] = s1_val * s2_val
                    combined_rows_inline.append(row_data)
                    combined_ser = pd.Series(row_data)

                    if (
                        args.visualize
                        and viz_output_dir is not None
                        and viz_count < max_viz
                        and viz_eligible_count % 4 == 0
                    ):
                        try:
                            mc_s1 = _load_metric_cache(stage1_caches[token])
                            stage2_data_viz: Dict[str, Dict] = {}
                            for j, s2r in s2_valid_tok.iterrows():
                                off = s2r["offset"]
                                stage2_data_viz[off] = {
                                    "simulated_states": stage2_simulated_states.get(token, {}).get(off),
                                    "trajectory": stage2_trajectories.get(token, {}).get(off),
                                    "start_x": float(s2r["start_point_x"]),
                                    "start_y": float(s2r["start_point_y"]),
                                    "heading": float(s2r.get("start_heading", mc_s1.ego_state.rear_axle.heading)),
                                    "weight": float(weights[j]),
                                    "metrics": s2r,
                                }
                            output_path = viz_output_dir / f"{token}.png"
                            visualize_prediction_two_stage(
                                metric_cache=mc_s1,
                                pred_trajectory=agent_trajectories.get(token),
                                stage1_row=s1_ser,
                                stage2_data=stage2_data_viz,
                                combined_row=combined_ser,
                                output_path=output_path,
                                map_root=map_root,
                                simulated_states=s1_simulated,
                            )
                            viz_count += 1
                            if viz_count % 50 == 0:
                                logger.info(f"      Saved {viz_count} visualizations so far")
                        except Exception:
                            logger.warning(f"      Viz FAILED {token}: {traceback.format_exc()}")

                    viz_eligible_count += 1

    # ===== Aggregate results =====
    logger.info("")
    logger.info("=" * 70)
    logger.info("Aggregating results")
    logger.info("=" * 70)

    stage1_df = pd.concat(stage1_results, ignore_index=True)
    stage2_df = pd.concat(stage2_results, ignore_index=True) if stage2_results else pd.DataFrame()

    # Save raw results
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    stage1_df.to_csv(output_dir / f"{timestamp}_stage1_raw.csv", index=False)
    if len(stage2_df) > 0:
        stage2_df.to_csv(output_dir / f"{timestamp}_stage2_raw.csv", index=False)

    # Score columns
    score_cols = [c for c in stage1_df.columns
                  if c not in {"valid", "token", "log_name", "start_time", "stage", "offset",
                               "endpoint_x", "endpoint_y", "start_point_x", "start_point_y",
                               "start_heading",
                               "frame_type", "weighted_metrics", "weighted_metrics_array",
                               "traffic_light_compliance"}
                  and stage1_df[c].dtype in [np.float64, np.float32, float, int]]

    # Save overall mean summary for stage1
    valid_s1 = stage1_df[stage1_df["valid"] == True]
    if len(valid_s1) > 0:
        s1_summary = valid_s1[sub_metric_cols].mean().to_frame(name="mean").T
        s1_summary.to_csv(output_dir / f"{timestamp}_stage1_summary.csv", index=False)

    # Save overall mean summary for stage2
    if len(stage2_df) > 0:
        valid_s2_all = stage2_df[stage2_df["valid"] == True]
        if len(valid_s2_all) > 0:
            s2_summary = valid_s2_all[sub_metric_cols].mean().to_frame(name="mean").T
            s2_summary.to_csv(output_dir / f"{timestamp}_stage2_summary.csv", index=False)

    # --- Stage 1 stats ---
    logger.info("")
    logger.info("Stage 1 Results (original ego pose):")
    valid_s1 = stage1_df[stage1_df["valid"] == True]
    for col in score_cols:
        if col in valid_s1.columns:
            logger.info(f"  {col:35s}: {valid_s1[col].mean():8.4f} ± {valid_s1[col].std():8.4f}")

    # --- Stage 2 stats per offset ---
    if len(stage2_df) > 0:
        logger.info("")
        logger.info("Stage 2 Results (per offset):")
        valid_s2 = stage2_df[stage2_df["valid"] == True]
        for offset_name in sorted(valid_s2["offset"].unique()):
            offset_data = valid_s2[valid_s2["offset"] == offset_name]
            pdm = offset_data["pdm_score"].mean() if "pdm_score" in offset_data.columns else float("nan")
            logger.info(f"  {offset_name:>12s}: pdm_score={pdm:.4f}  (n={len(offset_data)})")

        # --- Stage 2 aggregate (mean across offsets per token) ---
        logger.info("")
        logger.info("Stage 2 Aggregate (mean across all offsets):")
        for col in score_cols:
            if col in valid_s2.columns:
                logger.info(f"  {col:35s}: {valid_s2[col].mean():8.4f} ± {valid_s2[col].std():8.4f}")

    # --- Combined two-stage score (from inline computation) ---
    if combined_rows_inline:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Combined Two-Stage Score (Gaussian kernel weighted)")
        logger.info("=" * 70)

        combined_df = pd.DataFrame(combined_rows_inline)
        combined_df.to_csv(output_dir / f"{timestamp}_combined.csv", index=False)
        logger.info(f"Tokens with combined score: {len(combined_df)}")

        # Compute and save overall mean summary
        # Column order: n_tokens | combined_pdm_score | stage1_* | stage2_*
        summary_row = {"n_tokens": len(combined_df)}
        summary_row["combined_pdm_score"] = combined_df["combined_pdm_score"].mean() if "combined_pdm_score" in combined_df.columns else float("nan")
        for col in sub_metric_cols:
            key = f"stage1_{col}"
            summary_row[key] = combined_df[key].mean() if key in combined_df.columns else float("nan")
        for col in sub_metric_cols:
            key = f"stage2_{col}"
            summary_row[key] = combined_df[key].mean() if key in combined_df.columns else float("nan")
        summary_df = pd.DataFrame([summary_row])
        summary_df.to_csv(output_dir / f"{timestamp}_combined_summary.csv", index=False)

        logger.info(f"  combined_pdm_score: {summary_row['combined_pdm_score']:.4f}")
        logger.info(f"  {'metric':35s} {'stage1':>8s} {'stage2':>8s}")
        logger.info(f"  {'-'*35} {'-'*8} {'-'*8}")
        for col in sub_metric_cols:
            s1_mean = summary_row.get(f"stage1_{col}", float("nan"))
            s2_mean = summary_row.get(f"stage2_{col}", float("nan"))
            logger.info(f"  {col:35s} {s1_mean:8.4f} {s2_mean:8.4f}")

        if args.visualize:
            logger.info(f"\nVisualization: {viz_count} images saved to {viz_output_dir}")

    logger.info("")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)
    logger.info("Evaluation Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
