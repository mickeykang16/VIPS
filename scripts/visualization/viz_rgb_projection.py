#!/usr/bin/env python3
"""Visualize V2X-Real camera images with GT 3D boxes AND map lanes projected on them.

For one token's current frame, renders a grid of the vehicle cameras
(front / left / right / back) plus the cooperative infrastructure cameras, with:
  - GT 3D bounding boxes projected onto each image (reuses
    ``navsim.visualization.camera.add_annotations_to_camera_ax``), and
  - nearby map-lane centerlines projected onto each image (new, ~30 lines).

Both GT boxes and map lanes are stored in the GLOBAL frame, so they are first
transformed GLOBAL -> EGO(==LiDAR) using the current ego pose, then EGO -> CAMERA
with the same ``sensor2lidar_*`` convention used by ``navsim/visualization/camera.py``.

Run (vips env):
    source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate vips
    "$CONDA_PREFIX/bin/python" scripts/visualization/viz_rgb_projection.py
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
from pyquaternion import Quaternion

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from navsim.common.dataclasses import Annotations, Camera, SensorConfig  # noqa: E402
from navsim.common.dataloader_v2xreal import (  # noqa: E402
    SceneFilter,
    SceneLoaderV2XReal,
    V2XRealMapWrapper,
)
from navsim.visualization.camera import (  # noqa: E402
    _transform_points_to_image,
    add_annotations_to_camera_ax,
)

# V2XReal PKL camera names -> navsim Cameras field names (mirrors the eval script).
_VEH_CAM_MAPPING: Dict[str, str] = {
    "cam1": "cam_f0",  # front
    "cam2": "cam_l0",  # left
    "cam3": "cam_r0",  # right
    "cam4": "cam_b0",  # rear
}
_INFRA_CAM_MAPPING: Dict[str, str] = {
    "cam1": "cam_infra0",
    "cam2": "cam_infra1",
}
# Grid layout: (navsim field name, human-readable title).
_GRID_LAYOUT: List[Tuple[str, str]] = [
    ("cam_f0", "Vehicle Front"),
    ("cam_l0", "Vehicle Left"),
    ("cam_r0", "Vehicle Right"),
    ("cam_b0", "Vehicle Back"),
    ("cam_infra0", "Infra Cam 0"),
    ("cam_infra1", "Infra Cam 1"),
]
_NUM_HISTORY_FRAMES = 4  # current frame index == 3


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


def _make_T(rot, trans) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a rotation (quat wxyz or 3x3) + translation."""
    T = np.eye(4)
    rot = np.asarray(rot, dtype=np.float64)
    T[:3, :3] = Quaternion(rot).rotation_matrix if rot.ndim == 1 else rot
    T[:3, 3] = np.asarray(trans, dtype=np.float64)
    return T


def _build_camera(cam_info: Dict, blob: Path) -> Optional[Camera]:
    """Build a navsim Camera from a PKL camera info dict."""
    img_path = blob / cam_info["data_path"]
    if not img_path.exists():
        return None
    return Camera(
        image=np.array(Image.open(img_path)),
        sensor2lidar_rotation=np.asarray(cam_info["sensor2lidar_rotation"], dtype=np.float64),
        sensor2lidar_translation=np.asarray(cam_info["sensor2lidar_translation"], dtype=np.float64),
        intrinsics=np.asarray(cam_info["cam_intrinsic"], dtype=np.float64),
    )


def _compute_infra_sensor_to_ego_lidar(info: Dict, infra_cam: Dict, infra: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """sensor -> infra_LiDAR -> infra_ego -> global -> ego_ego -> ego_LiDAR (mirrors eval script)."""
    T_s2il = _make_T(infra_cam["sensor2lidar_rotation"], infra_cam["sensor2lidar_translation"])
    T_il2ie = _make_T(infra["lidar2ego_rotation"], infra["lidar2ego_translation"])
    T_ie2g = _make_T(infra["ego2global_rotation"], infra["ego2global_translation"])
    T_g2ee = np.linalg.inv(_make_T(info["ego2global_rotation"], info["ego2global_translation"]))
    T_ee2el = np.linalg.inv(_make_T(info["lidar2ego_rotation"], info["lidar2ego_translation"]))
    T = T_ee2el @ T_g2ee @ T_ie2g @ T_il2ie @ T_s2il
    return T[:3, :3], T[:3, 3]


def _load_cameras(info: Dict, blob: Path) -> Dict[str, Camera]:
    """Load the vehicle + infrastructure cameras for the current frame of one token."""
    cameras: Dict[str, Camera] = {}
    for pkl_name, ns_name in _VEH_CAM_MAPPING.items():
        ci = info.get("cams", {}).get(pkl_name)
        if ci is None:
            continue
        cam = _build_camera(ci, blob)
        if cam is not None:
            cameras[ns_name] = cam

    infra = info.get("other_agent_info_dict", {}).get("model_other_agent_inf")
    if infra and infra.get("cams"):
        for pkl_name, ns_name in _INFRA_CAM_MAPPING.items():
            ci = infra["cams"].get(pkl_name)
            if ci is None:
                continue
            ci = dict(ci)
            try:
                R, t = _compute_infra_sensor_to_ego_lidar(info, ci, infra)
            except Exception:
                continue
            ci["sensor2lidar_rotation"] = R
            ci["sensor2lidar_translation"] = t
            cam = _build_camera(ci, blob)
            if cam is not None:
                cameras[ns_name] = cam
    return cameras


def _global_to_ego_xy(x: np.ndarray, y: np.ndarray, ego_t: np.ndarray, ego_h: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate/translate global (x, y) into the ego (== LiDAR) frame of the current pose."""
    dx, dy = x - ego_t[0], y - ego_t[1]
    cos_h, sin_h = np.cos(ego_h), np.sin(ego_h)
    return dx * cos_h + dy * sin_h, -dx * sin_h + dy * cos_h


def _annotations_global_to_ego(boxes9: np.ndarray, ego_t: np.ndarray, ego_h: float, names: List[str]) -> Annotations:
    """Convert 9-dim global boxes [x,y,z,l,w,h,vx,vy,heading] -> 7-dim ego boxes for camera.py."""
    boxes9 = np.asarray(boxes9, dtype=np.float64).reshape(-1, 9)
    xe, ye = _global_to_ego_xy(boxes9[:, 0], boxes9[:, 1], ego_t, ego_h)
    boxes7 = np.zeros((len(boxes9), 7), dtype=np.float64)
    boxes7[:, 0] = xe
    boxes7[:, 1] = ye
    boxes7[:, 2] = boxes9[:, 2] - ego_t[2]
    boxes7[:, 3:6] = boxes9[:, 3:6]  # l, w, h
    boxes7[:, 6] = boxes9[:, 8] - ego_h
    n = len(boxes7)
    return Annotations(
        boxes=boxes7,
        names=np.asarray(names, dtype=object),  # camera.py indexes names with a boolean mask
        velocity_3d=np.zeros((n, 3), dtype=np.float64),
        instance_tokens=[""] * n,
        track_tokens=[""] * n,
        is_v2xreal=True,
    )


def _ego_to_camera(points_ego: np.ndarray, camera: Camera) -> np.ndarray:
    """Transform Nx3 points in ego (LiDAR) frame to the camera frame (same as camera.py)."""
    s2l_r = camera.sensor2lidar_rotation
    s2l_t = camera.sensor2lidar_translation
    lidar2cam_r = np.linalg.inv(s2l_r)
    lidar2cam_t = s2l_t @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    pc = np.concatenate([points_ego, np.ones((len(points_ego), 1))], axis=-1)
    return (lidar2cam_rt.T @ pc.T).T[:, :3]


def _collect_lane_polylines(
    map_api: V2XRealMapWrapper, ego_t: np.ndarray, ego_h: float, road_z_ego: float, radius_m: float
) -> List[np.ndarray]:
    """Return nearby lane centerlines as Nx3 polylines in the ego frame (z on the road plane)."""
    polylines: List[np.ndarray] = []
    for lane in map_api._get_lanes():
        states = lane.baseline_path.discrete_path
        if states is None or len(states) < 2:
            continue
        gx = np.array([s.x for s in states], dtype=np.float64)
        gy = np.array([s.y for s in states], dtype=np.float64)
        ex, ey = _global_to_ego_xy(gx, gy, ego_t, ego_h)
        keep = (np.abs(ex) < radius_m) & (np.abs(ey) < radius_m)
        if keep.sum() < 2:
            continue
        poly = np.column_stack([ex, ey, np.full(len(ex), road_z_ego)])
        polylines.append(poly)
    return polylines


def _draw_lanes_on_ax(
    ax: plt.Axes, camera: Camera, lane_polylines: List[np.ndarray]
) -> Tuple[int, int]:
    """Project ego-frame lane polylines onto the camera image and draw them. Returns (in_fov, total)."""
    img_h, img_w = camera.image.shape[:2]
    in_fov, total = 0, 0
    for poly in lane_polylines:
        pc_cam = _ego_to_camera(poly, camera)
        uv, fov = _transform_points_to_image(pc_cam, camera.intrinsics, image_shape=(img_h, img_w))
        total += len(fov)
        in_fov += int(fov.sum())
        if fov.sum() < 2:
            continue
        # Plot only consecutive in-FOV segments so we do not draw lines across wraps.
        seg_u: List[float] = []
        seg_v: List[float] = []
        for k in range(len(fov)):
            if fov[k]:
                seg_u.append(uv[k, 0])
                seg_v.append(uv[k, 1])
            else:
                if len(seg_u) >= 2:
                    ax.plot(seg_u, seg_v, color="deepskyblue", linewidth=2.0, alpha=0.9)
                seg_u, seg_v = [], []
        if len(seg_u) >= 2:
            ax.plot(seg_u, seg_v, color="deepskyblue", linewidth=2.0, alpha=0.9)
    return in_fov, total


def _road_z_ego(boxes9: np.ndarray, ego_t: np.ndarray) -> float:
    """Estimate the road-plane height in the ego(==LiDAR) frame from GT box bottoms.

    The ego LiDAR sits ~2 m above the road (``lidar2ego_translation`` is ~0 here, so the
    ego frame *is* the LiDAR frame), and the front camera is at z=-0.6 m. Lanes must be
    drawn on the ground, i.e. well *below* the camera; a high percentile of the scattered
    box bottoms lands near camera height and makes the lanes float on the horizon. We take
    a LOW percentile (the ground-level vehicles touching the road) and clamp it to the
    physically plausible road band, falling back to a sensible LiDAR-above-ground height.
    """
    boxes9 = np.asarray(boxes9, dtype=np.float64).reshape(-1, 9)
    if len(boxes9) == 0:
        return -2.1  # sensible default road height below the ego LiDAR
    z_center_ego = boxes9[:, 2] - ego_t[2]
    bottoms = z_center_ego - boxes9[:, 5] / 2.0
    return float(np.clip(np.percentile(bottoms, 20), -2.6, -1.6))


def main() -> None:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", type=str, default=None, help="scene token (default: first)")
    parser.add_argument("--pkl", type=str, default=defaults["pkl"])
    parser.add_argument("--map-root", type=str, default=defaults["map_root"])
    parser.add_argument("--sensor-blob", type=str, default=defaults["sensor_blob"])
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(_REPO_ROOT / "exp" / "viz" / "viz3_rgb_projection.png"),
    )
    parser.add_argument("--radius", type=float, default=60.0, help="lane crop radius (m) in ego frame")
    args = parser.parse_args()

    pkl_path = Path(args.pkl)
    blob = Path(args.sensor_blob)
    map_root = Path(args.map_root)

    # Raw PKL keyed by pkl token (carries infra info that the converted scene drops).
    raw_infos = {it["token"]: it for it in pickle.load(open(pkl_path, "rb"))["infos"]}

    scene_filter = SceneFilter(num_history_frames=_NUM_HISTORY_FRAMES, num_future_frames=20, frame_interval=1)
    loader = SceneLoaderV2XReal(
        pkl_path=pkl_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
        sensor_blob_path=blob,
        map_root=map_root,
    )
    token = args.token if args.token is not None else loader.tokens[0]
    print(f"[viz_rgb_projection] token: {token}")

    scene = loader.get_scene_from_token(token)
    current_dict = loader.scene_frames_dicts[token][_NUM_HISTORY_FRAMES - 1]
    pkl_token = current_dict["token"]
    info = raw_infos.get(pkl_token)
    if info is None:
        raise KeyError(f"Token {pkl_token} not found in raw PKL infos")

    ego_t = np.asarray(current_dict["ego2global_translation"], dtype=np.float64)
    ego_h = Quaternion(*current_dict["ego2global_rotation"]).yaw_pitch_roll[0]

    cameras = _load_cameras(info, blob)
    print(f"[viz_rgb_projection] loaded cameras: {sorted(cameras.keys())}")

    annotations = scene.frames[_NUM_HISTORY_FRAMES - 1].annotations
    boxes9 = np.asarray(annotations.boxes, dtype=np.float64)
    road_z = _road_z_ego(boxes9, ego_t)
    print(f"[viz_rgb_projection] estimated road plane z (ego frame): {road_z:.2f} m")
    ann_ego = _annotations_global_to_ego(boxes9, ego_t, ego_h, list(annotations.names))

    map_api = V2XRealMapWrapper(map_root=map_root)
    lane_polylines = _collect_lane_polylines(map_api, ego_t, ego_h, road_z, args.radius)
    print(f"[viz_rgb_projection] nearby lane polylines: {len(lane_polylines)}")

    n_cols = 3
    n_rows = (len(_GRID_LAYOUT) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, (ns_name, title) in zip(axes, _GRID_LAYOUT):
        camera = cameras.get(ns_name)
        if camera is None or camera.image is None:
            ax.set_title(f"{title}\n(unavailable)", fontsize=10)
            continue
        add_annotations_to_camera_ax(ax, camera, ann_ego)
        lane_in_fov, lane_total = _draw_lanes_on_ax(ax, camera, lane_polylines)
        frac = lane_in_fov / lane_total if lane_total else 0.0
        ax.set_title(f"{title}  (lane pts in-FOV: {lane_in_fov}/{lane_total} = {frac:.2f})", fontsize=10)
        ax.set_xlim(0, camera.image.shape[1])
        ax.set_ylim(camera.image.shape[0], 0)
        print(f"[viz_rgb_projection] {ns_name}: lane pts in-FOV {lane_in_fov}/{lane_total} ({frac:.2f})")

    fig.suptitle(f"V2X-Real GT boxes + map lanes  |  token: {token}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"[viz_rgb_projection] saved {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
