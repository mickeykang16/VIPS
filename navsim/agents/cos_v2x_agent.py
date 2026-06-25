"""
CoS-V2X agent (cooperative vehicle-infrastructure planning, built on SparseDrive).

Reads all of its input from the NAVSIM AgentInput (camera images, calibration, and
ego history) — no PKL call-stack hacks or external ego-status lookups.

Selected in the eval via AGENT=cos_v2x (alias: sparsedrive_navsim), or
AGENT_CLASS_PATH=navsim.agents.cos_v2x_agent.CoSV2XAgent. The model checkout,
config, checkpoint and eval mode all come from configs/eval/cos_v2x.py
(COS_V2X_FOLDER / COS_V2X_CONFIG_PATH / COS_V2X_MODEL_CHECKPOINT_PATH /
COS_V2X_MODE); the eval entry bridges them into the environment for the worker
processes. Legacy SPARSEDRIVE_* keys are still accepted.
"""
import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory, Scene
import os
from pathlib import Path

from navsim.common.vips_config import get_first

## NOTE: CoS-V2X model checkout ==================================== #
# Resolved from configs/eval/cos_v2x.py (COS_V2X_FOLDER; the legacy
# SPARSEDRIVE_FOLDER key is still accepted), bridged into the environment by the
# eval entry. The checkout must be on sys.path before the SparseDrive plugin
# modules are imported below.
COS_V2X_FOLDER = get_first("COS_V2X_FOLDER", "SPARSEDRIVE_FOLDER")
if COS_V2X_FOLDER is None:
    raise EnvironmentError(
        "[CoS-V2X] COS_V2X_FOLDER is not set. Set it in configs/eval/cos_v2x.py "
        "(the model checkout, e.g. models/CoS-V2X)."
    )
import sys
sys.path.append(COS_V2X_FOLDER)

import mmcv
from os import path as osp
import copy
import torch
import warnings
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet.models import build_detector
## ================================================================ ##

_CURRENT_FRAME_IDX = 3


class CoSV2XAgent(AbstractAgent):
    """CoS-V2X agent (cooperative V2X planning, built on SparseDrive) that reads
    ALL data from AgentInput only.

    All data is sourced from AgentInput (no PKL call-stack hacks or external
    ego-status lookups):
      - images:      agent_input.cameras[3].cam_*.image (RGB uint8)
      - calibration: agent_input.cameras[3].cam_*.sensor2lidar_*, intrinsics
      - ego2global: agent_input.ego2global_translation, ego2global_rotation
      - lidar2ego: agent_input.lidar2ego_rotation, lidar2ego_translation
      - timestamp: agent_input.timestamp
      - scene_token: agent_input.scene_token
      - ego_status: agent_input.ego_statuses[3].ego_velocity/acceleration/yaw_rate
      - driving_command: agent_input.ego_statuses[3].driving_command
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=4, interval_length=0.5
        ),
        cfg_path=None,
        model_checkpoint=None,
    ):
        super().__init__(trajectory_sampling)

        # Explicit constructor args win; otherwise resolve from configs/eval/cos_v2x.py
        # (COS_V2X_* keys; legacy SPARSEDRIVE_* still accepted).
        model_folder = COS_V2X_FOLDER
        cfg_path = cfg_path or get_first("COS_V2X_CONFIG_PATH", "SPARSEDRIVE_CONFIG_PATH")
        model_checkpoint = model_checkpoint or get_first(
            "COS_V2X_MODEL_CHECKPOINT_PATH", "SPARSEDRIVE_MODEL_CHECKPOINT_PATH"
        )

        if cfg_path is None:
            raise EnvironmentError(
                "[CoS-V2X] COS_V2X_CONFIG_PATH is not set (configs/eval/cos_v2x.py)."
            )
        if model_checkpoint is None:
            raise EnvironmentError(
                "[CoS-V2X] COS_V2X_MODEL_CHECKPOINT_PATH is not set (configs/eval/cos_v2x.py)."
            )
        print(f"[CoS-V2X] FOLDER={model_folder}  cfg={cfg_path}  ckpt={model_checkpoint}")

        cfg = Config.fromfile(cfg_path)

        # ── Rewrite kmeans paths (data/kmeans/ and HiP-AD/data/kmeans/) ──
        def replace_prefix_in_cfg(obj, old_prefix: str, new_prefix: str):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    obj[k] = replace_prefix_in_cfg(v, old_prefix, new_prefix)
                return obj
            elif isinstance(obj, list):
                return [replace_prefix_in_cfg(v, old_prefix, new_prefix) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(replace_prefix_in_cfg(v, old_prefix, new_prefix) for v in obj)
            elif isinstance(obj, str):
                if obj.startswith(old_prefix):
                    return new_prefix + obj[len(old_prefix):]
                return obj
            else:
                return obj

        kmeans_abs = str((Path(model_folder) / "data/kmeans").resolve()) + "/"
        cfg.kmeans_folder = kmeans_abs
        cfg._cfg_dict = replace_prefix_in_cfg(cfg._cfg_dict, "data/kmeans/", kmeans_abs)

        hipad_kmeans_abs = str((Path(model_folder) / "HiP-AD/data/kmeans").resolve()) + "/"
        cfg._cfg_dict = replace_prefix_in_cfg(cfg._cfg_dict, "HiP-AD/data/kmeans/", hipad_kmeans_abs)

        # ── Load plugin ──────────────────────────────────────────────
        if hasattr(cfg, "plugin") and cfg.plugin:
            import importlib
            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir).split("/")
            else:
                _module_dir = os.path.dirname(cfg_path).split("/")
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            print(_module_path)
            importlib.import_module(_module_path)

        # ── Model config ──────────────────────────────────────────────
        cfg.model.pretrained = None
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
            if samples_per_gpu > 1:
                cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            samples_per_gpu = max(ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test)
            if samples_per_gpu > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        set_random_seed(0, deterministic=True)

        _cfg_basename = osp.splitext(osp.basename(cfg_path))[0]
        cfg.work_dir = osp.join(model_folder, "work_dirs", _cfg_basename)
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        cfg.data.test.work_dir = cfg.work_dir

        # ── Build & load model ────────────────────────────────────────
        cfg.model.train_cfg = None
        self.model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.model)
        checkpoint = load_checkpoint(self.model, model_checkpoint, map_location="cpu")

        if "CLASSES" in checkpoint.get("meta", {}):
            self.model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            raise ValueError("[CoS-V2X] checkpoint has no CLASSES metadata.")

        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("[CoS-V2X] model moved to CUDA")
        else:
            self.model = self.model.float()
            print("[CoS-V2X] CUDA not available, model cast to float32")

        # ── Stage / offset info (set by the eval script) ─────────────
        self.current_stage = None
        self.current_offset = None
        self.current_ego_global_translation = None

        # ── Temporal state (instance bank save/restore) ─────────────
        self._saved_temporal_state = None

        print("[CoS-V2X] agent initialized.")

    # ================================================================ #
    #  Basic interface                                                  #
    # ================================================================ #

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        pass

    # ================================================================ #
    #  Instance Bank temporal state save / restore                      #
    # ================================================================ #

    @staticmethod
    def _clone_or_none(x):
        return x.clone() if x is not None else None

    @staticmethod
    def _save_instance_bank(ib):
        S = CoSV2XAgent
        return {
            "cached_feature": S._clone_or_none(ib.cached_feature),
            "cached_anchor": S._clone_or_none(ib.cached_anchor),
            "metas": copy.deepcopy(ib.metas),
            "mask": S._clone_or_none(ib.mask),
            "confidence": S._clone_or_none(ib.confidence),
            "temp_confidence": S._clone_or_none(ib.temp_confidence),
            "instance_id": S._clone_or_none(ib.instance_id),
            "prev_id": ib.prev_id,
        }

    @staticmethod
    def _restore_instance_bank(ib, state):
        S = CoSV2XAgent
        ib.cached_feature = S._clone_or_none(state["cached_feature"])
        ib.cached_anchor = S._clone_or_none(state["cached_anchor"])
        ib.metas = copy.deepcopy(state["metas"])
        ib.mask = S._clone_or_none(state["mask"])
        ib.confidence = S._clone_or_none(state["confidence"])
        ib.temp_confidence = S._clone_or_none(state["temp_confidence"])
        ib.instance_id = S._clone_or_none(state["instance_id"])
        ib.prev_id = state["prev_id"]

    @staticmethod
    def _save_instance_queue(iq):
        S = CoSV2XAgent
        return {
            "metas": copy.deepcopy(iq.metas),
            "prev_instance_id": S._clone_or_none(iq.prev_instance_id),
            "prev_confidence": S._clone_or_none(iq.prev_confidence),
            "period": S._clone_or_none(iq.period),
            "instance_feature_queue": [t.clone() for t in iq.instance_feature_queue],
            "anchor_queue": [t.clone() for t in iq.anchor_queue],
            "prev_ego_status": S._clone_or_none(iq.prev_ego_status),
            "ego_period": S._clone_or_none(iq.ego_period),
            "ego_feature_queue": [t.clone() for t in iq.ego_feature_queue],
            "ego_anchor_queue": [t.clone() for t in iq.ego_anchor_queue],
        }

    @staticmethod
    def _restore_instance_queue(iq, state):
        S = CoSV2XAgent
        iq.metas = copy.deepcopy(state["metas"])
        iq.prev_instance_id = S._clone_or_none(state["prev_instance_id"])
        iq.prev_confidence = S._clone_or_none(state["prev_confidence"])
        iq.period = S._clone_or_none(state["period"])
        iq.instance_feature_queue = [t.clone() for t in state["instance_feature_queue"]]
        iq.anchor_queue = [t.clone() for t in state["anchor_queue"]]
        iq.prev_ego_status = S._clone_or_none(state["prev_ego_status"])
        iq.ego_period = S._clone_or_none(state["ego_period"])
        iq.ego_feature_queue = [t.clone() for t in state["ego_feature_queue"]]
        iq.ego_anchor_queue = [t.clone() for t in state["ego_anchor_queue"]]

    def _save_temporal_state(self):
        head = self.model.head
        state = {}
        if hasattr(head, "veh_det_head") and hasattr(head.veh_det_head, "instance_bank"):
            state["veh_det_ib"] = self._save_instance_bank(head.veh_det_head.instance_bank)
        if hasattr(head, "infra_det_head") and hasattr(head.infra_det_head, "instance_bank"):
            state["infra_det_ib"] = self._save_instance_bank(head.infra_det_head.instance_bank)
        if hasattr(head, "det_head") and hasattr(head.det_head, "instance_bank"):
            state["det_ib"] = self._save_instance_bank(head.det_head.instance_bank)
        if hasattr(head, "map_head") and hasattr(head.map_head, "instance_bank"):
            state["map_ib"] = self._save_instance_bank(head.map_head.instance_bank)
        if hasattr(head, "motion_plan_head") and hasattr(head.motion_plan_head, "instance_queue"):
            state["motion_iq"] = self._save_instance_queue(head.motion_plan_head.instance_queue)
        self._saved_temporal_state = state

    def _restore_temporal_state(self):
        if self._saved_temporal_state is None:
            print("[CoS-V2X] WARNING: No saved temporal state to restore")
            return
        head = self.model.head
        state = self._saved_temporal_state
        if "veh_det_ib" in state and hasattr(head, "veh_det_head"):
            self._restore_instance_bank(head.veh_det_head.instance_bank, state["veh_det_ib"])
        if "infra_det_ib" in state and hasattr(head, "infra_det_head"):
            self._restore_instance_bank(head.infra_det_head.instance_bank, state["infra_det_ib"])
        if "det_ib" in state and hasattr(head, "det_head"):
            self._restore_instance_bank(head.det_head.instance_bank, state["det_ib"])
        if "map_ib" in state and hasattr(head, "map_head"):
            self._restore_instance_bank(head.map_head.instance_bank, state["map_ib"])
        if "motion_iq" in state and hasattr(head, "motion_plan_head") and hasattr(head.motion_plan_head, "instance_queue"):
            self._restore_instance_queue(head.motion_plan_head.instance_queue, state["motion_iq"])

    # ================================================================ #
    #  Sensor config                                                    #
    # ================================================================ #

    def get_sensor_config(self) -> SensorConfig:
        idx = [_CURRENT_FRAME_IDX]
        return SensorConfig(
            cam_f0=idx,
            cam_l0=idx,
            cam_l1=False,
            cam_l2=False,
            cam_r0=idx,
            cam_r1=False,
            cam_r2=False,
            cam_b0=idx,
            lidar_pc=False,
            cam_infra0=idx,
            cam_infra1=idx,
        )

    # ================================================================ #
    #  compute_trajectory  — ALL data from agent_input                  #
    # ================================================================ #

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        import math
        from pyquaternion import Quaternion as _Quat
        from PIL import Image as _PIL_Image

        # ── Config parameters ────────────────────────────────────────
        _IMG_NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        _IMG_NORM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        _H_ORIG, _W_ORIG = 900, 1600
        _FH, _FW = 256, 704
        _NAVSIM_PLANNING_STEPS = self._trajectory_sampling.num_poses

        # ── 1. Extract cameras (from agent_input only) ───────────────
        cameras = agent_input.cameras[_CURRENT_FRAME_IDX] if agent_input.cameras else None
        assert cameras is not None, "CoSV2XAgent requires camera images."

        _mode = (get_first("COS_V2X_MODE", "SPARSEDRIVE_MODE", default="")).lower()
        if _mode == "veh":
            is_cooperative = False
        elif _mode == "coop":
            is_cooperative = True
        else:
            is_cooperative = (
                cameras.cam_infra0 is not None
                and cameras.cam_infra0.image is not None
            )

        if is_cooperative:
            cam_list = [
                cameras.cam_f0, cameras.cam_l0, cameras.cam_r0, cameras.cam_b0,
                cameras.cam_infra0, cameras.cam_infra1,
            ]
        else:
            cam_list = [
                cameras.cam_f0, cameras.cam_l0, cameras.cam_r0, cameras.cam_b0,
            ]

        # ── 2. Load images + calibration (always from agent_input) ───
        imgs_bgr = []
        lidar2img_list = []
        cam_intrinsic_list = []

        for cam_idx, cam in enumerate(cam_list):
            # image: always from agent_input
            assert cam.image is not None, (
                f"CoSV2XAgent: camera image is None (cam_idx={cam_idx})"
            )
            img_bgr = cam.image[:, :, ::-1].copy()  # RGB→BGR uint8

            # calibration: from agent_input
            s2l_r = cam.sensor2lidar_rotation
            s2l_t = cam.sensor2lidar_translation
            K = np.array(cam.intrinsics, dtype=np.float64)

            if s2l_r is not None and s2l_t is not None:
                lidar2cam_r = np.linalg.inv(s2l_r)
                lidar2cam_t = s2l_t @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4, dtype=np.float64)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                viewpad = np.eye(4, dtype=np.float64)
                viewpad[: K.shape[0], : K.shape[1]] = K
                lidar2img_rt = viewpad @ lidar2cam_rt.T
            else:
                lidar2img_rt = np.eye(4, dtype=np.float64)
                K = np.eye(3, dtype=np.float64)

            imgs_bgr.append(img_bgr)
            lidar2img_list.append(lidar2img_rt)
            cam_intrinsic_list.append(K.copy())

        # ── 3. Resize / Crop ─────────────────────────────────────────
        _resize = max(_FH / _H_ORIG, _FW / _W_ORIG)
        _newW = int(_W_ORIG * _resize)
        _newH = int(_H_ORIG * _resize)
        _crop_h = int((1 - 0.0) * _newH) - _FH
        _crop_w = int(max(0, _newW - _FW) / 2)
        _crop = (_crop_w, _crop_h, _crop_w + _FW, _crop_h + _FH)

        _aug_mat_3x3 = np.eye(3, dtype=np.float64)
        _aug_mat_3x3[:2, :2] *= _resize
        _aug_mat_3x3[:2, 2] -= np.array(_crop[:2], dtype=np.float64)
        _aug_mat = np.eye(4, dtype=np.float64)
        _aug_mat[:3, :3] = _aug_mat_3x3

        imgs_f32 = []
        for cam_idx in range(len(cam_list)):
            _actual_H, _actual_W = imgs_bgr[cam_idx].shape[:2]
            _actual_resize_dims = (int(_actual_W * _resize), int(_actual_H * _resize))
            img_pil = _PIL_Image.fromarray(np.uint8(imgs_bgr[cam_idx]))
            img_pil = img_pil.resize(_actual_resize_dims).crop(_crop)
            imgs_f32.append(np.array(img_pil).astype(np.float32))

            lidar2img_list[cam_idx] = _aug_mat @ lidar2img_list[cam_idx]
            cam_intrinsic_list[cam_idx] = cam_intrinsic_list[cam_idx] * _resize

        # ── 4. Normalize ─────────────────────────────────────────────
        imgs_norm = [
            mmcv.imnormalize(img.copy(), _IMG_NORM_MEAN, _IMG_NORM_STD, to_rgb=True)
            for img in imgs_f32
        ]

        # ── 5. Adaptor: projection_mat, image_wh, focal, T_global ───
        projection_mat = np.float32(np.stack(lidar2img_list))
        img_shapes = [img.shape[:2] for img in imgs_norm]
        image_wh_np = np.ascontiguousarray(
            np.array(img_shapes, dtype=np.float32)[:, ::-1]
        )
        cam_intrinsic_np = np.float32(np.stack(cam_intrinsic_list))
        focal = cam_intrinsic_np[:, 0, 0]

        # ── Build T_global (from agent_input) ────────────────────────
        _e2g_t = agent_input.ego2global_translation
        _e2g_r = agent_input.ego2global_rotation
        _l2e_r = agent_input.lidar2ego_rotation
        _l2e_t = agent_input.lidar2ego_translation

        if _e2g_t is not None and _e2g_r is not None:
            # stage2: offset-adjusted translation (set by the eval script)
            if self.current_stage == "s2" and self.current_ego_global_translation is not None:
                e2g_t = np.array(self.current_ego_global_translation, dtype=np.float64)
            else:
                e2g_t = np.array(_e2g_t, dtype=np.float64)
            e2g_q = _Quat(_e2g_r)
            ego2global = np.eye(4, dtype=np.float64)
            ego2global[:3, :3] = e2g_q.rotation_matrix
            ego2global[:3, 3] = e2g_t

            if _l2e_r is not None and _l2e_t is not None:
                lidar2ego = np.eye(4, dtype=np.float64)
                lidar2ego[:3, :3] = _Quat(_l2e_r).rotation_matrix
                lidar2ego[:3, 3] = np.array(_l2e_t, dtype=np.float64)
            else:
                lidar2ego = np.eye(4, dtype=np.float64)
            T_global = (ego2global @ lidar2ego).astype(np.float32)
        else:
            T_global = np.eye(4, dtype=np.float32)

        T_global_inv = np.linalg.inv(T_global).astype(np.float32)

        # ── timestamp / scene_token (from agent_input) ───────────────
        _timestamp = float(agent_input.timestamp) / 1e6 if agent_input.timestamp is not None else 0.0
        _scene_token = agent_input.scene_token if agent_input.scene_token is not None else "navsim_scene"

        # ── 6. driving command (from agent_input) ────────────────────
        _NAV_TO_SD = {0: 2, 1: 1, 2: 0}
        _raw_dc = agent_input.ego_statuses[_CURRENT_FRAME_IDX].driving_command
        if hasattr(_raw_dc, "__len__"):
            dc_nav = int(np.argmax(_raw_dc)) if len(_raw_dc) > 1 else int(_raw_dc[0])
        else:
            dc_nav = int(_raw_dc)
        dc_sd = _NAV_TO_SD.get(dc_nav, 2)
        gt_ego_fut_cmd = np.zeros(3, dtype=np.float32)
        gt_ego_fut_cmd[dc_sd] = 1.0

        # ── 7. ego_status (10-dim, always from agent_input) ──────────
        _es = agent_input.ego_statuses[_CURRENT_FRAME_IDX]
        _ev = np.array(_es.ego_velocity, dtype=np.float32)
        _ea = np.array(_es.ego_acceleration, dtype=np.float32)
        _wz = float(_es.ego_yaw_rate)

        ego_status = np.array(
            [
                _ea[0], _ea[1], 0.0,    # accel [ax, ay, az]
                0.0, 0.0, _wz,          # rot_rate [wx, wy, wz]
                _ev[0], _ev[1], 0.0,    # vel [vx, vy, vz]
                0.0,                     # steering
            ],
            dtype=np.float32,
        )

        # ── 8. img tensor ────────────────────────────────────────────
        imgs_chw = np.ascontiguousarray(
            np.stack([img.transpose(2, 0, 1) for img in imgs_norm], axis=0)
        )
        img_tensor = torch.from_numpy(imgs_chw).float().unsqueeze(0)

        device = next(self.model.parameters()).device

        # ── 9. data dict ─────────────────────────────────────────────
        img_metas_entry = dict(
            T_global=T_global.astype(np.float64),
            T_global_inv=T_global_inv.astype(np.float64),
            timestamp=torch.tensor(_timestamp, dtype=torch.float64).to(device),
            scene_token=_scene_token,
        )

        data = dict(
            projection_mat=torch.from_numpy(projection_mat).float().to(device).unsqueeze(0),
            image_wh=torch.from_numpy(image_wh_np).float().to(device).unsqueeze(0),
            T_global=torch.from_numpy(T_global).float().to(device).unsqueeze(0),
            T_global_inv=torch.from_numpy(T_global_inv).float().to(device).unsqueeze(0),
            timestamp=torch.tensor([_timestamp], dtype=torch.float64).to(device),
            scene_token=[_scene_token],
            gt_ego_fut_cmd=torch.from_numpy(gt_ego_fut_cmd).float().to(device).unsqueeze(0),
            ego_status=torch.from_numpy(ego_status).float().to(device).unsqueeze(0),
            img_metas=[img_metas_entry],
            focal=torch.from_numpy(focal).float().to(device).unsqueeze(0),
            img=img_tensor.to(device),
        )

        # ── 10. Instance bank restore / save ─────────────────────────
        self._restore_temporal_state()

        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)

        if self.current_stage == "s1":
            self._save_temporal_state()

        # ── 11. Extract trajectory ───────────────────────────────────
        final_planning = result[0]["img_bbox"]["final_planning"]
        pred_xy = final_planning[:_NAVSIM_PLANNING_STEPS, :2].cpu().numpy()

        headings = np.zeros(_NAVSIM_PLANNING_STEPS, dtype=np.float32)
        for i in range(_NAVSIM_PLANNING_STEPS):
            if i == 0:
                dx, dy = float(pred_xy[i, 0]), float(pred_xy[i, 1])
            else:
                dx = float(pred_xy[i, 0] - pred_xy[i - 1, 0])
                dy = float(pred_xy[i, 1] - pred_xy[i - 1, 1])
            if abs(dx) > 0.05 or abs(dy) > 0.05:
                headings[i] = math.atan2(dy, dx)
            elif i > 0:
                headings[i] = headings[i - 1]

        poses = np.column_stack([pred_xy, headings]).astype(np.float32)
        return Trajectory(poses, self._trajectory_sampling)
