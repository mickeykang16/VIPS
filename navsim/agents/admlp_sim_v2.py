"""
Standalone ADMLP agent variant (v2) that:
  - Does NOT extend navsim.agents.admlp.ADMLP
  - Loads its own checkpoint + normalization stats (see ADMLP_V2_CKPT_PATH)
  - Computes velocity/acceleration via finite difference from ego_statuses poses
    (similar to sparse_converter_w_map_parallel.py)
  - Uses driving_command from EgoStatus for cmd
  - Keeps history computation from ego_statuses poses

Feature vector layout:
  [ego_lcf_feat(6), cmd(3), flattened_history(H x 3)]
"""

import math
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.common.vips_config import get


def _global_to_ego_xy(global_xy: np.ndarray, cur_xy: np.ndarray, cur_yaw: float) -> np.ndarray:
    """Convert global xy to ego-local xy at current pose."""
    dx = float(global_xy[0] - cur_xy[0])
    dy = float(global_xy[1] - cur_xy[1])
    c = math.cos(cur_yaw)
    s = math.sin(cur_yaw)
    ego_x = c * dx + s * dy
    ego_y = -s * dx + c * dy
    return np.array([ego_x, ego_y], dtype=np.float32)


class _ADMLPPlanNet(torch.nn.Module):
    """Simple ADMLP head architecture."""

    def __init__(self, input_dim: int, hidden_dim: int, future_steps: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.future_steps = int(future_steps)
        self.plan_head = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, self.future_steps * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.plan_head(x)
        return out.view(x.shape[0], self.future_steps, 2)


class ADMLPSim(AbstractAgent):
    """
    Standalone ADMLP agent that builds features from ego_statuses.

    Velocity and acceleration are computed via finite difference from poses,
    similar to sparse_converter_w_map_parallel.py:get_ego_status_no_canbus().
    """

    requires_scene = False
    wants_ego_local = True  # ego_statuses should be in ego-local frame

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        super().__init__(trajectory_sampling)

        # Checkpoint + normalization stats. Set ADMLP_V2_CKPT_PATH / ADMLP_V2_STATS_PATH
        # in configs/eval/admlp.py to point at your trained ADMLP weights; the
        # repo-relative defaults below are used as a fallback when those are unset.
        navsim_root = get("NAVSIM_DEVKIT_ROOT") or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        default_ckpt = os.path.join(navsim_root, "checkpoints/admlp/admlp_univ2x.pt")
        default_stats = os.path.join(navsim_root, "checkpoints/admlp/admlp_univ2x_train_stats.pkl")

        self._checkpoint_path = get("ADMLP_V2_CKPT_PATH", default_ckpt)
        self._stats_path = get("ADMLP_V2_STATS_PATH", default_stats)

        # ----- Configuration -----
        # Fixed for the released ADMLP checkpoint; the checkpoint's own args
        # override history_steps / normalize_input at load time (see below).
        self._history_steps_cfg = 2
        self._frame_dt_sec = 0.5  # assumed dt between frames
        self._normalize_input_cfg = True

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ----- Load model -----
        self._model, self._future_steps, self._input_dim, ckpt_args = self._load_model_from_checkpoint()
        self._history_steps = int(ckpt_args.get("history_steps", self._history_steps_cfg))

        ckpt_normalize = bool(ckpt_args.get("normalize_input", self._normalize_input_cfg))
        self._normalize_input = self._normalize_input_cfg and ckpt_normalize
        self._normalizer = self._load_normalizer()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model_from_checkpoint(self) -> Tuple[_ADMLPPlanNet, int, int, Dict[str, Any]]:
        if not os.path.exists(self._checkpoint_path):
            raise FileNotFoundError(f"ADMLP checkpoint not found: {self._checkpoint_path}")

        ckpt = torch.load(self._checkpoint_path, map_location="cpu")
        ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        if not isinstance(state_dict, dict):
            raise RuntimeError("Invalid ADMLP checkpoint format: no state_dict found.")

        # Infer architecture from weights
        w0 = state_dict.get("plan_head.0.weight")
        w_last = state_dict.get("plan_head.4.weight")
        if w0 is None or w_last is None:
            raise RuntimeError("ADMLP checkpoint missing expected layers (plan_head.0/4).")

        hidden_dim = int(w0.shape[0])
        input_dim = int(w0.shape[1])
        future_steps = int(w_last.shape[0] // 2)

        model = _ADMLPPlanNet(input_dim=input_dim, hidden_dim=hidden_dim, future_steps=future_steps)
        model.load_state_dict(state_dict, strict=True)
        model.to(self._device)
        model.eval()

        print(f"[ADMLPSim-v2] Loaded checkpoint: {self._checkpoint_path}")
        print(f"[ADMLPSim-v2] input_dim={input_dim}, hidden_dim={hidden_dim}, future_steps={future_steps}")

        return model, future_steps, input_dim, ckpt_args

    def _load_normalizer(self) -> Optional[Dict[str, np.ndarray]]:
        if not self._normalize_input:
            return None
        if not os.path.exists(self._stats_path):
            raise FileNotFoundError(f"ADMLP stats file not found: {self._stats_path}")
        with open(self._stats_path, "rb") as f:
            stats = pickle.load(f)
        print(f"[ADMLPSim-v2] Loaded stats: {self._stats_path}")
        return {
            "lcf_mean": np.asarray(stats["lcf_mean"], dtype=np.float32),
            "lcf_std": np.asarray(stats["lcf_std"], dtype=np.float32),
            "his_mean": np.asarray(stats["his_mean"], dtype=np.float32),
            "his_std": np.asarray(stats["his_std"], dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # AbstractAgent interface
    # ------------------------------------------------------------------
    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        """No-op for compatibility."""

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        feat = self._build_feature_from_ego_statuses(agent_input)
        if feat is None:
            return self._constant_velocity_fallback(agent_input)
        pred_xy = self._predict_xy(feat)
        return self._xy_to_trajectory(pred_xy)

    # ------------------------------------------------------------------
    # Feature construction from ego_statuses
    # ------------------------------------------------------------------
    def _build_feature_from_ego_statuses(self, agent_input: AgentInput) -> Optional[np.ndarray]:
        statuses = agent_input.ego_statuses
        if not statuses or len(statuses) < 3:
            # Need at least 3 frames for velocity and acceleration computation
            return None

        cur = statuses[-1]
        prev1 = statuses[-2]
        prev2 = statuses[-3]

        # Get current pose info
        cur_xy = cur.ego_pose[:2].astype(np.float64)
        cur_yaw = float(cur.ego_pose[2])
        in_global = cur.in_global_frame

        # -----------------------------------------------------------------
        # 1) Compute velocity and acceleration via finite difference
        #    (similar to sparse_converter_w_map_parallel.py:get_ego_status_no_canbus)
        # -----------------------------------------------------------------
        dt = self._frame_dt_sec  # assumed fixed dt between frames

        if in_global:
            # Poses are in global frame - compute velocity in global then rotate to ego
            prev1_xy = prev1.ego_pose[:2].astype(np.float64)
            prev2_xy = prev2.ego_pose[:2].astype(np.float64)

            # Velocity in global frame: v = (p1 - p0) / dt
            v_global = (cur_xy - prev1_xy) / dt  # velocity at current time
            v_prev_global = (prev1_xy - prev2_xy) / dt  # velocity at prev time

            # Rotate to ego frame at current pose (global -> ego)
            c = math.cos(cur_yaw)
            s = math.sin(cur_yaw)
            # R_ge (global to ego) = [[c, s], [-s, c]]
            vx = c * v_global[0] + s * v_global[1]
            vy = -s * v_global[0] + c * v_global[1]

            vx_prev = c * v_prev_global[0] + s * v_prev_global[1]
            vy_prev = -s * v_prev_global[0] + c * v_prev_global[1]

            # Acceleration: a = (v_curr - v_prev) / dt
            ax = (vx - vx_prev) / dt
            ay = (vy - vy_prev) / dt
        else:
            # Poses are in ego-local frame (relative to current frame)
            # prev1.ego_pose gives the position of prev1 frame relative to current frame
            # This means prev1_xy is actually the displacement from current to prev1
            prev1_xy = prev1.ego_pose[:2].astype(np.float64)
            prev2_xy = prev2.ego_pose[:2].astype(np.float64)

            # Velocity: displacement per time
            # cur_xy should be [0, 0] in ego-local frame
            # velocity = (cur_xy - prev1_xy) / dt = -prev1_xy / dt
            vx = -prev1_xy[0] / dt
            vy = -prev1_xy[1] / dt

            # Previous velocity
            vx_prev = -(prev2_xy[0] - prev1_xy[0]) / dt
            vy_prev = -(prev2_xy[1] - prev1_xy[1]) / dt

            # Acceleration
            ax = (vx - vx_prev) / dt
            ay = (vy - vy_prev) / dt

        ego_lcf_feat = np.array(
            [vx, vy, 0.0, ax, ay, 0.0], dtype=np.float32,
        )

        # -----------------------------------------------------------------
        # 2) cmd: driving_command from EgoStatus -> one-hot [left, right, straight]
        # -----------------------------------------------------------------
        cmd_val = int(cur.driving_command[0]) if len(cur.driving_command) > 0 else 2
        # 0=left, 1=right, 2=straight (v2xreal convention)
        cmd = np.zeros(3, dtype=np.float32)
        if cmd_val == 0:
            cmd[0] = 1.0  # left
        elif cmd_val == 1:
            cmd[1] = 1.0  # right
        else:
            cmd[2] = 1.0  # straight

        # -----------------------------------------------------------------
        # 3) history trajectory in ego-local frame (same as admlp_sim.py)
        # -----------------------------------------------------------------
        n_available = len(statuses) - 1
        n_hist = min(self._history_steps, n_available)

        if n_hist < 1:
            return None

        hist_rel_list = []
        for k in range(n_hist, 0, -1):
            past = statuses[-(1 + k)]
            if in_global:
                rel_xy = _global_to_ego_xy(
                    past.ego_pose[:2].astype(np.float32), cur_xy.astype(np.float32), cur_yaw,
                )
            else:
                # Already ego-local (relative to current frame)
                rel_xy = past.ego_pose[:2].astype(np.float32)
            hist_rel_list.append(rel_xy)

        hist_rel_xy = np.stack(hist_rel_list, axis=0)  # [H, 2]
        hist_rel = np.concatenate(
            [hist_rel_xy, np.zeros((hist_rel_xy.shape[0], 1), dtype=np.float32)],
            axis=1,
        )  # [H, 3]

        # Pad history if fewer frames than _history_steps
        if n_hist < self._history_steps:
            pad = np.repeat(hist_rel[:1], self._history_steps - n_hist, axis=0)
            hist_rel = np.concatenate([pad, hist_rel], axis=0)

        # -----------------------------------------------------------------
        # 4) normalize
        # -----------------------------------------------------------------
        if self._normalizer is not None:
            ego_lcf_feat = (ego_lcf_feat - self._normalizer["lcf_mean"]) / self._normalizer["lcf_std"]
            hist_rel = (hist_rel - self._normalizer["his_mean"]) / self._normalizer["his_std"]

        # -----------------------------------------------------------------
        # 5) concat: [ego_lcf_feat, cmd, flattened_history]
        # -----------------------------------------------------------------
        feat = np.concatenate(
            [ego_lcf_feat, cmd, hist_rel.reshape(-1)], axis=0,
        ).astype(np.float32)

        return self._match_input_dim(feat)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _match_input_dim(self, feat_vec: np.ndarray) -> np.ndarray:
        cur_dim = int(feat_vec.shape[0])
        if cur_dim == self._input_dim:
            return feat_vec
        if cur_dim < self._input_dim:
            pad = np.zeros((self._input_dim - cur_dim,), dtype=np.float32)
            return np.concatenate([feat_vec, pad], axis=0)
        return feat_vec[: self._input_dim]

    def _predict_xy(self, feat: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(feat).to(self._device).unsqueeze(0)
        with torch.no_grad():
            pred = self._model(x).squeeze(0).detach().cpu().numpy()  # [future_steps, 2]

        target_steps = int(self._trajectory_sampling.num_poses)
        if pred.shape[0] < target_steps:
            tail = np.repeat(pred[-1:, :], target_steps - pred.shape[0], axis=0)
            pred = np.concatenate([pred, tail], axis=0)
        elif pred.shape[0] > target_steps:
            pred = pred[:target_steps]
        return pred.astype(np.float32)

    def _xy_to_trajectory(self, pred_xy: np.ndarray) -> Trajectory:
        n = pred_xy.shape[0]
        headings = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if i == 0:
                dx, dy = float(pred_xy[i, 0]), float(pred_xy[i, 1])
            else:
                dx = float(pred_xy[i, 0] - pred_xy[i - 1, 0])
                dy = float(pred_xy[i, 1] - pred_xy[i - 1, 1])
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                headings[i] = math.atan2(dy, dx)
            else:
                headings[i] = headings[i - 1] if i > 0 else 0.0
        poses = np.column_stack([pred_xy, headings]).astype(np.float32)
        return Trajectory(poses, self._trajectory_sampling)

    def _constant_velocity_fallback(self, agent_input: AgentInput) -> Trajectory:
        ego_velocity_2d = agent_input.ego_statuses[-1].ego_velocity
        ego_speed = float((ego_velocity_2d**2).sum(-1) ** 0.5)
        num_poses = int(self._trajectory_sampling.num_poses)
        dt = float(self._trajectory_sampling.interval_length)
        poses = np.array(
            [[(time_idx + 1) * dt * ego_speed, 0.0, 0.0] for time_idx in range(num_poses)],
            dtype=np.float32,
        )
        return Trajectory(poses, self._trajectory_sampling)
