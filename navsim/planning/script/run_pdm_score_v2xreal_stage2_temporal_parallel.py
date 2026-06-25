#!/usr/bin/env python
"""
V2X-Real Two-Stage PDM Score Evaluation - Temporal Parallel Mode

Scene-parallel version of run_pdm_score_v2xreal_stage2_temporal.py.
Parallelizes evaluation across multiple GPUs by assigning scenes to workers
(round-robin). All frames within a scene are processed by the same worker in
temporal order, preserving the _saved_temporal_state chain.

Usage:
    # Sequential (same as temporal.py):
    python run_pdm_score_v2xreal_stage2_temporal_parallel.py ... --num_workers=1

    # Parallel (8 workers on 4 GPUs → 2 workers per GPU):
    python run_pdm_score_v2xreal_stage2_temporal_parallel.py ... \
        --num_workers=8 --gpu_ids=0,1,2,3
"""

import argparse
import dataclasses as _dc
import logging
import multiprocessing as mp
import os
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import all utility functions / constants from the sequential temporal script
# to maximise code reuse. Framework classes (PDMScorer etc.) are also accessible
# as attributes of that module (Python re-exports top-level names).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_pdm_score_v2xreal_stage2_temporal import (  # noqa: E402
    # Constants
    V2XREAL_CAM_MAPPING, V2XREAL_INFRA_CAM_MAPPING, _ALL_NS_CAM_NAMES, NUM_HISTORY_FRAMES,
    # Utility / helper functions
    _parse_offset_name, _load_agent, _load_scene_filter_config, _build_future_token_map,
    _load_metric_cache, _load_proposal_sampling_from_cache, _make_ego_status,
    _agent_input_from_metric_cache, _compute_infra_sensor_to_ego_lidar,
    _build_v2xreal_agent_input, _is_offset_dir_name, _path_has_offset_dir,
    _discover_metric_caches, _build_temporal_schedule, _discover_offset_dirs, _score_single,
    # Framework classes (re-exported from temporal.py's module-level imports)
    VehicleParameters, PDMScorer, PDMScorerConfig, PDMSimulator,
    LogReplayTrafficAgents, SceneLoaderV2XReal, SceneFilter,
    PDMResults, Trajectory, StateIndex, visualize_prediction_two_stage,
    _build_traffic_agents_policy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_gpu_ids(gpu_ids: str) -> List[str]:
    ids = [x.strip() for x in gpu_ids.split(",") if x.strip()]
    for gpu_id in ids:
        if not gpu_id.isdigit():
            raise ValueError(f"Invalid GPU id '{gpu_id}' in --gpu_ids='{gpu_ids}'")
    return ids


# ---------------------------------------------------------------------------
# Worker function  (runs in a spawned child process)
# ---------------------------------------------------------------------------

def _temporal_worker_fn(
    worker_idx: int,
    gpu_id: str,
    scene_items: List[Tuple[str, List]],
    shared_cfg: Dict[str, Any],
    temp_dir: str,
) -> None:
    """Evaluate an assigned subset of scenes in temporal order.

    Each worker owns its own model, scorer, and temporal state.
    All frames within a scene are processed sequentially to maintain
    the _saved_temporal_state chain.

    :param worker_idx: 0-based worker index (used for log prefix / filenames).
    :param gpu_id: CUDA device id string, e.g. "2".
    :param scene_items: list of (scene_token, schedule) assigned to this worker.
    :param shared_cfg: serialisable config dict built by main().
    :param temp_dir: directory to write per-worker CSV result files.
    """
    try:
        # ── 1. GPU assignment (must happen before any CUDA import) ────────────
        if gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

        # ── 2. Per-worker logging ─────────────────────────────────────────────
        logging.basicConfig(
            level=logging.INFO,
            format=f"%(asctime)s [W{worker_idx:02d}|GPU{gpu_id}] %(levelname)s %(message)s",
            force=True,
        )
        wlog = logging.getLogger(f"worker_{worker_idx}")

        wlog.info(f"Worker {worker_idx} started (pid={os.getpid()}, gpu={gpu_id}, "
                  f"scenes={len(scene_items)})")

        # ── 3. Reconstruct all evaluation components locally ──────────────────
        cache_root = Path(shared_cfg["metric_cache_path"])
        map_root = Path(shared_cfg["map_root"])
        sensor_blob_path = (
            Path(shared_cfg["sensor_blob_path"])
            if shared_cfg.get("sensor_blob_path")
            else None
        )

        proposal_sampling = _load_proposal_sampling_from_cache(cache_root)

        # Mondeo vehicle parameters (same as temporal.py – critical for DAC accuracy)
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
            config=PDMScorerConfig(use_pdms_v1=False, human_penalty_filter=True),
            vehicle_parameters=mondeo_params,
        )
        scorer_stage2 = PDMScorer(
            proposal_sampling=proposal_sampling,
            config=PDMScorerConfig(use_pdms_v1=False, human_penalty_filter=False),
            vehicle_parameters=mondeo_params,
        )
        simulator = PDMSimulator(proposal_sampling=proposal_sampling)
        traffic_policy_name = shared_cfg.get("traffic_policy", "log_replay")
        traffic_agents_policy = _build_traffic_agents_policy(
            traffic_policy_name, proposal_sampling, map_root,
        )
        wlog.info(f"Traffic agents policy: {traffic_policy_name}")

        agent, resolved_agent_name = _load_agent(
            shared_cfg["agent"],
            proposal_sampling,
            shared_cfg.get("agent_class_path"),
        )
        if hasattr(agent, "initialize"):
            agent.initialize()

        # Reconstruct cache path dicts (strings → Path)
        stage1_caches: Dict[str, Path] = {
            k: Path(v) for k, v in shared_cfg["stage1_caches"].items()
        }
        stage2_caches: Dict[str, Dict[str, Path]] = {
            offset: {tok: Path(p) for tok, p in oc.items()}
            for offset, oc in shared_cfg["stage2_caches"].items()
        }

        sf_cfg = shared_cfg["sf_cfg"]
        stage1_only = bool(shared_cfg["stage1_only"])

        # Sensor config
        sensor_cfg = agent.get_sensor_config()
        needs_sensors = any(
            v is True or (isinstance(v, list) and len(v) > 0)
            for v in _dc.asdict(sensor_cfg).values()
        )

        # PKL info dict (needed for sensor agents)
        info_dict: Dict = {}
        token_to_future_pkl_token: Dict[str, str] = {}
        if sensor_blob_path:
            with open(shared_cfg["v2xreal_pkl_path"], "rb") as _f:
                _pkl = pickle.load(_f)
            pkl_infos = _pkl["infos"]
            info_dict = {info["token"]: info for info in pkl_infos}
            token_to_future_pkl_token = _build_future_token_map(
                pkl_infos,
                num_future=sf_cfg["num_future"],
                frame_interval=sf_cfg["frame_interval"],
            )
        elif needs_sensors:
            wlog.warning("Agent needs sensors but sensor_blob_path not provided")

        # Scene loader (requires_scene=True agents, e.g. HumanAgent)
        scene_loader = None
        if agent.requires_scene:
            scene_filter = SceneFilter(
                num_history_frames=sf_cfg["num_history"],
                num_future_frames=sf_cfg["num_future"],
                frame_interval=sf_cfg["frame_interval"],
            )
            scene_loader = SceneLoaderV2XReal(
                pkl_path=Path(shared_cfg["v2xreal_pkl_path"]),
                scene_filter=scene_filter,
                sensor_config=sensor_cfg,
                sensor_blob_path=sensor_blob_path,
                map_root=map_root,
            )

        # Visualisation config
        do_viz = bool(shared_cfg.get("visualize", False))
        viz_output_dir: Optional[Path] = (
            Path(shared_cfg["viz_output_dir"])
            if shared_cfg.get("viz_output_dir")
            else None
        )
        max_viz_local = shared_cfg.get("max_viz")  # per-worker limit
        # Analysis hook (off by default): restrict visualization to specific tokens
        # (one full token per line); these bypass the every-Nth throttle.
        _viz_only_tokens: set = set()
        _vot_file = os.environ.get("VIZ_ONLY_TOKENS_FILE")
        if _vot_file and os.path.isfile(_vot_file):
            with open(_vot_file) as _vf:
                _viz_only_tokens = {ln.strip() for ln in _vf if ln.strip()}

        # ── 4. Evaluation loop (mirrors temporal.py main() exactly) ──────────
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
        stage1_rows_by_token: Dict[str, pd.DataFrame] = {}
        stage2_rows_by_token: Dict[str, List[pd.DataFrame]] = {}
        stage2_trajectories: Dict[str, Dict[str, Trajectory]] = {}
        stage2_simulated_states: Dict[str, Dict[str, np.ndarray]] = {}
        combined_rows_inline: List[Dict] = []

        total_tasks = sum(len(schedule) for _, schedule in scene_items)
        total_s1_tokens = sum(
            1 for _, schedule in scene_items for _, st, _, _ in schedule if st == "s1"
        )
        _max_viz = max_viz_local if max_viz_local is not None else total_s1_tokens
        viz_count = 0
        viz_eligible_count = 0
        global_idx = 0

        wlog.info(f"Tasks: {total_tasks} ({total_s1_tokens} s1 tokens) "
                  f"across {len(scene_items)} scenes")

        for scene_token, schedule in scene_items:
            wlog.info(f"  Scene: {scene_token} ({len(schedule)} tasks)")
            # Reset temporal state at scene boundary
            if hasattr(agent, "_saved_temporal_state"):
                agent._saved_temporal_state = None

            for time_idx, stage, token, offset_names in schedule:
                global_idx += 1

                # ── Stage 1 ───────────────────────────────────────────────────
                if stage == "s1":
                    wlog.info(f"    [{global_idx}/{total_tasks}] t={time_idx} S1 {token}")
                    mc_s1 = None
                    s1_simulated = None
                    try:
                        mc_s1 = _load_metric_cache(stage1_caches[token])
                        if hasattr(agent, "current_stage"):
                            agent.current_stage = "s1"
                            agent.current_offset = None
                        if hasattr(agent, "current_ego_global_translation"):
                            agent.current_ego_global_translation = None
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
                            _ai_s1 = _agent_input_from_metric_cache(mc_s1)
                            _log = getattr(mc_s1, "log_name", "")
                            _ai_s1.token = token[len(_log) + 1:] if (_log and token.startswith(_log + "_")) else token
                            traj_s1 = agent.compute_trajectory(_ai_s1)
                        agent_trajectories[token] = traj_s1

                        s1_row, s1_simulated = _score_single(
                            token, traj_s1, mc_s1, simulator, scorer_stage1,
                            traffic_agents_policy,
                        )
                        s1_row["stage"] = "stage1"
                        stage1_simulated_states[token] = s1_simulated
                    except Exception:
                        wlog.warning(f"      Stage1 FAILED: {traceback.format_exc()}")
                        s1_row = pd.DataFrame([PDMResults.get_empty_results()])
                        s1_row["valid"] = False
                        s1_row["token"] = token
                        s1_row["stage"] = "stage1"
                    stage1_results.append(s1_row)
                    stage1_rows_by_token[token] = s1_row

                    # Stage1-only visualisation
                    s1_ser = s1_row.iloc[0] if isinstance(s1_row, pd.DataFrame) else s1_row
                    s1_valid = bool(s1_ser.get("valid", False))
                    if (
                        stage1_only and do_viz and viz_output_dir is not None
                        and viz_count < _max_viz and s1_valid
                    ):
                        try:
                            if mc_s1 is None:
                                mc_s1 = _load_metric_cache(stage1_caches[token])
                            visualize_prediction_two_stage(
                                metric_cache=mc_s1,
                                pred_trajectory=agent_trajectories.get(token),
                                stage1_row=s1_ser,
                                stage2_data=None,
                                combined_row=None,
                                output_path=viz_output_dir / f"{token}.png",
                                map_root=map_root,
                                simulated_states=s1_simulated,
                            )
                            viz_count += 1
                        except Exception:
                            wlog.warning(f"      Viz FAILED {token}: {traceback.format_exc()}")

                # ── Stage 2 ───────────────────────────────────────────────────
                elif stage == "s2":
                    wlog.info(
                        f"    [{global_idx}/{total_tasks}] t={time_idx} S2 {token} "
                        f"(offsets={len(offset_names)})"
                    )
                    for offset_name in sorted(offset_names):
                        offset_caches = stage2_caches[offset_name]
                        if token not in offset_caches:
                            continue
                        try:
                            mc_s2 = _load_metric_cache(offset_caches[token])
                            if hasattr(agent, "current_stage"):
                                agent.current_stage = "s2"
                                agent.current_offset = offset_name
                            if hasattr(agent, "current_ego_global_translation"):
                                _es2 = mc_s2.ego_state
                                agent.current_ego_global_translation = np.array(
                                    [_es2.rear_axle.x, _es2.rear_axle.y, 0.0],
                                    dtype=np.float64,
                                )
                            if agent.requires_scene:
                                traj_s2 = agent_trajectories.get(token)
                                if traj_s2 is None:
                                    _ai_s2_req = _agent_input_from_metric_cache(mc_s2)
                                    _log_req = getattr(mc_s2, "log_name", "")
                                    _ai_s2_req.token = token[len(_log_req) + 1:] if (_log_req and token.startswith(_log_req + "_")) else token
                                    traj_s2 = agent.compute_trajectory(_ai_s2_req)
                            elif needs_sensors and sensor_blob_path is not None:
                                _log = getattr(mc_s2, "log_name", "")
                                _pkl_tok = (
                                    token[len(_log) + 1:]
                                    if (_log and token.startswith(_log + "_"))
                                    else token
                                )
                                _future_pkl_tok = token_to_future_pkl_token.get(_pkl_tok)
                                future_info_s2 = (
                                    info_dict.get(_future_pkl_tok) if _future_pkl_tok else None
                                )
                                agent_input_s2 = _build_v2xreal_agent_input(
                                    mc_s2, token, info_dict, sensor_blob_path, sensor_cfg,
                                    offset_name=offset_name,
                                    future_info=future_info_s2,
                                )
                                traj_s2 = agent.compute_trajectory(agent_input_s2)
                            else:
                                _ai_s2 = _agent_input_from_metric_cache(mc_s2)
                                _log_s2 = getattr(mc_s2, "log_name", "")
                                _ai_s2.token = token[len(_log_s2) + 1:] if (_log_s2 and token.startswith(_log_s2 + "_")) else token
                                traj_s2 = agent.compute_trajectory(_ai_s2)
                            s2_row, s2_simulated = _score_single(
                                token, traj_s2, mc_s2, simulator, scorer_stage2,
                                traffic_agents_policy,
                            )
                            s2_row["stage"] = "stage2"
                            s2_row["offset"] = offset_name
                            stage2_trajectories.setdefault(token, {})[offset_name] = traj_s2
                            stage2_simulated_states.setdefault(token, {})[offset_name] = (
                                s2_simulated
                            )
                        except Exception:
                            wlog.warning(
                                f"      Stage2 FAILED {offset_name}/{token}: "
                                f"{traceback.format_exc()}"
                            )
                            s2_row = pd.DataFrame([PDMResults.get_empty_results()])
                            s2_row["valid"] = False
                            s2_row["token"] = token
                            s2_row["stage"] = "stage2"
                            s2_row["offset"] = offset_name
                        stage2_results.append(s2_row)
                        stage2_rows_by_token.setdefault(token, []).append(s2_row)

                    # ── Combined score ────────────────────────────────────────
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

                        d_sq = (
                            (s1_endpoint_x - s2_valid_tok["start_point_x"]) ** 2
                            + (s1_endpoint_y - s2_valid_tok["start_point_y"]) ** 2
                        )
                        weights = np.exp(-d_sq.values / (2 * SIGMA_SQUARED))
                        weight_sum = weights.sum()
                        if np.isclose(weight_sum, 0.0) or np.isnan(weight_sum):
                            weights = np.ones(len(s2_valid_tok)) / len(s2_valid_tok)
                        else:
                            weights = weights / weight_sum

                        row_data: Dict[str, Any] = {"token": token}
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
                            do_viz and viz_output_dir is not None
                            and viz_count < _max_viz
                            and (
                                token in _viz_only_tokens if _viz_only_tokens
                                else viz_eligible_count % 4 == 0
                            )
                        ):
                            try:
                                mc_s1_viz = _load_metric_cache(stage1_caches[token])
                                # Policy-propagated background traffic for this token's
                                # stage-1 ego (so the drawn agents reflect idm/log_replay).
                                _sim_tracks = None
                                try:
                                    _sim_tracks = traffic_agents_policy.simulate_environment(
                                        s1_simulated, mc_s1_viz
                                    )
                                except Exception:
                                    _sim_tracks = None
                                stage2_data_viz: Dict[str, Dict] = {}
                                for j, s2r in s2_valid_tok.iterrows():
                                    off = s2r["offset"]
                                    stage2_data_viz[off] = {
                                        "simulated_states": stage2_simulated_states.get(
                                            token, {}
                                        ).get(off),
                                        "trajectory": stage2_trajectories.get(
                                            token, {}
                                        ).get(off),
                                        "start_x": float(s2r["start_point_x"]),
                                        "start_y": float(s2r["start_point_y"]),
                                        "heading": float(
                                            s2r.get(
                                                "start_heading",
                                                mc_s1_viz.ego_state.rear_axle.heading,
                                            )
                                        ),
                                        "weight": float(weights[j]),
                                        "metrics": s2r,
                                    }
                                visualize_prediction_two_stage(
                                    metric_cache=mc_s1_viz,
                                    pred_trajectory=agent_trajectories.get(token),
                                    stage1_row=s1_ser,
                                    stage2_data=stage2_data_viz,
                                    combined_row=combined_ser,
                                    output_path=viz_output_dir / f"{token}.png",
                                    map_root=map_root,
                                    simulated_states=s1_simulated,
                                    simulated_tracks=_sim_tracks,
                                )
                                viz_count += 1
                                if viz_count % 50 == 0:
                                    wlog.info(f"      Saved {viz_count} visualisations so far")
                            except Exception:
                                wlog.warning(
                                    f"      Viz FAILED {token}: {traceback.format_exc()}"
                                )

                        viz_eligible_count += 1

        # ── 5. Write per-worker CSV files ────────────────────────────────────
        tmp = Path(temp_dir)
        prefix = f"worker_{worker_idx:03d}"

        if stage1_results:
            s1_df = pd.concat(stage1_results, ignore_index=True)
            s1_df.to_csv(tmp / f"{prefix}_stage1.csv", index=False)

        if stage2_results:
            s2_df = pd.concat(stage2_results, ignore_index=True)
            s2_df.to_csv(tmp / f"{prefix}_stage2.csv", index=False)

        if combined_rows_inline:
            c_df = pd.DataFrame(combined_rows_inline)
            c_df.to_csv(tmp / f"{prefix}_combined.csv", index=False)

        # Analysis hook (off by default): dump this worker's real (temporal)
        # stage-1 simulated ego trajectories for offline visualization.
        _dump_dir = os.environ.get("VIZ_DUMP_DIR")
        if _dump_dir:
            import pickle as _pickle
            os.makedirs(_dump_dir, exist_ok=True)
            with open(os.path.join(_dump_dir, f"{prefix}_simstates.pkl"), "wb") as _df_f:
                _pickle.dump(stage1_simulated_states, _df_f)

        wlog.info(
            f"Worker {worker_idx} done: "
            f"{len(stage1_results)} s1 rows, "
            f"{len(stage2_results)} s2 rows, "
            f"{len(combined_rows_inline)} combined rows"
            + (f", {viz_count} viz" if do_viz else "")
        )

    except Exception:
        logging.getLogger(f"worker_{worker_idx}").error(
            f"Worker {worker_idx} FATAL ERROR:\n{traceback.format_exc()}"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Result aggregation + logging  (shared between sequential and parallel paths)
# ---------------------------------------------------------------------------

def _save_and_log_results(
    output_dir: Path,
    stage1_dfs: List[pd.DataFrame],
    stage2_dfs: List[pd.DataFrame],
    combined_rows: List[Dict[str, Any]],
    sub_metric_cols: List[str],
    visualize: bool = False,
    viz_count: int = 0,
    viz_output_dir: Optional[Path] = None,
) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info("Aggregating results")
    logger.info("=" * 70)

    stage1_df = pd.concat(stage1_dfs, ignore_index=True)
    stage2_df = pd.concat(stage2_dfs, ignore_index=True) if stage2_dfs else pd.DataFrame()

    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    stage1_df.to_csv(output_dir / f"{timestamp}_stage1_raw.csv", index=False)
    if len(stage2_df) > 0:
        stage2_df.to_csv(output_dir / f"{timestamp}_stage2_raw.csv", index=False)

    score_cols = [
        c for c in stage1_df.columns
        if c not in {
            "valid", "token", "log_name", "start_time", "stage", "offset",
            "endpoint_x", "endpoint_y", "start_point_x", "start_point_y",
            "start_heading", "frame_type", "weighted_metrics",
            "weighted_metrics_array", "traffic_light_compliance",
        }
        and stage1_df[c].dtype in [np.float64, np.float32, float, int]
    ]

    valid_s1 = stage1_df[stage1_df["valid"] == True]
    if len(valid_s1) > 0:
        s1_summary = valid_s1[sub_metric_cols].mean().to_frame(name="mean").T
        s1_summary.to_csv(output_dir / f"{timestamp}_stage1_summary.csv", index=False)

    if len(stage2_df) > 0:
        valid_s2_all = stage2_df[stage2_df["valid"] == True]
        if len(valid_s2_all) > 0:
            s2_summary = valid_s2_all[sub_metric_cols].mean().to_frame(name="mean").T
            s2_summary.to_csv(output_dir / f"{timestamp}_stage2_summary.csv", index=False)

    logger.info("")
    logger.info("Stage 1 Results (original ego pose):")
    for col in score_cols:
        if col in valid_s1.columns:
            logger.info(
                f"  {col:35s}: {valid_s1[col].mean():8.4f} ± {valid_s1[col].std():8.4f}"
            )

    if len(stage2_df) > 0:
        logger.info("")
        logger.info("Stage 2 Results (per offset):")
        valid_s2 = stage2_df[stage2_df["valid"] == True]
        for off in sorted(valid_s2["offset"].unique()):
            od = valid_s2[valid_s2["offset"] == off]
            pdm = od["pdm_score"].mean() if "pdm_score" in od.columns else float("nan")
            logger.info(f"  {off:>12s}: pdm_score={pdm:.4f}  (n={len(od)})")
        logger.info("")
        logger.info("Stage 2 Aggregate (mean across all offsets):")
        for col in score_cols:
            if col in valid_s2.columns:
                logger.info(
                    f"  {col:35s}: {valid_s2[col].mean():8.4f} ± {valid_s2[col].std():8.4f}"
                )

    if combined_rows:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Combined Two-Stage Score (Gaussian kernel weighted)")
        logger.info("=" * 70)

        combined_df = pd.DataFrame(combined_rows)
        combined_df.to_csv(output_dir / f"{timestamp}_combined.csv", index=False)
        logger.info(f"Tokens with combined score: {len(combined_df)}")

        # Column order: n_tokens | combined_pdm_score | stage1_* | stage2_*
        summary_row: Dict[str, Any] = {"n_tokens": len(combined_df)}
        summary_row["combined_pdm_score"] = (
            combined_df["combined_pdm_score"].mean()
            if "combined_pdm_score" in combined_df.columns
            else float("nan")
        )
        for col in sub_metric_cols:
            key = f"stage1_{col}"
            summary_row[key] = (
                combined_df[key].mean() if key in combined_df.columns else float("nan")
            )
        for col in sub_metric_cols:
            key = f"stage2_{col}"
            summary_row[key] = (
                combined_df[key].mean() if key in combined_df.columns else float("nan")
            )
        summary_df = pd.DataFrame([summary_row])
        summary_df.to_csv(output_dir / f"{timestamp}_combined_summary.csv", index=False)

        logger.info(f"  combined_pdm_score: {summary_row['combined_pdm_score']:.4f}")
        logger.info(f"  {'metric':35s} {'stage1':>8s} {'stage2':>8s}")
        logger.info(f"  {'-'*35} {'-'*8} {'-'*8}")
        for col in sub_metric_cols:
            s1_m = summary_row.get(f"stage1_{col}", float("nan"))
            s2_m = summary_row.get(f"stage2_{col}", float("nan"))
            logger.info(f"  {col:35s} {s1_m:8.4f} {s2_m:8.4f}")

        if visualize:
            logger.info(f"\nVisualization: {viz_count} images saved to {viz_output_dir}")

    logger.info("")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)
    logger.info("Evaluation Complete!")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="V2X-Real Temporal Two-Stage PDM Score Evaluation (Scene-Parallel)"
    )
    # ── Eval config ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--config", type=str, default=None,
        help="Eval config (configs/eval/*.py). Seeds agent / paths / policy / "
             "workers from the config + configs/eval/paths.py. Explicit flags "
             "and environment variables still override it.",
    )
    # ── Args from temporal.py (resolved from --config when omitted) ──────────
    # Machine paths default to None and are filled from configs/eval/paths.py.
    parser.add_argument("--v2xreal_pkl_path", type=str, default=None)
    parser.add_argument("--map_root", type=str, default=None)
    parser.add_argument("--metric_cache_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--agent", type=str, default=None,
        choices=["constant_velocity", "stop", "human", "cos_v2x", "sparsedrive_navsim"],
    )
    parser.add_argument("--agent_class_path", type=str, default=None)
    parser.add_argument("--sensor_blob_path", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Limit s1 tokens for debugging (applied before scene assignment)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save BEV visualisations (per-worker, no filename collisions)")
    parser.add_argument("--viz_output_dir", type=str, default=None,
                        help="Visualisation output dir (default: <output_dir>/viz)")
    parser.add_argument("--max_viz", type=int, default=None,
                        help="Max visualisations per worker (default: all)")
    parser.add_argument("--stage1_only", action="store_true",
                        help="Run Stage 1 only; skip Stage 2 and combined score")
    parser.add_argument("--scene_filter_file", type=str, default=None,
                        help="Restrict evaluation to the scenes (one log_name per line) listed in "
                             "this file, e.g. the no-stop benchmark subset. The aggregated score is "
                             "then computed directly over those scenes (no post-hoc filtering needed).")
    parser.add_argument("--traffic_policy", type=str, default=None,
                        choices=["log_replay", "idm", "constant_velocity"],
                        help="Background traffic agents policy: log_replay (GT replay, default), "
                             "idm (reactive Intelligent Driver Model), or constant_velocity")
    # ── Parallelism args ─────────────────────────────────────────────────────
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of scene-parallel workers. 1 = sequential (identical to temporal.py).",
    )
    parser.add_argument(
        "--gpu_ids", type=str, default=None,
        help="Comma-separated GPU IDs, e.g. '0,1,2,3'. Required when num_workers > 1.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive worker/GPU confirmation prompt.",
    )
    args = parser.parse_args()

    # ── Resolve settings from the eval config (configs/eval/*.py) ─────────────
    # Precedence: explicit CLI flag > environment variable > experiment config >
    # configs/eval/paths.py > built-in default. The shell entry only picks the
    # conda env and hands off --config; everything else is resolved here.
    from navsim.common.vips_config import load_eval_config

    cfg = load_eval_config(args.config)
    repo_root = os.environ.get("NAVSIM_DEVKIT_ROOT") or str(Path(__file__).resolve().parents[3])

    def _cfg(key, default=None):
        val = os.environ.get(key)
        if val not in (None, ""):
            return val
        val = cfg.get(key)
        if val not in (None, ""):
            return val
        return default

    def _truthy(val) -> bool:
        return str(val).strip().lower() in ("1", "true", "yes", "on")

    args.v2xreal_pkl_path = args.v2xreal_pkl_path or _cfg("V2XREAL_PKL_PATH")
    args.map_root = args.map_root or _cfg("V2XREAL_MAP_ROOT")
    args.metric_cache_path = args.metric_cache_path or _cfg("METRIC_CACHE_PATH")
    args.agent = args.agent or _cfg("AGENT", "constant_velocity")
    args.agent_class_path = args.agent_class_path or _cfg("AGENT_CLASS_PATH")
    args.traffic_policy = args.traffic_policy or _cfg("TRAFFIC_POLICY", "log_replay")
    args.num_workers = int(args.num_workers if args.num_workers is not None else _cfg("NUM_WORKERS", 1))
    args.gpu_ids = args.gpu_ids if args.gpu_ids is not None else _cfg("GPU_IDS", "0")
    args.visualize = bool(args.visualize) or _truthy(_cfg("VISUALIZE", "0"))
    args.stage1_only = bool(args.stage1_only) or _truthy(_cfg("STAGE1_ONLY", "0"))
    if args.max_viz is None and _cfg("MAX_VIZ"):
        args.max_viz = int(_cfg("MAX_VIZ"))
    if args.max_tokens is None and _cfg("MAX_TOKENS"):
        args.max_tokens = int(_cfg("MAX_TOKENS"))

    # Sensor blob: needed by the CoS-V2X model and custom agent classes; the
    # geometry-only baselines (constant_velocity/stop/human) don't load sensors.
    if args.sensor_blob_path is None and (
        args.agent in ("cos_v2x", "sparsedrive_navsim") or args.agent_class_path
    ):
        args.sensor_blob_path = _cfg("SENSOR_BLOB_PATH")

    # Scene filter: default to the no-stop benchmark subset so the aggregate is
    # the reported number directly. An explicit empty value means "full split".
    if args.scene_filter_file is None:
        sff = os.environ.get("SCENE_FILTER_FILE", cfg.get("SCENE_FILTER_FILE"))
        if sff is None:
            sff = os.path.join(repo_root, "scripts/evaluation/test_scene_tokens_5s_no_stop.txt")
        args.scene_filter_file = sff or None

    # Output dir base: OUTPUT_DIR (explicit), else
    # <OUTPUT_BASE_DIR | NAVSIM_EXP_ROOT/...>[/EXPERIMENT_NAME].
    # main() appends the agent + traffic-policy sub-dir below.
    if args.output_dir is None:
        args.output_dir = _cfg("OUTPUT_DIR")
    if args.output_dir is None:
        navsim_exp_root = _cfg("NAVSIM_EXP_ROOT", os.path.join(repo_root, "exp_v2xreal"))
        base = _cfg("OUTPUT_BASE_DIR",
                    os.path.join(navsim_exp_root, "v2xreal_pdm_score_stage2_coop_2hz_5s"))
        exp_name = _cfg("EXPERIMENT_NAME", "")
        args.output_dir = os.path.join(base, exp_name) if exp_name else base

    # Bridge model-location knobs from the experiment config (configs/eval/*.py)
    # to the environment so the agent — which reads them via vips_config in the
    # spawned ('spawn') workers — sees them. Relative paths (e.g. models/CoS-V2X)
    # are resolved against the repo root so they work regardless of the worker CWD.
    for _k in ("COS_V2X_FOLDER", "COS_V2X_CONFIG_PATH", "COS_V2X_MODEL_CHECKPOINT_PATH",
               "ADMLP_V2_CKPT_PATH", "ADMLP_V2_STATS_PATH"):
        _v = _cfg(_k)
        if _v and not os.environ.get(_k):
            os.environ[_k] = _v if os.path.isabs(_v) else os.path.join(repo_root, _v)
    for _k in ("COS_V2X_MODE", "SPARSEDRIVE_MODE"):
        _v = _cfg(_k)
        if _v and not os.environ.get(_k):
            os.environ[_k] = _v

    _missing = [n for n, v in (
        ("v2xreal_pkl_path", args.v2xreal_pkl_path),
        ("map_root", args.map_root),
        ("metric_cache_path", args.metric_cache_path),
    ) if not v]
    if _missing:
        parser.error(
            "missing required path(s): " + ", ".join(f"--{n}" for n in _missing)
            + " (set them in configs/eval/paths.py or pass the flag)"
        )

    logger.info("Resolved eval config:")
    logger.info(f"  agent={args.agent}  class={args.agent_class_path or 'builtin'}  "
                f"policy={args.traffic_policy}  workers={args.num_workers}  gpus={args.gpu_ids or 'N/A'}")
    logger.info(f"  pkl={args.v2xreal_pkl_path}")
    logger.info(f"  map={args.map_root}")
    logger.info(f"  cache={args.metric_cache_path}")
    logger.info(f"  out={args.output_dir}")
    logger.info(f"  scene_filter={args.scene_filter_file or '(full split)'}")

    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    if args.num_workers > 1 and not args.gpu_ids:
        raise ValueError("Parallel mode (num_workers > 1) requires --gpu_ids, e.g. '0,1,2,3'")
    gpu_ids = _parse_gpu_ids(args.gpu_ids) if args.gpu_ids else []

    cache_root = Path(args.metric_cache_path)
    map_root = Path(args.map_root)

    logger.info("=" * 70)
    logger.info("V2X-Real Temporal Two-Stage PDM Score Evaluation (Scene-Parallel)")
    logger.info("=" * 70)
    logger.info(f"num_workers={args.num_workers}  gpu_ids={args.gpu_ids or 'N/A'}")

    # ── Common setup ─────────────────────────────────────────────────────────
    with open(args.v2xreal_pkl_path, "rb") as _f:
        _pkl = pickle.load(_f)
    pkl_infos = _pkl["infos"]
    logger.info(f"Loaded PKL: {len(pkl_infos)} frames")

    sf_cfg = _load_scene_filter_config(cache_root)
    logger.info(f"Scene filter config: {sf_cfg}")

    stage1_caches = _discover_metric_caches(cache_root, include_offset_dirs=False)
    logger.info(f"Stage 1 caches: {len(stage1_caches)}")

    stage2_caches: Dict[str, Dict[str, Path]] = {}
    if not args.stage1_only:
        for offset_name, offset_dir in _discover_offset_dirs(cache_root):
            stage2_caches[offset_name] = _discover_metric_caches(offset_dir)
            logger.info(f"  {offset_name}: {len(stage2_caches[offset_name])} caches")
    logger.info(f"Stage 2 offsets: {len(stage2_caches)}")

    temporal_schedule = _build_temporal_schedule(
        pkl_infos, stage1_caches, stage2_caches,
        num_future=sf_cfg["num_future"],
        frame_interval=sf_cfg["frame_interval"],
    )
    total_tasks = sum(len(v) for v in temporal_schedule.values())
    total_tokens = sum(
        1 for v in temporal_schedule.values() for _, st, _, _ in v if st == "s1"
    )
    logger.info(
        f"Temporal schedule: {total_tasks} tasks ({total_tokens} s1 tokens) "
        f"across {len(temporal_schedule)} scenes"
    )

    # Scene-level evaluation filter (e.g. the no-stop benchmark subset). Restricts
    # the eval to the listed scenes up front, so the aggregated score is the
    # reported number with no post-hoc filtering. Temporal state resets per scene,
    # so dropping whole scenes does not change the remaining scenes' results.
    if args.scene_filter_file:
        sf_path = Path(args.scene_filter_file)
        if not sf_path.exists():
            raise FileNotFoundError(f"--scene_filter_file not found: {sf_path}")
        allowed_scenes = {ln.strip() for ln in sf_path.read_text().splitlines() if ln.strip()}
        temporal_schedule = {k: v for k, v in temporal_schedule.items() if k in allowed_scenes}
        missing = allowed_scenes - set(temporal_schedule.keys())
        total_tasks = sum(len(v) for v in temporal_schedule.values())
        total_tokens = sum(
            1 for v in temporal_schedule.values() for _, st, _, _ in v if st == "s1"
        )
        logger.info(
            f"Scene filter ({sf_path.name}): {len(allowed_scenes)} listed -> "
            f"{len(temporal_schedule)} matched, {total_tokens} s1 tokens, {total_tasks} tasks"
        )
        if missing:
            logger.warning(f"  {len(missing)} listed scenes not present in the cache: {sorted(missing)[:5]}")

    # Analysis hook (off by default): restrict to specific scenes for targeted
    # visualization. Temporal state resets per scene, so dropping whole scenes
    # does not affect the remaining scenes' results.
    _viz_scenes = os.environ.get("VIZ_TARGET_SCENES", "")
    if _viz_scenes:
        _keep = [s.strip() for s in _viz_scenes.split(",") if s.strip()]
        temporal_schedule = {
            k: v for k, v in temporal_schedule.items() if any(t in k for t in _keep)
        }
        total_tasks = sum(len(v) for v in temporal_schedule.values())
        logger.info(f"  (VIZ_TARGET_SCENES → {len(temporal_schedule)} scenes, {total_tasks} tasks)")

    # Apply max_tokens limit before splitting scenes
    if args.max_tokens:
        limited_schedule: Dict = {}
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

    sub_metric_cols = [
        "no_at_fault_collisions", "drivable_area_compliance",
        "driving_direction_compliance",
        "ego_progress", "time_to_collision_within_bound",
        "lane_keeping", "history_comfort", "pdm_score",
    ]

    # ── Resolve output directory ──────────────────────────────────────────────
    # Need agent name to build output path; resolve without fully loading agent here
    if args.agent_class_path:
        agent_out_name = args.agent_class_path.rsplit(".", 1)[-1]
    else:
        agent_out_name = args.agent
    if args.stage1_only:
        agent_out_name = f"{agent_out_name}_stage1"
    # Suffix with traffic policy so different policies don't overwrite each other
    agent_out_name = f"{agent_out_name}_{args.traffic_policy}"
    output_dir = Path(args.output_dir) / agent_out_name
    output_dir.mkdir(parents=True, exist_ok=True)

    viz_output_dir: Optional[Path] = None
    if args.visualize:
        viz_output_dir = (
            Path(args.viz_output_dir) if args.viz_output_dir else output_dir / "viz"
        )
        viz_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Scene assignment (greedy least-loaded for balanced distribution) ────
    import heapq
    num_workers = args.num_workers
    # Sort scenes by descending task count so largest scenes are placed first
    scene_tokens = sorted(
        temporal_schedule.keys(),
        key=lambda s: len(temporal_schedule[s]),
        reverse=True,
    )
    worker_scene_groups: List[List] = [[] for _ in range(num_workers)]
    # Min-heap: (current_task_count, worker_index)
    heap = [(0, i) for i in range(num_workers)]
    for sc in scene_tokens:
        load, wi = heapq.heappop(heap)
        worker_scene_groups[wi].append((sc, temporal_schedule[sc]))
        heapq.heappush(heap, (load + len(temporal_schedule[sc]), wi))

    # ── GPU-Worker mapping (randomised to avoid load collision across jobs) ──
    import random as _random
    _gpu_pool = (gpu_ids * ((num_workers // len(gpu_ids)) + 1))[:num_workers]
    _random.shuffle(_gpu_pool)
    worker_gpu_ids: List[str] = _gpu_pool

    gpu_worker_map: Dict[str, List[int]] = {}
    for i in range(num_workers):
        gid = worker_gpu_ids[i]
        gpu_worker_map.setdefault(gid, []).append(i)

    total_scenes = len(scene_tokens)
    total_tasks = sum(len(sched) for sched in temporal_schedule.values())

    print("\n" + "=" * 60)
    print("  GPU / Worker Assignment Summary")
    print("=" * 60)
    for gid in sorted(gpu_worker_map.keys(), key=lambda x: int(x)):
        workers = gpu_worker_map[gid]
        worker_details = []
        for wi in workers:
            n_s = len(worker_scene_groups[wi])
            n_t = sum(len(sched) for _, sched in worker_scene_groups[wi])
            worker_details.append(f"W{wi:02d}({n_s}sc/{n_t}tasks)")
        print(f"  GPU {gid}: {len(workers)} worker(s) — {', '.join(worker_details)}")
    print("-" * 60)
    print(f"  Total: {num_workers} workers across {len(gpu_worker_map)} GPU(s)")
    print(f"  Scenes: {total_scenes},  Visualize: {bool(args.visualize)}")
    print("=" * 60)

    if not args.yes:
        if not sys.stdin.isatty():
            # Non-interactive stdin (pipe, redirect, background): don't block on input().
            print("Non-interactive stdin detected; proceeding (pass --yes to silence this).")
        else:
            confirm = input("\nProceed with evaluation? [y/N]: ").strip().lower()
            if confirm not in ("y", "yes"):
                print("Aborted by user.")
                return

    if num_workers > 1:
        for i, group in enumerate(worker_scene_groups):
            n_s = len(group)
            n_t = sum(len(sched) for _, sched in group)
            n_s1 = sum(1 for _, sched in group for _, st, _, _ in sched if st == "s1")
            logger.info(
                f"  Worker {i} (GPU {worker_gpu_ids[i]}): "
                f"{n_s} scenes, {n_t} tasks, {n_s1} s1 tokens"
            )

    # ── Build serialisable shared config ────────────────────────────────────
    shared_cfg: Dict[str, Any] = {
        "v2xreal_pkl_path": str(args.v2xreal_pkl_path),
        "map_root": str(map_root),
        "metric_cache_path": str(cache_root),
        "agent": args.agent,
        "agent_class_path": args.agent_class_path,
        "sensor_blob_path": args.sensor_blob_path,
        "stage1_only": bool(args.stage1_only),
        "traffic_policy": args.traffic_policy,
        "stage1_caches": {k: str(v) for k, v in stage1_caches.items()},
        "stage2_caches": {
            offset: {tok: str(p) for tok, p in oc.items()}
            for offset, oc in stage2_caches.items()
        },
        "sf_cfg": sf_cfg,
        "visualize": bool(args.visualize),
        "viz_output_dir": str(viz_output_dir) if viz_output_dir else None,
        "max_viz": args.max_viz,
    }

    # ── Sequential path (num_workers == 1) ───────────────────────────────────
    if num_workers == 1:
        logger.info("Running in sequential mode (num_workers=1)")
        temp_dir = output_dir / "worker_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        # Assign first (and only) GPU if provided
        single_gpu = gpu_ids[0] if gpu_ids else (os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        _temporal_worker_fn(0, single_gpu, worker_scene_groups[0], shared_cfg, str(temp_dir))
        # Aggregate from single worker
        stage1_dfs, stage2_dfs, combined_rows = _collect_worker_csvs(temp_dir, num_workers)
        _save_and_log_results(
            output_dir, stage1_dfs, stage2_dfs, combined_rows, sub_metric_cols,
            visualize=bool(args.visualize), viz_output_dir=viz_output_dir,
        )
        return

    # ── Parallel path (num_workers > 1) ─────────────────────────────────────
    logger.info(f"Running in parallel mode ({num_workers} workers)")
    temp_dir = output_dir / "worker_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    processes: List[mp.Process] = []
    for i in range(num_workers):
        p = ctx.Process(
            target=_temporal_worker_fn,
            args=(i, worker_gpu_ids[i], worker_scene_groups[i], shared_cfg, str(temp_dir)),
            name=f"temporal_worker_{i}",
        )
        p.start()
        processes.append(p)
        logger.info(f"Started worker {i} (pid={p.pid}, gpu={worker_gpu_ids[i]})")

    for p in processes:
        p.join()

    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        logger.warning(
            f"Workers {failed} exited with non-zero codes "
            f"{[processes[i].exitcode for i in failed]}. "
            "Aggregating available results."
        )

    stage1_dfs, stage2_dfs, combined_rows = _collect_worker_csvs(temp_dir, num_workers)
    if not stage1_dfs:
        logger.error("No stage1 results found from any worker. Evaluation failed.")
        return

    _save_and_log_results(
        output_dir, stage1_dfs, stage2_dfs, combined_rows, sub_metric_cols,
        visualize=bool(args.visualize), viz_output_dir=viz_output_dir,
    )


def _collect_worker_csvs(
    temp_dir: Path, num_workers: int
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[Dict[str, Any]]]:
    """Read per-worker CSV files written by _temporal_worker_fn."""
    stage1_dfs: List[pd.DataFrame] = []
    stage2_dfs: List[pd.DataFrame] = []
    combined_rows: List[Dict[str, Any]] = []
    for i in range(num_workers):
        prefix = f"worker_{i:03d}"
        s1_f = temp_dir / f"{prefix}_stage1.csv"
        if s1_f.exists():
            stage1_dfs.append(pd.read_csv(s1_f))
        else:
            logger.warning(f"Worker {i} stage1 CSV not found: {s1_f}")
        s2_f = temp_dir / f"{prefix}_stage2.csv"
        if s2_f.exists():
            stage2_dfs.append(pd.read_csv(s2_f))
        c_f = temp_dir / f"{prefix}_combined.csv"
        if c_f.exists():
            combined_rows.extend(pd.read_csv(c_f).to_dict("records"))
    return stage1_dfs, stage2_dfs, combined_rows


if __name__ == "__main__":
    main()
