import copy
from dataclasses import dataclass
import os
from typing import List, Optional
from shapely.geometry import Point

import numpy as np
import numpy.typing as npt
import pandas as pd
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import Point, creation

from navsim.common.dataclasses import PDMResults
from navsim.planning.metric_caching.metric_cache import MapParameters
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import ego_is_comfortable
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type
from navsim.planning.simulation.planner.pdm_planner.scoring.vis_utils import *
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    coords_array_to_polygon_array,
    ego_states_to_state_array,
    state_array_to_coords_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    EgoAreaIndex,
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


import matplotlib.pyplot as plt

from shapely import affinity

def _draw_heading_arrow(
    ax: plt.Axes,
    x: float,
    y: float,
    heading: float,
    length: float = 5.0,
    color: str = "tab:blue",
    linewidth: float = 2.0,
    text_prefix: str = "ego_heading",
    text_offset: float = 0.8,
):
    """
    Draw a heading arrow at (x, y) with angle `heading` in radians.
    Convention: heading 0 -> +x direction, heading pi/2 -> +y direction.
    """
    dx = float(np.cos(heading) * length)
    dy = float(np.sin(heading) * length)

    # Arrow (use quiver for consistent scale)
    ax.quiver(
        [x], [y], [dx], [dy],
        angles="xy", scale_units="xy", scale=1,
        width=0.006, headwidth=4.0, headlength=6.0,
        color=color
    )

    # Text (deg + rad)
    deg = float(np.degrees(heading))
    ax.text(
        x + text_offset, y + text_offset,
        f"{text_prefix}: {heading:+.3f} rad ({deg:+.1f} deg)",
        fontsize=9,
        color=color,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"),
    )


def _shapely_to_ego_frame(geom, ego_pose: StateSE2):
    """
    Transform shapely geometry from GLOBAL -> EGO frame.
    EGO frame convention:
      x = forward, y = left  (nuPlan StateSE2)
    We rotate by -heading and translate by (-x, -y).
    """
    if geom is None:
        return None

    # translate global so ego at origin
    translated = affinity.affine_transform(geom, [1, 0, 0, 1, -ego_pose.x, -ego_pose.y])

    # rotate by -heading around origin
    c = float(np.cos(-ego_pose.heading))
    s = float(np.sin(-ego_pose.heading))
    # [a, b, d, e, xoff, yoff] where:
    # x' = a*x + b*y + xoff
    # y' = d*x + e*y + yoff
    rotated = affinity.affine_transform(translated, [c, -s, s, c, 0, 0])
    return rotated

def _plot_shapely(ax, geom, **kwargs):
    """Plot shapely Polygon or MultiPolygon to matplotlib axis."""
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
        # fallback: try bounds box
        minx, miny, maxx, maxy = geom.bounds
        ax.plot([minx, maxx, maxx, minx, minx],
                [miny, miny, maxy, maxy, miny], **kwargs)

@dataclass
class PDMScorerConfig:

    # weighted metric weights
    progress_weight: float = 5.0
    ttc_weight: float = 5.0
    comfort_weight: float = 5.0  # Used in PDMS v1 instead of HC + EC
    lane_keeping_weight: float = 2.0
    history_comfort_weight: float = 2.0
    two_frame_extended_comfort_weight: float = 2.0

    # thresholds
    # comfort related config in navsim/planning/simulation/planner/pdm_planner/scoring/pdm_comfort_metrics.py
    driving_direction_horizon: float = 1.0  # [s] (driving direction) (nuplan)
    driving_direction_compliance_threshold: float = 2.0  # [m] (driving direction) (nuplan)
    driving_direction_violation_threshold: float = 6.0  # [m] (driving direction) (nuplan)

    stopped_speed_threshold: float = 5e-03  # [m/s] (ttc)
    future_collision_horizon_window: float = 1.0  # [s] (ttc)
    progress_distance_threshold: float = 5.0  # [m] (progress)
    lane_keeping_deviation_limit: float = 0.5  # [m] (lane keeping) (hydraMDP++)
    lane_keeping_horizon_window: float = 2.0  # [s] (lane keeping) (hydraMDP++)

    # version flag
    use_pdms_v1: bool = False  # If True, use PDMS v1 metrics (NC, DAC, EP, TTC, Comfort); else use EPDMS v2

    # DAC flag
    dac_use_center: bool = True  # True: center-based (relaxed), False: 4-corners based (original strict)

    # human flag
    human_penalty_filter: Optional[bool] = None

    @property
    def weighted_metrics_array(self) -> npt.NDArray[np.float64]:
        weighted_metrics = np.zeros(len(WeightedMetricIndex), dtype=np.float64)
        if self.use_pdms_v1:
            # PDMS v1: use comfort instead of lane keeping, HC, and EC
            weighted_metrics[WeightedMetricIndex.PROGRESS] = self.progress_weight
            weighted_metrics[WeightedMetricIndex.TTC] = self.ttc_weight
            weighted_metrics[WeightedMetricIndex.LANE_KEEPING] = self.comfort_weight  # Reuse for comfort
        else:
            # EPDMS v2
            weighted_metrics[WeightedMetricIndex.PROGRESS] = self.progress_weight
            weighted_metrics[WeightedMetricIndex.TTC] = self.ttc_weight
            weighted_metrics[WeightedMetricIndex.LANE_KEEPING] = self.lane_keeping_weight
            weighted_metrics[WeightedMetricIndex.HISTORY_COMFORT] = self.history_comfort_weight
            weighted_metrics[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = self.two_frame_extended_comfort_weight
        return weighted_metrics


class PDMScorer:
    """Class to score proposals in PDM pipeline. Re-implements nuPlan's closed-loop metrics."""

    def __init__(
        self,
        proposal_sampling: TrajectorySampling,
        config: PDMScorerConfig = PDMScorerConfig(),
        vehicle_parameters: VehicleParameters = get_pacifica_parameters(),
    ):
        """
        Constructor of PDMScorer
        :param proposal_sampling: Sampling parameters for proposals
        """
        self.proposal_sampling = proposal_sampling
        self._config = config
        self._vehicle_parameters = vehicle_parameters

        # lazy loaded
        self._observation: Optional[PDMObservation] = None
        self._centerline: Optional[PDMPath] = None
        self._route_lane_ids: Optional[List[str]] = None
        self._drivable_area_map: Optional[PDMDrivableMap] = None
        self._human_past_trajectory: Optional[InterpolatedTrajectory] = None

        self._num_proposals: Optional[int] = None
        self._states: Optional[npt.NDArray[np.float64]] = None
        self._ego_coords: Optional[npt.NDArray[np.float64]] = None
        self._ego_polygons: Optional[npt.NDArray[np.object_]] = None

        self._ego_areas: Optional[npt.NDArray[np.bool_]] = None

        self._multi_metrics: Optional[npt.NDArray[np.float64]] = None
        self._weighted_metrics: Optional[npt.NDArray[np.float64]] = None
        self._progress_raw: Optional[npt.NDArray[np.float64]] = None

        self._collision_time_idcs: Optional[npt.NDArray[np.float64]] = None
        self._ttc_time_idcs: Optional[npt.NDArray[np.float64]] = None


    ############### NOTE HM added debugging functions

    def debug_viz_proposals_with_ddc(
        self,
        proposal_indices=None,
        every_k: int = 1,
        radius_m: float = 80.0,
        ego_frame: bool = False,              # False: GLOBAL, True: current ego frame
        time_idx_for_ego_frame: int = 0,       # reference time index when ego_frame=True
        out_path: str = None,
        title: str = None,
        show_centerline: bool = True,
        show_oncoming_points: bool = True,    # mark points classified as ONCOMING_TRAFFIC
        annotate_each: bool = True,           # show a text box per proposal
        ):
        """
        Plot multiple proposal trajectories and overlay each proposal's driving_direction_compliance (DDC).
        - DDC uses the value of self._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, p].
        - The ONCOMING_TRAFFIC mask (ego_area) is also shown to verify the computation behaves as intended.
        """

        if self._ego_coords is None or self._states is None:
            raise ValueError("Call after _reset() & coordinate computation.")
        if self._multi_metrics is None:
            raise ValueError("Call after metrics computed.")
        if self._ego_areas is None:
            raise ValueError("Call after _calculate_ego_area().")

        nP = self._ego_coords.shape[0]
        T = self._ego_coords.shape[1]

        if proposal_indices is None:
            proposal_indices = list(range(nP))
        else:
            proposal_indices = list(proposal_indices)

        # centerline shapely LineString (GLOBAL)
        ls = getattr(self._centerline, "linestring", None) if self._centerline is not None else None

        # get coordinates: ego center (GLOBAL)
        centers_g = self._ego_coords[:, :, BBCoordsIndex.CENTER, :]  # [P, T, 2]
        centers_g_s = centers_g[:, ::every_k, :]
        Ts = centers_g_s.shape[1]

        # reference pose for ego-frame transform
        ego_pose_g = None
        if ego_frame:
            se2 = self._states[proposal_indices[0], time_idx_for_ego_frame, StateIndex.STATE_SE2]
            ego_pose_g = StateSE2(float(se2[0]), float(se2[1]), float(se2[2]))

            def to_ego_xy(xy_g):
                dx = xy_g[:, 0] - ego_pose_g.x
                dy = xy_g[:, 1] - ego_pose_g.y
                c = float(np.cos(-ego_pose_g.heading))
                s = float(np.sin(-ego_pose_g.heading))
                x = dx * c - dy * s
                y = dx * s + dy * c
                return np.stack([x, y], axis=1)

        fig, ax = plt.subplots(figsize=(9, 9))

        # centerline draw
        if show_centerline and ls is not None:
            clx, cly = ls.xy
            cl = np.stack([np.asarray(clx), np.asarray(cly)], axis=1)
            if ego_frame:
                cl = to_ego_xy(cl)
            ax.plot(cl[:, 0], cl[:, 1], linewidth=2.0, alpha=0.9, label="centerline")

        # ONCOMING mask (GLOBAL T) -> align to sampled index
        oncoming_mask_full = self._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]  # [P, T]
        oncoming_mask = oncoming_mask_full[:, ::every_k]  # [P, Ts]

        # plot per proposal
        for pi, p in enumerate(proposal_indices):
            traj = centers_g_s[p]  # [Ts, 2] in GLOBAL
            if ego_frame:
                traj = to_ego_xy(traj)

            # base trajectory
            ax.plot(traj[:, 0], traj[:, 1], linewidth=1.8, alpha=0.9, marker="o", markersize=2.0)

            # highlight points classified as ONCOMING
            if show_oncoming_points:
                m = oncoming_mask[p]
                if np.any(m):
                    ax.scatter(traj[m, 0], traj[m, 1], s=22, marker="x", zorder=4)

            # final DDC score
            ddc = float(self._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, p])

            # text box per proposal: next to the start point
            if annotate_each:
                x0, y0 = float(traj[0, 0]), float(traj[0, 1])
                ax.text(
                    x0, y0,
                    f"p={p}  DDC={ddc:.2f}",
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"),
                )

        # cosmetics
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        add_lane_polygons_with_ids_and_arrows(
            ax=ax,
            drivable_area_map=self._drivable_area_map,
            map_api=getattr(self, "_map_api", None),   # pass if available, else None
            ego_pose_se2=ego_pose_g,                  # global ego pose
            ego_frame=False,
            radius_m=200.0,
            draw_lane=True,
            draw_lane_connector=True,
            draw_intersection=False,
            max_items=250,
            arrow_every=1,
            id_fontsize=7,
        )

        # view window: based on the first proposal's start point
        cx, cy = centers_g_s[proposal_indices[0], 0]
        if ego_frame:
            cx, cy = 0.0, 0.0
        ax.set_xlim(cx - radius_m, cx + radius_m)
        ax.set_ylim(cy - radius_m, cy + radius_m)

        if title is None:
            title = f"Proposals + DrivingDirectionCompliance | ego_frame={ego_frame}"
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)

        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def debug_viz_proposal_with_ddc_and_route_lanes(
        self,
        proposal_idx: int = 0,
        every_k: int = 1,
        radius_m: float = 12.0,
        ego_frame: bool = False,              # False: GLOBAL, True: EGO frame at time_idx_for_ego_frame
        time_idx_for_ego_frame: int = 0,
        out_path: str = None,
        title: str  = None,
        show_centerline: bool = True,
        show_oncoming_points: bool = True,    # mark points where EgoAreaIndex.ONCOMING_TRAFFIC is True
        annotate: bool = True,                # show DDC + oncoming progress max
        # map drawing
        draw_lane: bool = True,
        draw_lane_connector: bool = True,
        draw_intersection: bool = True,
        max_items: int = 350,
        id_fontsize: int = 7,
        route_lane_color: str = "tab:orange", # on-route lanes will be recolored to this
        route_lane_lw: float = 3.0,
        route_lane_alpha: float = 0.95,
    ):
        """
        Draw ONLY ONE proposal trajectory and its DDC, and draw map lane polygons with on-route lanes highlighted.

        Notes:
        - Uses self._route_lane_ids to decide "on-route".
        - Uses self._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, proposal_idx] for final DDC.
        - Uses self._ego_areas[..., EgoAreaIndex.ONCOMING_TRAFFIC] to mark oncoming points.
        """

        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from shapely.geometry import Point as ShapelyPoint
        from nuplan.common.actor_state.state_representation import StateSE2

        # -------- sanity --------
        if self._ego_coords is None or self._states is None:
            raise ValueError("Call after _reset() & coordinate computation.")
        if self._multi_metrics is None:
            raise ValueError("Call after metrics computed.")
        if self._ego_areas is None:
            raise ValueError("Call after _calculate_ego_area().")
        if self._drivable_area_map is None:
            raise ValueError("self._drivable_area_map is missing.")
        if self._route_lane_ids is None:
            # not fatal, but route highlight will be empty
            self._route_lane_ids = []

        nP = self._ego_coords.shape[0]
        if not (0 <= proposal_idx < nP):
            raise ValueError(f"proposal_idx out of range: {proposal_idx} (nP={nP})")

        # -------- centerline --------
        ls = getattr(self._centerline, "linestring", None) if self._centerline is not None else None

        # -------- ego pose for transforming to ego frame --------
        ego_pose_g = None
        if ego_frame:
            se2 = self._states[proposal_idx, time_idx_for_ego_frame, StateIndex.STATE_SE2]
            ego_pose_g = StateSE2(float(se2[0]), float(se2[1]), float(se2[2]))

            def to_ego_xy(xy_g: np.ndarray) -> np.ndarray:
                dx = xy_g[:, 0] - ego_pose_g.x
                dy = xy_g[:, 1] - ego_pose_g.y
                c = float(np.cos(-ego_pose_g.heading))
                s = float(np.sin(-ego_pose_g.heading))
                x = dx * c - dy * s
                y = dx * s + dy * c
                return np.stack([x, y], axis=1)

        # -------- trajectory (GLOBAL) --------
        centers_g = self._ego_coords[:, :, BBCoordsIndex.CENTER, :]  # [P, T, 2]
        traj_g = centers_g[proposal_idx, ::every_k, :]               # [Ts, 2]
        Ts = traj_g.shape[0]

        # -------- ONCOMING mask (sampled) --------
        oncoming_mask_full = self._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]  # [P, T]
        oncoming_mask = oncoming_mask_full[proposal_idx, ::every_k]               # [Ts]

        # -------- compute the oncoming-progress max (same logic as metric) --------
        # Only for debug display (not used to decide DDC, which is already computed).
        oncoming_progress = np.zeros(Ts, dtype=np.float64)
        if Ts >= 2:
            oncoming_progress[1:] = np.linalg.norm(traj_g[1:] - traj_g[:-1], axis=-1)

        # remove intersection & not-oncoming (same as metric)
        for ti in range(Ts):
            ego_position = ShapelyPoint(float(traj_g[ti, 0]), float(traj_g[ti, 1]))
            try:
                is_in_intersection = self._drivable_area_map.is_in_layer(ego_position, SemanticMapLayer.INTERSECTION)
            except Exception:
                is_in_intersection = False
            if (not bool(oncoming_mask[ti])) or bool(is_in_intersection):
                oncoming_progress[ti] = 0.0

        horizon = int(self._config.driving_direction_horizon / self.proposal_sampling.interval_length)
        # horizon in sampled indices
        horizon_s = max(1, int(round(horizon / max(1, every_k))))
        # rolling sum over [t-horizon_s, t]
        roll = []
        for t in range(Ts):
            roll.append(oncoming_progress[max(0, t - horizon_s): t + 1].sum())
        oncoming_progress_max = float(np.max(roll)) if roll else 0.0

        # -------- final DDC --------
        ddc = float(self._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, proposal_idx])

        # -------- figure --------
        fig, ax = plt.subplots(figsize=(9, 9))

        # centerline
        if show_centerline and ls is not None:
            clx, cly = ls.xy
            cl = np.stack([np.asarray(clx), np.asarray(cly)], axis=1)
            if ego_frame:
                cl = to_ego_xy(cl)
            ax.plot(cl[:, 0], cl[:, 1], linewidth=2.0, alpha=0.9, label="centerline", color="black")

        # map lanes: first draw all lanes (existing helper)
        add_lane_polygons_with_ids_and_arrows(
            ax=ax,
            drivable_area_map=self._drivable_area_map,
            map_api=getattr(self, "_map_api", None),
            ego_pose_se2=ego_pose_g,
            ego_frame=ego_frame,
            radius_m=max(radius_m, 120.0),
            draw_lane=draw_lane,
            draw_lane_connector=draw_lane_connector,
            draw_intersection=draw_intersection,
            max_items=max_items,
            arrow_every=1,
            id_fontsize=id_fontsize,
        )

        # overlay: highlight on-route lanes with distinct color
        # We rely on drivable_area_map.tokens + drivable_area_map.polygons + get_indices_of_map_type.
        try:
            lane_idcs = self._drivable_area_map.get_indices_of_map_type([SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
        except Exception:
            lane_idcs = list(range(len(getattr(self._drivable_area_map, "tokens", []))))

        route_set = set(self._route_lane_ids)
        

        def _poly_iter(poly):
            gt = getattr(poly, "geom_type", "")
            if gt == "Polygon":
                yield poly
            elif gt == "MultiPolygon":
                for g in poly.geoms:
                    yield g

        # ego reference for radius filtering
        if ego_frame:
            ref_x, ref_y = 0.0, 0.0
            ref_pt = ShapelyPoint(0.0, 0.0)
        else:
            ref_x, ref_y = float(traj_g[0, 0]), float(traj_g[0, 1])
            ref_pt = ShapelyPoint(ref_x, ref_y)

        for i in lane_idcs:
            tok = self._drivable_area_map.tokens[i]
            if tok not in route_set:
                continue
            poly_g = self._drivable_area_map._geometries[i]
            if poly_g is None or poly_g.is_empty:
                continue
            poly = _to_ego_frame(poly_g, ego_pose_g) if (ego_frame and ego_pose_g is not None) else poly_g

            # radius gate (centroid)
            c = poly.centroid
            if float(c.distance(ref_pt)) > radius_m:
                continue

            for g in _poly_iter(poly):
                x, y = g.exterior.xy
                ax.plot(x, y, color=route_lane_color, linewidth=route_lane_lw, alpha=route_lane_alpha, zorder=6)

        # trajectory
        traj = to_ego_xy(traj_g) if ego_frame else traj_g
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2.2, alpha=0.95, marker="o", markersize=2.2, label=f"proposal {proposal_idx}")

        # oncoming points
        if show_oncoming_points and np.any(oncoming_mask):
            m = oncoming_mask.astype(bool)
            ax.scatter(traj[m, 0], traj[m, 1], s=28, marker="x", zorder=7, label="ONCOMING_TRAFFIC points")

        # annotation box near start
        if annotate:
            x0, y0 = float(traj[0, 0]), float(traj[0, 1])
            ax.text(
                x0+6.0, y0+6.0,
                f"p={proposal_idx}\nDDC={ddc:.2f}\nmax_oncoming_progress={oncoming_progress_max:.2f} m",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.80, edgecolor="none"),
                zorder=10
            )

        # view window around start
        cx, cy = (0.0, 0.0) if ego_frame else (float(traj[0, 0]), float(traj[0, 1]))
        ax.set_xlim(cx - radius_m, cx + radius_m)
        ax.set_ylim(cy - radius_m, cy + radius_m)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        if title is None:
            title = f"Proposal {proposal_idx} + DDC (ego_frame={ego_frame})"
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)

        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def debug_plot_centerline_and_ego_centers_global(
        self,
        proposal_indices=None,
        every_k: int = 1,
        radius_m: float = 80.0,
        annotate_t: bool = False,
        annotate_step: int = 5,
        show_start_end: bool = True,
        out_path: str = None,
        title: str = None,

        # --- NEW options ---
        show_projection: bool = True,          # show projection (snap) points
        show_proj_lines: bool = True,          # show ego -> proj connecting lines
        proj_every_k: int = 1,                 # projection display sampling (reduce if too dense)
        highlight_jump: bool = True,           # highlight s-jump segments
        jump_threshold_m: float = 10.0,        # jump when |s_t - s_{t-1}| > threshold
    ):
        """
        Visualize centerline (self._centerline.linestring) and ego centers (self._ego_coords[..., CENTER]) in GLOBAL XY.
        Additionally visualize projection (snap) points on the centerline and optional connecting segments.

        Assumptions:
        - self._centerline.linestring is a shapely LineString in GLOBAL frame
        - self._ego_coords are GLOBAL XY coordinates (same frame as centerline)
        """
        if self._centerline is None or getattr(self._centerline, "linestring", None) is None:
            raise ValueError("self._centerline.linestring is missing.")
        if self._ego_coords is None:
            raise ValueError("self._ego_coords is missing.")

        nP = self._ego_coords.shape[0]
        if proposal_indices is None:
            proposal_indices = list(range(nP))
        else:
            proposal_indices = list(proposal_indices)

        # --- centerline ---
        ls = self._centerline.linestring
        cl_x, cl_y = ls.xy
        cl_x = np.asarray(cl_x, dtype=float)
        cl_y = np.asarray(cl_y, dtype=float)

        # --- ego centers (sampled) ---
        centers_full = self._ego_coords[:, :, BBCoordsIndex.CENTER, :]  # [P, T, 2]
        centers = centers_full[:, ::every_k, :]
        Ts = centers.shape[1]

        cx0, cy0 = centers[proposal_indices[0], 0]

        fig, ax = plt.subplots(figsize=(8, 8))

        # centerline
        ax.plot(cl_x, cl_y, linewidth=2.0, label="centerline")

        # ego trajectories + projections
        for p in proposal_indices:
            traj = centers[p]  # [Ts, 2]
            ax.plot(
                traj[:, 0], traj[:, 1],
                marker="o", markersize=1.5, linewidth=1.5, alpha=0.9,
                label=f"ego_center p={p}"
            )

            if show_start_end:
                ax.scatter([traj[0, 0]], [traj[0, 1]], s=20, marker="s")
                ax.scatter([traj[-1, 0]], [traj[-1, 1]], s=20, marker="X")

            if annotate_t:
                for ti in range(0, Ts, annotate_step):
                    ax.text(traj[ti, 0], traj[ti, 1], str(ti * every_k), fontsize=8)

            # ---- projection points on centerline ----
            if show_projection:
                proj_xs, proj_ys = [], []
                s_list = []

                # projection sampling index set
                idxs = list(range(0, Ts, max(1, proj_every_k)))

                for ti in idxs:
                    ex, ey = float(traj[ti, 0]), float(traj[ti, 1])
                    pt = Point(ex, ey)

                    s = float(ls.project(pt))
                    proj_pt = ls.interpolate(s)   # snap point on centerline
                    proj_xs.append(float(proj_pt.x))
                    proj_ys.append(float(proj_pt.y))
                    s_list.append(s)

                    if show_proj_lines:
                        ax.plot([ex, float(proj_pt.x)], [ey, float(proj_pt.y)],
                                linewidth=0.8, alpha=0.6)  # default color (matplotlib cycle)

                # projection points scatter
                ax.scatter(proj_xs, proj_ys, s=12, marker="x", label=f"proj_on_centerline p={p}")

                # ---- highlight s-jumps (optional) ----
                if highlight_jump and len(s_list) >= 2:
                    for k in range(1, len(s_list)):
                        if abs(s_list[k] - s_list[k-1]) > jump_threshold_m:
                            # draw the two ego points and two proj points of the jump segment thicker
                            t_prev = idxs[k-1]
                            t_now  = idxs[k]
                            ex0, ey0 = float(traj[t_prev, 0]), float(traj[t_prev, 1])
                            ex1, ey1 = float(traj[t_now, 0]),  float(traj[t_now, 1])
                            px0, py0 = proj_xs[k-1], proj_ys[k-1]
                            px1, py1 = proj_xs[k],   proj_ys[k]

                            # highlight: ego segment
                            ax.plot([ex0, ex1], [ey0, ey1], linewidth=3.0, alpha=0.9)
                            # highlight: proj segment ("jump" along the centerline)
                            ax.plot([px0, px1], [py0, py1], linewidth=3.0, alpha=0.9)

                            # jump magnitude as text
                            mx, my = 0.5*(ex0+ex1), 0.5*(ey0+ey1)
                            ax.text(mx, my, f"Δs={s_list[k]-s_list[k-1]:+.1f}m",
                                    fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"))

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(cx0 - radius_m, cx0 + radius_m)
        ax.set_ylim(cy0 - radius_m, cy0 + radius_m)

        if title is None:
            title = f"GLOBAL: centerline + ego centers | proposals={proposal_indices} | every_k={every_k}"
        ax.set_title(title)
        ax.legend(loc="best")

        if out_path is not None:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def _debug_print_objects_within_radius(
        self,
        proposal_idx: int,
        time_idx: int,
        radius_m: float = 4.0,
        topk: int = 50,
        only_agents: bool = False,
    ) -> None:
        """
        Print observation objects within `radius_m` from ego center at (proposal_idx, time_idx).
        Assumes both ego center and observation geometries are in GLOBAL XY.
        """
        obs_t = self._observation[time_idx]

        # ego center (global)
        ego_xy = self._ego_coords[proposal_idx, time_idx, BBCoordsIndex.CENTER]
        ex, ey = float(ego_xy[0]), float(ego_xy[1])

        rows = []
        tokens = list(obs_t.tokens)

        for tok in tokens:
            geom = obs_t[tok]  # shapely geometry
            if geom is None:
                continue

            c = geom.centroid
            dx = float(c.x) - ex
            dy = float(c.y) - ey
            dist = (dx * dx + dy * dy) ** 0.5

            if dist > radius_m:
                continue

            # tracked object type (if exists)
            ttype = None
            if tok in self._observation.unique_objects:
                tobj = self._observation.unique_objects[tok]
                ttype = getattr(tobj, "tracked_object_type", None)

                if only_agents and (ttype not in AGENT_TYPES):
                    continue
            else:
                if only_agents:
                    continue

            b = geom.bounds  # (minx, miny, maxx, maxy)
            rows.append((dist, tok, ttype, (float(c.x), float(c.y)), b))

        rows.sort(key=lambda x: x[0])

        # print(
        #     f"\n[DEBUG obs within {radius_m:.1f}m] proposal_idx={proposal_idx}, time_idx={time_idx}"
        #     f" | ego_center=({ex:.3f},{ey:.3f}) | #near={len(rows)} / #total={len(tokens)}"
        #     f" | only_agents={only_agents}"
        # )
        # for i, (dist, tok, ttype, cent, b) in enumerate(rows[:topk]):
        #     minx, miny, maxx, maxy = b
        #     print(
        #         f"{i:02d} dist={dist:5.2f}  type={ttype}  token={tok}  "
        #         f"cent=({cent[0]:.3f},{cent[1]:.3f})  "
        #         f"bounds=({minx:.3f},{miny:.3f},{maxx:.3f},{maxy:.3f})"
        #     )
    
    def _debug_save_global_scene(
        self,
        proposal_idx: int,
        time_idx: int,
        highlight_tokens: Optional[List[str]] = None,
        out_dir: str = "exp_new_spli/collision_test/global_scene_viz",
        radius_m: float = 20.0,
        max_objects: int = 300,
    ) -> None:
        """
        Save a GLOBAL XY plot:
          - ego polygon (global)
          - all observation geometries (global)
          - highlight certain tokens in red
        """
        os.makedirs(out_dir, exist_ok=True)
        if highlight_tokens is None:
            highlight_tokens = []

        obs_t = self._observation[time_idx]
        ego_poly = self._ego_polygons[proposal_idx, time_idx]  # global shapely polygon

        # ego center (global)
        ego_xy = self._ego_coords[proposal_idx, time_idx, BBCoordsIndex.CENTER]
        # print(ego_xy)
        ex, ey = float(ego_xy[0]), float(ego_xy[1])


        

        # gather candidates (within radius to keep plot readable)
        rows = []
        for tok in list(obs_t.tokens):
            geom = obs_t[tok]
            if geom is None:
                continue
            c = geom.centroid
            dx = float(c.x) - ex
            dy = float(c.y) - ey
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= radius_m:
                rows.append((dist, tok, geom))
        rows.sort(key=lambda x: x[0])
        rows = rows[:max_objects]

        fig, ax = plt.subplots(figsize=(7, 7))

        # ego pose (GLOBAL) at this time_idx for this proposal
        ego_state_se2 = self._states[proposal_idx, time_idx, StateIndex.STATE_SE2]  # [x, y, heading]
        eh = float(ego_state_se2[2])

        # draw ego heading arrow + text at ego center
        _draw_heading_arrow(
            ax=ax,
            x=ex,
            y=ey,
            heading=eh,
            length=6.0,
            color="tab:blue",
            text_prefix="ego_heading(G)",
        )

        # plot all objects (thin)
        for dist, tok, geom in rows:
            if tok in highlight_tokens:
                continue
            _plot_shapely(ax, geom, color="0.5", linewidth=1.0, alpha=0.6)

        # plot highlights (thick red)
        for tok in highlight_tokens:
            if tok in obs_t.tokens:
                geom = obs_t[tok]
                _plot_shapely(ax, geom, color="red", linewidth=2.5, alpha=0.95)

        # plot ego polygon (blue thick)
        _plot_shapely(ax, ego_poly, color="blue", linewidth=2.5, alpha=0.95)

        # also mark ego center
        ax.scatter([ex], [ey], s=30)

        # zoom: ego + nearest highlighted if exists else ego bounds
        minx, miny, maxx, maxy = ego_poly.bounds
        if highlight_tokens:
            for tok in highlight_tokens:
                if tok in obs_t.tokens:
                    b = obs_t[tok].bounds
                    minx = min(minx, b[0]); miny = min(miny, b[1])
                    maxx = max(maxx, b[2]); maxy = max(maxy, b[3])

        pad = max(2.0, radius_m * 0.2)
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        ax.set_title(f"GLOBAL scene | proposal={proposal_idx} t={time_idx} | #objs={len(rows)} | r={radius_m}m")
        fname = f"global_p{proposal_idx}_t{time_idx}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)
    
    def _debug_save_ego_scene(
        self,
        proposal_idx: int,
        time_idx: int,
        highlight_tokens: Optional[List[str]] = None,
        out_dir: str = "exp/debug_viz/ego_scene_viz",
        radius_m: float = 20.0,
        max_objects: int = 300,
        only_agents: bool = False,   # True: draw only AGENT_TYPES
    ) -> None:
        """
        Save an EGO-FRAME plot:
        - ego polygon (ego frame)
        - all observation geometries transformed GLOBAL->EGO
        - highlight certain tokens in red (also transformed)
        EGO pose is taken from self._states[proposal_idx, time_idx, StateIndex.STATE_SE2].
        """
        os.makedirs(out_dir, exist_ok=True)
        if highlight_tokens is None:
            highlight_tokens = []

        obs_t = self._observation[time_idx]

        # ego pose (GLOBAL) at this proposal/time
        ego_state_se2 = self._states[proposal_idx, time_idx, StateIndex.STATE_SE2]
        ego_pose_g = StateSE2(float(ego_state_se2[0]), float(ego_state_se2[1]), float(ego_state_se2[2]))

        # ego polygon (GLOBAL -> EGO)
        ego_poly_g = self._ego_polygons[proposal_idx, time_idx]
        ego_poly_e = _shapely_to_ego_frame(ego_poly_g, ego_pose_g)

        # gather candidates within radius (in EGO frame) to keep plot readable
        rows = []
        for tok in list(obs_t.tokens):
            geom_g = obs_t[tok]
            if geom_g is None:
                continue

            # optional agent-only filter
            if only_agents:
                if tok not in self._observation.unique_objects:
                    continue
                ttype = getattr(self._observation.unique_objects[tok], "tracked_object_type", None)
                if ttype not in AGENT_TYPES:
                    continue

            geom_e = _shapely_to_ego_frame(geom_g, ego_pose_g)
            if geom_e is None:
                continue

            c = geom_e.centroid
            dist = float(np.hypot(c.x, c.y))  # distance in ego frame
            if dist <= radius_m:
                rows.append((dist, tok, geom_e))

        rows.sort(key=lambda x: x[0])
        rows = rows[:max_objects]

        fig, ax = plt.subplots(figsize=(7, 7))

        # plot all objects (thin gray), excluding highlights
        for dist, tok, geom_e in rows:
            if tok in highlight_tokens:
                continue
            _plot_shapely(ax, geom_e, color="0.5", linewidth=1.0, alpha=0.6)

        # plot highlights (thick red)
        for tok in highlight_tokens:
            if tok in obs_t.tokens:
                geom_g = obs_t[tok]
                geom_e = _shapely_to_ego_frame(geom_g, ego_pose_g)
                _plot_shapely(ax, geom_e, color="red", linewidth=2.5, alpha=0.95)

        # plot ego polygon (blue thick)
        _plot_shapely(ax, ego_poly_e, color="blue", linewidth=2.5, alpha=0.95)

        # mark ego origin
        ax.scatter([0.0], [0.0], s=30)

        # view window: ego frame radius
        pad = max(2.0, radius_m * 0.2)
        ax.set_xlim(-radius_m - pad, radius_m + pad)
        ax.set_ylim(-radius_m - pad, radius_m + pad)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        ax.set_title(f"EGO scene | proposal={proposal_idx} t={time_idx} | #objs={len(rows)} | r={radius_m}m")
        fname = f"ego_p{proposal_idx}_t{time_idx}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _debug_save_all_timesteps_for_collision(
        self,
        proposal_idx: int,
        token: str,
        out_dir: str = "exp_debug/collision_all_timesteps",
        radius_m: float = 30.0,
        use_ego_frame: bool = True,   # True: ego frame, False: global frame
        only_agents: bool = True,
    ) -> None:
        """
        For a proposal where a collision occurred, visualize the ego polygon + colliding agent polygon at every time step.
        Save a separate PNG for each time step.
        If the token is not in that time step's observation, draw only the ego without any highlight.
        """
        os.makedirs(out_dir, exist_ok=True)
        T = self.proposal_sampling.num_poses + 1

        for time_idx in range(T):
            obs_t = self._observation[time_idx]
            highlight = [token] if token in obs_t.token_to_idx else []

            if use_ego_frame:
                self._debug_save_ego_scene(
                    proposal_idx=proposal_idx,
                    time_idx=time_idx,
                    highlight_tokens=highlight,
                    out_dir=out_dir,
                    radius_m=radius_m,
                    max_objects=300,
                    only_agents=only_agents,
                )
            else:
                self._debug_save_global_scene(
                    proposal_idx=proposal_idx,
                    time_idx=time_idx,
                    highlight_tokens=highlight,
                    out_dir=out_dir,
                    radius_m=radius_m,
                    max_objects=300,
                )

    def _save_collision_debug_viz(
        self,
        proposal_idx: int,
        time_idx: int,
        token: str,
        out_dir: str,
        pad: float = 2.0,
    ) -> None:
        """
        Save both GLOBAL and EGO-frame plots for ego polygon vs collided object geometry.
        EGO-frame is centered at ego pose at (proposal_idx, time_idx) in GLOBAL coordinates.
        """
        os.makedirs(out_dir, exist_ok=True)

        ego_poly_g = self._ego_polygons[proposal_idx, time_idx]
        obj_geom_g = self._observation[time_idx][token]

        # ego pose (GLOBAL) at this time_idx for this proposal
        # self._states[..., StateIndex.STATE_SE2] = [x, y, heading] in GLOBAL
        ego_state_se2 = self._states[proposal_idx, time_idx, StateIndex.STATE_SE2]
        ego_pose_g = StateSE2(float(ego_state_se2[0]), float(ego_state_se2[1]), float(ego_state_se2[2]))

        # Transform to ego frame
        ego_poly_e = _shapely_to_ego_frame(ego_poly_g, ego_pose_g)
        obj_geom_e = _shapely_to_ego_frame(obj_geom_g, ego_pose_g)

        # filenames
        short = token[:8]
        base = f"coll_p{proposal_idx}_t{time_idx}_{short}"

        # ----- GLOBAL plot -----
        fig, ax = plt.subplots(figsize=(6, 6))
        _plot_shapely(ax, ego_poly_g, color="red", linewidth=2, label="ego_poly(G)")
        _plot_shapely(ax, obj_geom_g, color="blue", linewidth=2, label="obj_geom(G)")
        minx = min(ego_poly_g.bounds[0], obj_geom_g.bounds[0])
        miny = min(ego_poly_g.bounds[1], obj_geom_g.bounds[1])
        maxx = max(ego_poly_g.bounds[2], obj_geom_g.bounds[2])
        maxy = max(ego_poly_g.bounds[3], obj_geom_g.bounds[3])
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f"GLOBAL | p={proposal_idx} t={time_idx} token={short}")
        fig.savefig(os.path.join(out_dir, base + "_GLOBAL.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # ----- EGO-FRAME plot -----
        fig, ax = plt.subplots(figsize=(6, 6))
        _plot_shapely(ax, ego_poly_e, color="red", linewidth=2, label="ego_poly(E)")
        _plot_shapely(ax, obj_geom_e, color="blue", linewidth=2, label="obj_geom(E)")
        # zoom based on transformed bounds
        minx = min(ego_poly_e.bounds[0], obj_geom_e.bounds[0])
        miny = min(ego_poly_e.bounds[1], obj_geom_e.bounds[1])
        maxx = max(ego_poly_e.bounds[2], obj_geom_e.bounds[2])
        maxy = max(ego_poly_e.bounds[3], obj_geom_e.bounds[3])
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f"EGO@t | p={proposal_idx} t={time_idx} ego=(0,0)")
        fig.savefig(os.path.join(out_dir, base + "_EGO.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Optional: print quick overlap status in EGO frame
        try:
            inter_area = ego_poly_e.intersection(obj_geom_e).area
            # print(f"[DBG] EGO-frame intersection area = {inter_area:.6f}")
        except Exception as e:
            pass
            # print(f"[DBG] intersection area compute failed: {e}")


    ##################################################





    def time_to_at_fault_collision(self, proposal_idx: int) -> float:
        """
        Returns time to at-fault collision for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        return self._collision_time_idcs[proposal_idx] * self.proposal_sampling.interval_length

    def time_to_ttc_infraction(self, proposal_idx: int) -> float:
        """
        Returns time to ttc infraction for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        return self._ttc_time_idcs[proposal_idx] * self.proposal_sampling.interval_length

    def score_proposals(
        self,
        states: npt.NDArray[np.float64],
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_ids: List[str],
        drivable_area_map: PDMDrivableMap,
        map_parameters: Optional[MapParameters] = None,
        simulated_agent_detections_tracks: Optional[List[DetectionsTracks]] = None,
        human_past_trajectory: Optional[InterpolatedTrajectory] = None,
        debug=False, 
    ) -> List[pd.DataFrame]:
        self.debug = debug
        """
        TODO: Update this docstring
        Scores proposal similar to nuPlan's closed-loop metrics
        :param states: array representation of simulated proposals
        :param observation: PDM's observation class
        :param centerline: path of the centerline
        :param route_lane_ids: list containing on-route lane ids
        :param drivable_area_map: Occupancy map of drivable are polygons
        :return: A List containing the PDMResult for each proposal
        """
        
        if simulated_agent_detections_tracks is not None:
            # pass
            observation.update_detections_tracks(
                detection_tracks=simulated_agent_detections_tracks,
            )

        # initialize & lazy load class values
        self._reset(
            states,
            observation,
            centerline,
            route_lane_ids,
            drivable_area_map,
            human_past_trajectory,
        )

        # fill value ego-area array (used in multiple metrics)
        self._calculate_ego_area()

        # 1. multiplicative metrics
        self._calculate_no_at_fault_collision()
        self._calculate_drivable_area_compliance()
        
        if not self._config.use_pdms_v1:
            self._calculate_traffic_light_compliance()
            self._calculate_driving_direction_compliance()

        # 2. weighted metrics
        self._calculate_progress()
        self._calculate_ttc()
        if self._config.use_pdms_v1:
            self._calculate_comfort()
        else:
            self._calculate_lane_keeping()
            self._calculate_history_comfort()

        pdm_scores = self._aggregate_pdm_scores()
        multiplicative_metrics_prods, weighted_metrics_all = self._multi_metrics.prod(axis=0), self._weighted_metrics

        results: List[pd.DataFrame] = []
        for proposal_idx in range(self._num_proposals):

            no_at_fault_collisions = self._multi_metrics[MultiMetricIndex.NO_COLLISION, proposal_idx]
            drivable_area_compliance = self._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, proposal_idx]
            driving_direction_compliance = self._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, proposal_idx] if not self._config.use_pdms_v1 else 1.0
            traffic_light_compliance = self._multi_metrics[MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE, proposal_idx] if not self._config.use_pdms_v1 else 1.0

            ego_progress = self._weighted_metrics[WeightedMetricIndex.PROGRESS, proposal_idx]
           
            time_to_collision_within_bound = self._weighted_metrics[WeightedMetricIndex.TTC, proposal_idx]
            lane_keeping = self._weighted_metrics[WeightedMetricIndex.LANE_KEEPING, proposal_idx]  # comfort in v1 mode
            history_comfort = self._weighted_metrics[WeightedMetricIndex.HISTORY_COMFORT, proposal_idx] if not self._config.use_pdms_v1 else 0.0

            multiplicative_metrics_prod = multiplicative_metrics_prods[proposal_idx]
            weighted_metrics = weighted_metrics_all[:, proposal_idx]
            pdm_score = pdm_scores[proposal_idx]

            results.append(
                pd.DataFrame(
                    [
                        PDMResults(
                            no_at_fault_collisions=no_at_fault_collisions,
                            drivable_area_compliance=drivable_area_compliance,
                            driving_direction_compliance=driving_direction_compliance,
                            traffic_light_compliance=traffic_light_compliance,
                            ego_progress=ego_progress,
                            time_to_collision_within_bound=time_to_collision_within_bound,
                            lane_keeping=lane_keeping,
                            history_comfort=history_comfort,
                            multiplicative_metrics_prod=multiplicative_metrics_prod,
                            weighted_metrics=weighted_metrics,
                            weighted_metrics_array=self._config.weighted_metrics_array,
                            pdm_score=pdm_score,
                        )
                    ]
                )
            )
        return results

    def _aggregate_pdm_scores(self) -> npt.NDArray[np.float64]:
        """
        Score for PDM proposals.
        - PDMS v1: (NC × DAC) × (EP + TTC + Comfort) / 3
        - EPDMS v2: (NC × DAC × DDC × TLC) × (EP + TTC + HC + LK + EC) / sum(weights)
        """

        # accumulate multiplicative metrics
        if self._config.use_pdms_v1:
            multiplicate_metric_scores = (
                self._multi_metrics[MultiMetricIndex.NO_COLLISION]
                * self._multi_metrics[MultiMetricIndex.DRIVABLE_AREA]
            )
        else:
            multiplicate_metric_scores = self._multi_metrics.prod(axis=0)

        # normalize and fill progress values
        
        masked_progress = self._progress_raw * multiplicate_metric_scores
        norm_constant_progress = np.max(masked_progress)
        if norm_constant_progress > self._config.progress_distance_threshold:
            normalized_progress = np.clip(self._progress_raw / norm_constant_progress, 0.0, 1.0)
        else:
            normalized_progress = np.ones(len(masked_progress), dtype=np.float64)
        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = normalized_progress

        if self._config.use_pdms_v1:
            # PDMS v1: (NC × DAC) × (EP + TTC + Comfort) / 3
            weighted_metrics_array = self._config.weighted_metrics_array
            # For v1, only use PROGRESS, TTC, and LANE_KEEPING (which holds comfort)
            mask = np.zeros_like(weighted_metrics_array, dtype=bool)
            mask[WeightedMetricIndex.PROGRESS] = True
            mask[WeightedMetricIndex.TTC] = True
            mask[WeightedMetricIndex.LANE_KEEPING] = True  # Comfort is stored here in v1 mode
            
            weighted_metric_scores = (self._weighted_metrics[mask] * weighted_metrics_array[mask, None]).sum(axis=0)
            weighted_metric_scores /= weighted_metrics_array[mask].sum()
        else:
            # EPDMS v2: Exclude the two-frame extended comfort metric from the weighted metrics calculation
            mask = np.ones_like(self._config.weighted_metrics_array, dtype=bool)
            mask[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = False

            weighted_metrics_array = self._config.weighted_metrics_array
            weighted_metric_scores = (self._weighted_metrics[mask] * weighted_metrics_array[mask, None]).sum(axis=0)
            weighted_metric_scores /= weighted_metrics_array[mask].sum()

        # calculate final scores
        final_scores = multiplicate_metric_scores * weighted_metric_scores

        return final_scores

    def _reset(
        self,
        states: npt.NDArray[np.float64],
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_ids: List[str],
        drivable_area_map: PDMDrivableMap,
        human_past_trajectory: Optional[InterpolatedTrajectory],
    ) -> None:
        """
        Resets metric values and lazy loads input classes.
        :param states: array representation of simulated proposals
        :param observation: PDM's observation class
        :param centerline: path of the centerline
        :param route_lane_ids: list containing on-route lane ids
        :param drivable_area_map: Occupancy map of drivable are polygons
        """
        assert states.ndim == 3
        assert states.shape[1] == self.proposal_sampling.num_poses + 1
        assert states.shape[2] == StateIndex.size()

        self._observation = observation
        self._centerline = centerline
        self._route_lane_ids = route_lane_ids
        self._drivable_area_map = drivable_area_map
        self._human_past_trajectory = human_past_trajectory

        self._num_proposals = states.shape[0]

        # save ego state values
        self._states = states

        # calculate coordinates of ego corners and center
        self._ego_coords = state_array_to_coords_array(states, self._vehicle_parameters)

        # initialize all ego polygons from corners
        self._ego_polygons = coords_array_to_polygon_array(self._ego_coords)

        # zero initialize all remaining arrays.
        self._ego_areas = np.zeros(
            (
                self._num_proposals,
                self.proposal_sampling.num_poses + 1,
                len(EgoAreaIndex),
            ),
            dtype=np.bool_,
        )
        self._multi_metrics = np.zeros((len(MultiMetricIndex), self._num_proposals), dtype=np.float64)
        self._weighted_metrics = np.zeros((len(WeightedMetricIndex), self._num_proposals), dtype=np.float64)
        self._progress_raw = np.zeros(self._num_proposals, dtype=np.float64)

        # initialize infraction arrays with infinity (meaning no infraction occurs)
        self._collision_time_idcs = np.zeros(self._num_proposals, dtype=np.float64)
        self._ttc_time_idcs = np.zeros(self._num_proposals, dtype=np.float64)
        self._collision_time_idcs.fill(np.inf)
        self._ttc_time_idcs.fill(np.inf)

    def _calculate_ego_area(self) -> None:
        """
        Determines the area of proposals over time.
        Areas are (1) in multiple lanes, (2) non-drivable area, or (3) oncoming traffic
        """

        n_proposals, n_horizon, n_points, _ = self._ego_coords.shape

        in_polygons = self._drivable_area_map.points_in_polygons(self._ego_coords)
        in_polygons = in_polygons.transpose(1, 2, 0, 3)  # shape: n_proposals, n_horizon, n_polygons, n_points

        drivable_area_idcs = self._drivable_area_map.get_indices_of_map_type(
            [
                SemanticMapLayer.ROADBLOCK,
                SemanticMapLayer.INTERSECTION,
                SemanticMapLayer.DRIVABLE_AREA,
                SemanticMapLayer.CARPARK_AREA,
                SemanticMapLayer.LANE,
            ]
        )

        drivable_lane_idcs = self._drivable_area_map.get_indices_of_map_type(
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        )

        drivable_on_route_idcs: List[int] = [
            idx for idx in drivable_lane_idcs if self._drivable_area_map.tokens[idx] in self._route_lane_ids
        ]  # index mask for on-route lanes

        corners_in_polygon = in_polygons[..., :-1]  # ignore center coordinate
        center_in_polygon = in_polygons[..., -1]  # only center

        # in_multiple_lanes: if
        # - more than one drivable polygon contains at least one corner
        # - no polygon contains all corners
        batch_multiple_lanes_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_multiple_lanes_mask = (corners_in_polygon[:, :, drivable_lane_idcs].sum(axis=-1) > 0).sum(axis=-1) > 1

        batch_not_single_lanes_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_not_single_lanes_mask = np.all(corners_in_polygon[:, :, drivable_lane_idcs].sum(axis=-1) != 4, axis=-1)

        multiple_lanes_mask = np.logical_and(batch_multiple_lanes_mask, batch_not_single_lanes_mask)
        self._ego_areas[multiple_lanes_mask, EgoAreaIndex.MULTIPLE_LANES] = True

        # in_nondrivable_area: center-based (relaxed) or corner-based (strict)
        batch_nondrivable_area_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        
        if self._config.dac_use_center:
            # relaxed: violation if center is outside the drivable area
            batch_nondrivable_area_mask = center_in_polygon[:, :, drivable_area_idcs].sum(axis=-1) == 0
        else:
            # original: violation if any of the 4 corners is outside the drivable area
            batch_nondrivable_area_mask = (corners_in_polygon[:, :, drivable_area_idcs].sum(axis=-2) > 0).sum(axis=-1) < 4
        self._ego_areas[batch_nondrivable_area_mask, EgoAreaIndex.NON_DRIVABLE_AREA] = True

        # in_oncoming_traffic: if center not in any drivable polygon that is on-route
        batch_oncoming_traffic_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_oncoming_traffic_mask = center_in_polygon[..., drivable_on_route_idcs].sum(axis=-1) == 0
        self._ego_areas[batch_oncoming_traffic_mask, EgoAreaIndex.ONCOMING_TRAFFIC] = True

    def _calculate_no_at_fault_collision(self) -> None:
        """
        Re-implementation of nuPlan's at-fault collision metric.
        """
        no_at_fault_collision_scores = np.ones(self._num_proposals, dtype=np.float64)

        proposal_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        # (proposal_idx, token) pairs where at-fault collision was detected
        _at_fault_pairs: set = set()

        for time_idx in range(self.proposal_sampling.num_poses + 1):
            ego_polygons = self._ego_polygons[:, time_idx]
            intersecting = self._observation[time_idx].query(ego_polygons, predicate="intersects")
            
            
            if len(intersecting) == 0:
                continue

            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                token = self._observation[time_idx].tokens[geometry_idx]
                if (self._observation.red_light_token in token) or (token in proposal_collided_track_ids[proposal_idx]):
                    continue

                ego_in_multiple_lanes_or_nondrivable_area = (
                    self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                    or self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
                )

                tracked_object = self._observation.unique_objects[token]

                # classify collision
                collision_type: CollisionType = get_collision_type(
                    self._states[proposal_idx, time_idx],
                    self._ego_polygons[proposal_idx, time_idx],
                    tracked_object,
                    self._observation[time_idx][token],
                )
                
                collisions_at_stopped_track_or_active_front: bool = collision_type in [
                    CollisionType.ACTIVE_FRONT_COLLISION,
                    CollisionType.STOPPED_TRACK_COLLISION,
                ]
                collision_at_lateral: bool = collision_type == CollisionType.ACTIVE_LATERAL_COLLISION

                # self._debug_save_global_scene(
                #     proposal_idx=proposal_idx,
                #     time_idx=time_idx,
                #     highlight_tokens=[token],
                #     radius_m=100,
                #     max_objects=1000,
                # )
                # self._debug_save_ego_scene(
                #     proposal_idx=proposal_idx,
                #     time_idx=time_idx,
                #     highlight_tokens=[token],
                #     radius_m=30,
                #     max_objects=1000,
                #     only_agents=True,
                # )
                
                # 1. at fault collision
                if collisions_at_stopped_track_or_active_front or (
                    ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
                ):
                    no_at_fault_collision_score = 0.0 if tracked_object.tracked_object_type in AGENT_TYPES else 0.5
                    no_at_fault_collision_scores[proposal_idx] = np.minimum(
                        no_at_fault_collision_scores[proposal_idx],
                        no_at_fault_collision_score,
                    )
                    self._collision_time_idcs[proposal_idx] = min(time_idx, self._collision_time_idcs[proposal_idx])
                    _at_fault_pairs.add((proposal_idx, token))

                else:  # 2. no at fault collision
                    proposal_collided_track_ids[proposal_idx].append(token)

        # DEBUG: for each at-fault (proposal, token) pair, save all time steps
        if False:
            for p_idx, tok in _at_fault_pairs:
                short = tok[:8]
                out_dir = os.path.join(
                    "exp_newspli/collision_all_timesteps",
                    f"p{p_idx}_{short}",
                )
                use_ego = False  # for now, visualize all time steps in the global frame
                self._debug_save_all_timesteps_for_collision(
                    proposal_idx=p_idx,
                    token=tok,
                    out_dir=out_dir,
                    radius_m=30.0,
                    use_ego_frame=use_ego,
                    only_agents=True,
                )
            pass

        self._multi_metrics[MultiMetricIndex.NO_COLLISION] = no_at_fault_collision_scores

    def _calculate_drivable_area_compliance(self) -> None:
        """
        Re-implementation of nuPlan's drivable area compliance metric
        """

        drivable_area_compliance_scores = np.ones(self._num_proposals, dtype=np.float64)
        off_road_mask = self._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA].any(axis=-1)

        drivable_area_compliance_scores[off_road_mask] = 0.0
        self._multi_metrics[MultiMetricIndex.DRIVABLE_AREA] = drivable_area_compliance_scores

    def _calculate_driving_direction_compliance(self) -> None:
        """
        Re-implementation of nuPlan's driving direction compliance metric
        """
        center_coordinates = self._ego_coords[:, :, BBCoordsIndex.CENTER]
        oncoming_progress = np.zeros(
            (self._num_proposals, self.proposal_sampling.num_poses + 1),
            dtype=np.float64,
        )
        oncoming_progress[:, 1:] = np.linalg.norm(center_coordinates[:, 1:] - center_coordinates[:, :-1], axis=-1)

        # mask out points that are not in oncoming traffic
        oncoming_traffic_masks = self._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]

        # remove intersection
        for proposal_idx in range(self._num_proposals):
            for time_idx in range(self.proposal_sampling.num_poses + 1):
                ego_position = Point(*center_coordinates[proposal_idx, time_idx])
                is_in_intersection = self._drivable_area_map.is_in_layer(ego_position, SemanticMapLayer.INTERSECTION)
                if not oncoming_traffic_masks[proposal_idx, time_idx] or is_in_intersection:
                    oncoming_progress[proposal_idx, time_idx] = 0.0

        # aggregate
        driving_direction_compliance_scores = np.ones(self._num_proposals, dtype=np.float64)
        horizon = int(self._config.driving_direction_horizon / self.proposal_sampling.interval_length)

        oncoming_progress_over_horizon = np.concatenate(
            [
                oncoming_progress[:, max(0, time_idx - horizon) : time_idx + 1].sum(axis=-1)[..., None]
                for time_idx in range(oncoming_progress.shape[-1])
            ],
            dtype=np.float64,
            axis=-1,
        )

        for proposal_idx, progress in enumerate(oncoming_progress_over_horizon.max(axis=-1)):
            if progress < self._config.driving_direction_compliance_threshold:
                driving_direction_compliance_scores[proposal_idx] = 1.0
            elif progress < self._config.driving_direction_violation_threshold:
                driving_direction_compliance_scores[proposal_idx] = 0.5
            else:
                driving_direction_compliance_scores[proposal_idx] = 0.0

        self._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION] = driving_direction_compliance_scores

    def _calculate_progress(self) -> None:
        """
        Re-implementation of nuPlan's progress metric (non-normalized).
        Calculates progress along the centerline.
        """

        # calculate raw progress in meter
        progress_in_meter = np.zeros(self._num_proposals, dtype=np.float64)
        for proposal_idx in range(self._num_proposals):
            start_point = Point(*self._ego_coords[proposal_idx, 0, BBCoordsIndex.CENTER])
            end_point = Point(*self._ego_coords[proposal_idx, -1, BBCoordsIndex.CENTER])
            progress = self._centerline.project([start_point, end_point])
            progress_in_meter[proposal_idx] = progress[1] - progress[0]
        
        if False:
            # if progress_in_meter[1] > 3:
            # print(f"start_poing: {start_point}, end_point: {end_point}")
            # print(f"progress: {progress}")
            # print(progress_in_meter[1])
            
            # self.debug_plot_centerline_and_ego_centers_global(
            #     proposal_indices=[1],
            #     every_k=1,
            #     radius_m=80,
            #     out_path="exp2/debug_centerline/p0.png",
            # )
            # print(type(self._centerline.linestring), self._centerline.linestring.geom_type)
            # print("is_ring:", getattr(self._centerline.linestring, "is_ring", None))
            # print("is_closed:", self._centerline.linestring.is_closed if hasattr(self._centerline.linestring, "is_closed") else None)
            # print("n_coords:", len(self._centerline.linestring.coords))
            self.debug_plot_centerline_and_ego_centers_global(
                proposal_indices=[0, 1],
                every_k=1,
                proj_every_k=2,          # plot projection every 2 steps
                show_projection=False,
                show_proj_lines=True,
                highlight_jump=False,
                jump_threshold_m=10.0,
                radius_m=120,
                out_path="exp_debug/debug_plot_centerline_and_ego_centers_global.png",
            )

        self._progress_raw = np.clip(progress_in_meter, a_min=0, a_max=None)

    def _calculate_ttc(self):
        """
        Re-implementation of nuPlan's time-to-collision metric.
        """

        ttc_scores = np.ones(self._num_proposals, dtype=np.float64)
        temp_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        # Sample TTC checks approximately every 0.3s up to the configured horizon.
        # This keeps behavior close to the original 10Hz implementation, while being
        # robust to arbitrary proposal sampling intervals (e.g., 0.1s or 0.5s).
        proposal_dt = float(self.proposal_sampling.interval_length)
        max_future_step = int(np.floor(self._config.future_collision_horizon_window / proposal_dt))
        max_future_step = min(max_future_step, self.proposal_sampling.num_poses)
        ttc_stride = max(1, int(round(0.3 / proposal_dt)))
        future_time_idcs = np.arange(0, max_future_step + 1, ttc_stride, dtype=np.int64)
        if future_time_idcs[-1] != max_future_step:
            future_time_idcs = np.append(future_time_idcs, max_future_step)
        n_future_steps = len(future_time_idcs)

        # create polygons for each ego position and specific time horizon (default:1s) future projection
        coords_exterior = self._ego_coords.copy()
        coords_exterior[:, :, BBCoordsIndex.CENTER, :] = coords_exterior[:, :, BBCoordsIndex.FRONT_LEFT, :]
        coords_exterior_time_steps = np.repeat(coords_exterior[:, :, None], n_future_steps, axis=2)

        speeds = np.hypot(
            self._states[..., StateIndex.VELOCITY_X],
            self._states[..., StateIndex.VELOCITY_Y],
        )

        dxy_per_s = np.stack(
            [
                np.cos(self._states[..., StateIndex.HEADING]) * speeds,
                np.sin(self._states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )

        for idx, future_time_idx in enumerate(future_time_idcs):
            delta_t = float(future_time_idx) * self.proposal_sampling.interval_length
            coords_exterior_time_steps[:, :, idx] = (
                coords_exterior_time_steps[:, :, idx] + dxy_per_s[:, :, None] * delta_t
            )

        polygons = creation.polygons(coords_exterior_time_steps)

        # ttc needs to look future_time_idcs into the future,
        # so we can only calculate it for n_proposal_steps_to_evaluate steps

        n_proposal_steps_to_evaluate = self.proposal_sampling.num_poses - int(np.max(future_time_idcs))
        # check collision for each proposal and projection
        for time_idx in range(n_proposal_steps_to_evaluate + 1):
            for step_idx, future_time_idx in enumerate(future_time_idcs):
                current_time_idx = time_idx + future_time_idx
                polygons_at_time_step = polygons[:, time_idx, step_idx]
                intersecting = self._observation[current_time_idx].query(polygons_at_time_step, predicate="intersects")

                if len(intersecting) == 0:
                    continue

                for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                    token = self._observation[current_time_idx].tokens[geometry_idx]
                    if (
                        (self._observation.red_light_token in token)
                        or (token in temp_collided_track_ids[proposal_idx])
                        or (speeds[proposal_idx, time_idx] < self._config.stopped_speed_threshold)
                    ):
                        continue

                    ego_in_multiple_lanes_or_nondrivable_area = (
                        self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                        or self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
                    )
                    ego_rear_axle: StateSE2 = StateSE2(*self._states[proposal_idx, time_idx, StateIndex.STATE_SE2])

                    centroid = self._observation[current_time_idx][token].centroid
                    track_heading = self._observation.unique_objects[token].box.center.heading
                    track_state = StateSE2(centroid.x, centroid.y, track_heading)
                    # TODO: fix ego_area for intersection
                    if is_agent_ahead(ego_rear_axle, track_state) or (
                        (
                            ego_in_multiple_lanes_or_nondrivable_area
                            or self._drivable_area_map.is_in_layer(
                                ego_rear_axle.point, layer=SemanticMapLayer.INTERSECTION
                            )
                        )
                        and not is_agent_behind(ego_rear_axle, track_state)
                    ):
                        ttc_scores[proposal_idx] = np.minimum(ttc_scores[proposal_idx], 0.0)
                        self._ttc_time_idcs[proposal_idx] = min(time_idx, self._ttc_time_idcs[proposal_idx])
                    else:
                        temp_collided_track_ids[proposal_idx].append(token)

        self._weighted_metrics[WeightedMetricIndex.TTC] = ttc_scores

    def _calculate_traffic_light_compliance(self) -> None:
        """
        Re-implementation of hydraMDP++'s traffic light compliance metric.
        """
        # Initialize scores for all proposals to 1 (compliant by default)
        traffic_light_compliance_scores = np.ones(self._num_proposals, dtype=np.float64)

        # Iterate over each time step within the horizon
        for time_idx in range(self.proposal_sampling.num_poses + 1):
            # Get ego polygons (vehicle shapes) at the current time step
            ego_polygons = self._ego_polygons[:, time_idx]
            # Query objects intersecting with the ego polygons
            intersecting = self._observation[time_idx].query(ego_polygons, predicate="intersects")

            # If no intersections, skip this time step
            if len(intersecting) == 0:
                continue

            # Iterate over each intersecting object
            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                # Skip if the score is already 0
                if traffic_light_compliance_scores[proposal_idx] == 0.0:
                    continue

                token = self._observation[time_idx].tokens[geometry_idx]

                # Check if the intersecting object is a red light
                if token.startswith(self._observation.red_light_token):
                    traffic_light_compliance_scores[proposal_idx] = 0.0

        # Store the scores in the multi-metrics system for later evaluation
        self._multi_metrics[MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE] = traffic_light_compliance_scores

    def _calculate_lane_keeping(self) -> None:
        """
        Revised implementation of hydraMDP++'s lane keeping metric.
        The trajectory is considered failing lane-keeping only if it deviates beyond
        the lateral threshold continuously for at least certain seconds.

        """
        # Initialize lane-keeping scores to 1.0
        lane_keeping_scores = np.ones(self._num_proposals, dtype=np.float64)
        lateral_deviation_limit = self._config.lane_keeping_deviation_limit

        interval_length = self.proposal_sampling.interval_length
        continuous_steps_required = int(np.ceil(self._config.lane_keeping_horizon_window / interval_length))

        centerline = self._centerline.linestring

        for proposal_idx in range(self._num_proposals):
            consecutive_exceeds = 0
            for time_idx in range(self.proposal_sampling.num_poses + 1):
                ego_position = Point(*self._ego_coords[proposal_idx, time_idx, BBCoordsIndex.CENTER])

                is_in_intersection = self._drivable_area_map.is_in_layer(
                    ego_position, layer=SemanticMapLayer.INTERSECTION
                )

                if is_in_intersection:
                    continue

                lateral_deviation = ego_position.distance(centerline)

                if lateral_deviation > lateral_deviation_limit:
                    consecutive_exceeds += 1
                else:
                    consecutive_exceeds = 0

                if consecutive_exceeds >= continuous_steps_required:
                    lane_keeping_scores[proposal_idx] = 0.0
                    break

        self._weighted_metrics[WeightedMetricIndex.LANE_KEEPING] = lane_keeping_scores

    def _calculate_history_comfort(self) -> None:
        """
        Implementation of comfort metric, padded with past history states.
        """

        is_history_comfortable = np.ones(self._num_proposals, dtype=np.float64)

        if self._human_past_trajectory is not None:

            # interpolate human past trajectory
            history_start_time_us = self._human_past_trajectory.start_time.time_us
            history_end_time_us = self._human_past_trajectory.end_time.time_us
            # Align history sampling with proposal sampling so states/time axis use the same dt.
            time_interval_us = int(self.proposal_sampling.interval_length * 1e6)

            history_time_us = np.arange(
                history_start_time_us,
                history_end_time_us,
                time_interval_us,
                dtype=np.int64,
            )
            history_time_us = np.clip(history_time_us, history_start_time_us, history_end_time_us)

            history_timepoints = [TimePoint(time_us) for time_us in history_time_us[:-1]]

            if len(history_timepoints) > 0:
                history_state_array = ego_states_to_state_array(
                    self._human_past_trajectory.get_state_at_times(history_timepoints)
                )
            else:
                history_state_array = np.zeros((0, StateIndex.size()), dtype=np.float64)

            # Create state array padded with past human states
            num_padded_poses = len(history_state_array) + self._states.shape[1]
            padded_states_array = np.zeros(
                (self._num_proposals, num_padded_poses, StateIndex.size()),
                dtype=np.float64,
            )

            padded_states_array[:, : len(history_state_array)] = history_state_array
            padded_states_array[:, len(history_state_array) :] = self._states

            # create new timepoints with padding and compute comfort scores
            time_point_s: npt.NDArray[np.float64] = (
                np.arange(0, num_padded_poses).astype(np.float64) * self.proposal_sampling.interval_length
            )
            is_history_comfortable = ego_is_comfortable(padded_states_array, time_point_s).all(axis=-1)

        self._weighted_metrics[WeightedMetricIndex.HISTORY_COMFORT] = is_history_comfortable

    def _calculate_comfort(self) -> None:
        """
        PDMS v1 Comfort metric - simplified version without history padding.
        Checks if trajectory is comfortable based on acceleration/jerk thresholds.
        """
        is_comfortable = np.ones(self._num_proposals, dtype=np.float64)

        # Compute comfort scores for current trajectory
        time_point_s: npt.NDArray[np.float64] = (
            np.arange(0, self._states.shape[1]).astype(np.float64) * self.proposal_sampling.interval_length
        )
        is_comfortable_per_metric = ego_is_comfortable(self._states, time_point_s)
        is_comfortable = is_comfortable_per_metric.all(axis=-1).astype(np.float64)

        self._weighted_metrics[WeightedMetricIndex.LANE_KEEPING] = is_comfortable
