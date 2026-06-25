from abc import ABC
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from shapely.geometry import Point

from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import Dijkstra
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from navsim.planning.simulation.planner.pdm_planner.utils.route_utils import interpolate_lane_path, route_roadblock_correction
from navsim.planning.simulation.planner.pdm_planner.vis_utils import debug_plot_lane_boundaries_and_baseline, plot_centerline_discrete_path, plot_route_plan_with_polygons

import os

class AbstractPDMPlanner(AbstractPlanner, ABC):
    """
    Interface for planners incorporating PDM-* variants.
    """

    def __init__(
        self,
        map_radius: float,
    ):
        """
        Constructor of AbstractPDMPlanner.
        :param map_radius: radius around ego to consider
        """

        self._map_radius: int = map_radius  # [m]
        self._iteration: int = 0

        # lazy loaded
        self._map_api: Optional[AbstractMap] = None
        self._route_roadblock_dict: Optional[Dict[str, RoadBlockGraphEdgeMapObject]] = None
        self._route_lane_dict: Optional[Dict[str, LaneGraphEdgeMapObject]] = None

        self._centerline: Optional[PDMPath] = None
        self._drivable_area_map: Optional[PDMDrivableMap] = None

    def _load_route_dicts(self, route_roadblock_ids: List[str]) -> None:
        """
        Loads roadblock and lane dictionaries of the target route from the map-api.
        :param route_roadblock_ids: ID's of on-route roadblocks
        """
        # remove repeated ids while remaining order in list
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

        self._route_roadblock_dict = {}
        self._route_lane_dict = {}
        # If no roadblocks provided (e.g., V2X-Real), use all available lanes as route
        if not route_roadblock_ids:
            # Get all lanes from map within a large radius

            position = Point(0, 0)  # Will be updated on first iteration
            all_lanes = self._map_api.get_proximal_map_objects(
                position, 1000.0, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
            )
            for layer in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
                for lane in all_lanes.get(layer, []):
                    if hasattr(lane, 'id'):
                        if lane.id in self._map_api.get_route_lane_ids():  # filter lane based on route_lane_ids()
                            self._route_lane_dict[lane.id] = lane
                        # else:
                        #     self._route_lane_dict[lane.id] = lane

            return

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)

            self._route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                self._route_lane_dict[lane.id] = lane

    def _route_roadblock_correction(self, ego_state: EgoState) -> None:
        """
        Corrects the roadblock route and reloads lane-graph dictionaries.
        :param ego_state: state of the ego vehicle.
        """
        # For V2X-Real dataset: if no roadblocks exist, find lanes near ego position
        if not self._route_roadblock_dict:
            # Get lanes from map near ego position
            if self._route_lane_dict is not None:
                return
            ego_position = Point(ego_state.rear_axle.x, ego_state.rear_axle.y)
            all_lanes = self._map_api.get_proximal_map_objects(
                ego_position, self._map_radius, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
            )
            self._route_lane_dict = {}
            for layer in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
                for lane in all_lanes.get(layer, []):
                    if hasattr(lane, 'id') and lane.id:
                        self._route_lane_dict[lane.id] = lane
            return
        route_roadblock_ids = route_roadblock_correction(ego_state.rear_axle, self._map_api, self._route_roadblock_dict)
        
        self._load_route_dicts(route_roadblock_ids)

    def _get_discrete_centerline(self, current_lane: LaneGraphEdgeMapObject, search_depth: int = 30) -> List[StateSE2]:
        """
        Applies a Dijkstra search on the lane-graph to retrieve discrete centerline.
        :param current_lane: lane object of starting lane.
        :param search_depth: depth of search (for runtime), defaults to 30
        :return: list of discrete states on centerline (x,y,θ)
        """

        roadblocks = list(self._route_roadblock_dict.values())
        roadblock_ids = list(self._route_roadblock_dict.keys())
        # If no roadblocks (e.g., V2X-Real), return current lane baseline
        # debug_plot_lane_boundaries_and_baseline(
        #     current_lane,
        #     zoom_mode="bounds",
        #     pad_m=2.0,          # smaller padding zooms in further
        #     arrow_len=2.0,
        #     heading_every=1,
        #     annotate_step=1,
        #     out_path="exp2/debug_lane_baseline/lane_1-18_zoom.png",
        # )

        if not roadblock_ids:
            centerline_discrete_path: List[StateSE2] = []
            if current_lane and current_lane.baseline_path:
                # original code
                if len(current_lane.incoming_edges) > 0:
                    centerline_discrete_path.extend(current_lane.incoming_edges[0].baseline_path.discrete_path)
                centerline_discrete_path.extend(current_lane.baseline_path.discrete_path)
                # plot_centerline_discrete_path(centerline_discrete_path, out_path="exp2/v2xreal_curlane_centerline.png")
                cur_l = current_lane
                while len(cur_l.outgoing_edges) > 0:
                    cur_l = cur_l.outgoing_edges[0]
                    centerline_discrete_path.extend(cur_l.baseline_path.discrete_path)
                
                # interpolation code added
                # --- change: interpolate each lane by type to generate dense points ---
                # if len(current_lane.incoming_edges) > 0:
                #     centerline_discrete_path.extend(
                #         interpolate_lane_path(current_lane.incoming_edges[0])
                #     )
                # centerline_discrete_path.extend(
                #     interpolate_lane_path(current_lane)
                # )
                # cur_l = current_lane
                # while len(cur_l.outgoing_edges) > 0:
                #     cur_l = cur_l.outgoing_edges[0]
                #     centerline_discrete_path.extend(
                #         interpolate_lane_path(cur_l)
                #     )
                # --- end of change ---
    
                # plot_centerline_discrete_path(centerline_discrete_path, out_path="exp_debug/v2xreal_centerline.png")
                return centerline_discrete_path
            else:
                # Return empty path if current_lane is None or has no baseline_path
                return []

        # find current roadblock index
        start_idx = np.argmax(np.array(roadblock_ids) == current_lane.get_roadblock_id())
        roadblock_window = roadblocks[start_idx : start_idx + search_depth]

        graph_search = Dijkstra(current_lane, list(self._route_lane_dict.keys()))
        route_plan, path_found = graph_search.search(roadblock_window[-1])
        plot_route_plan_with_polygons(
            route_plan=route_plan,
            out_path="exp2/route_plan_poly.png",
            annotate=True,
            draw_polygons=True,
        )
        centerline_discrete_path: List[StateSE2] = []
        for lane in route_plan:
            centerline_discrete_path.extend(lane.baseline_path.discrete_path)
        
        # plot_centerline_discrete_path(centerline_discrete_path, out_path="exp2/centerline.png")

        return centerline_discrete_path

    def _get_starting_lane(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        """
        Returns the most suitable starting lane, in ego's vicinity.
        :param ego_state: state of ego-vehicle
        :return: lane object (on-route)
        """
        starting_lane: LaneGraphEdgeMapObject = None
        on_route_lanes, heading_error = self._get_intersecting_lanes(ego_state)

        if len(on_route_lanes) > 0:
            # 1. Option: find lanes from lane occupancy-map
            # select lane with lowest heading error
            starting_lane = on_route_lanes[np.argmin(np.abs(heading_error))]
            return starting_lane

        else:
            # 2. Option: find any intersecting or close lane on-route
            closest_distance = np.inf
            for edge in self._route_lane_dict.values():
                if edge.contains_point(ego_state.center):
                    starting_lane = edge
                    break

                distance = edge.polygon.distance(ego_state.car_footprint.geometry)
                if distance < closest_distance:
                    starting_lane = edge
                    closest_distance = distance

        return starting_lane

    def _get_intersecting_lanes(self, ego_state: EgoState) -> Tuple[List[LaneGraphEdgeMapObject], List[float]]:
        """
        Returns on-route lanes and heading errors where ego-vehicle intersects.
        :param ego_state: state of ego-vehicle
        :return: tuple of lists with lane objects and heading errors [rad].
        """
        # For datasets like V2X-Real without drivable_area_map, return empty list
        if not self._drivable_area_map:
            return [], []

        ego_position_array: npt.NDArray[np.float64] = ego_state.rear_axle.array
        ego_rear_axle_point: Point = Point(*ego_position_array)
        ego_heading: float = ego_state.rear_axle.heading

        intersecting_lanes = self._drivable_area_map.intersects(ego_rear_axle_point)

        on_route_lanes, on_route_heading_errors = [], []
        for lane_id in intersecting_lanes:
            if lane_id in self._route_lane_dict.keys():
                # collect baseline path as array
                lane_object = self._route_lane_dict[lane_id]
                lane_discrete_path: List[StateSE2] = lane_object.baseline_path.discrete_path
                lane_state_se2_array = np.array([state.array for state in lane_discrete_path], dtype=np.float64)
                # calculate nearest state on baseline
                lane_distances = (ego_position_array[None, ...] - lane_state_se2_array) ** 2
                lane_distances = lane_distances.sum(axis=-1) ** 0.5

                # calculate heading error
                heading_error = lane_discrete_path[np.argmin(lane_distances)].heading - ego_heading
                heading_error = np.abs(normalize_angle(heading_error))

                # add lane to candidates
                on_route_lanes.append(lane_object)
                on_route_heading_errors.append(heading_error)

        return on_route_lanes, on_route_heading_errors
