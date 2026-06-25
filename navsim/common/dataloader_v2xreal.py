from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
from collections import defaultdict
import os
from tqdm import tqdm
from shapely.geometry import Polygon, LineString, Point
from matplotlib.lines import Line2D

from navsim.common.dataclasses import AgentInput, Scene, SceneFilter, SceneMetadata, SensorConfig, Frame, V2XLaneConnectorObject
from navsim.common.loader_vis import visualize_lane_boundaries_with_arrows_and_legend
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.metric_caching.metric_cache import MetricCache
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from pyquaternion import Quaternion

import numpy as np

FrameList = List[Dict[str, Any]]

# Filter duplicated ego-like boxes in V2X-Real annotations.
# Radius is measured in ego frame (meters). Set <= 0 to disable.
V2X_EGO_OVERLAP_FILTER_RADIUS_M = float(os.getenv("NAVSIM_V2X_EGO_OVERLAP_FILTER_RADIUS_M", "1.0"))
V2X_EGO_OVERLAP_FILTER_TYPES = {"car", "truck", "bus", "motorcycle", "vehicle", "ego"}

# driving command
CMD_STRAIGHT = 0
CMD_LEFT = 1
CMD_RIGHT = 2
CMD_UNKNOWN = 3


class V2XMapObject:
    """Wrapper for V2X map objects to provide .polygon and .id attributes."""
    def __init__(self, lane_dict: Dict[str, Any], map_data: Dict[str, Any], lane_id: str = ''):
        self._lane_dict = lane_dict
        self._map_data = map_data
        self._lane_id = lane_id  # Store the lane ID from map dict key
        self._polygon_cache = None
        self._incoming_edges_cache = None
        self._outgoing_edges_cache = None
    
    @property
    def id(self) -> str:
        """Get object ID."""
        # Prefer explicit lane_id, fallback to token in dict
        return self._lane_id or self._lane_dict.get('token', '')
    
    @property
    def polygon(self) -> Polygon:
        """Get polygon geometry."""
        if self._polygon_cache is not None:
            return self._polygon_cache
        
        # V2X-Real format: left_boundary and right_boundary as string coordinates
        left_boundary = self._lane_dict.get('left_boundary', [])
        right_boundary = self._lane_dict.get('right_boundary', [])
        
        if not left_boundary or not right_boundary:
            self._polygon_cache = Polygon()
            return self._polygon_cache
        
        # Parse boundary strings like "(x, y)" to float tuples
        def parse_coord(coord_str):
            try:
                # Remove parentheses and split by comma
                coord_str = coord_str.strip('()')
                x, y = map(float, coord_str.split(','))
                return (x, y)
            except:
                return None
        
        left_coords = [parse_coord(c) for c in left_boundary if parse_coord(c) is not None]
        right_coords = [parse_coord(c) for c in right_boundary if parse_coord(c) is not None]
        
        if len(left_coords) < 2 or len(right_coords) < 2:
            self._polygon_cache = Polygon()
            return self._polygon_cache
        
        # Create polygon: left boundary forward + right boundary reversed
        polygon_coords = left_coords + list(reversed(right_coords))
        
        try:
            self._polygon_cache = Polygon(polygon_coords)
        except:
            self._polygon_cache = Polygon()
        
        return self._polygon_cache
    
    @staticmethod
    def _parse_boundary(coord_str_list: List[str]) -> np.ndarray:
        """Parse boundary coordinate strings like '(x, y)' into Nx2 array."""
        coords = []
        for s in coord_str_list:
            s = s.strip('() ')
            parts = s.split(',')
            coords.append((float(parts[0]), float(parts[1])))
        return np.array(coords, dtype=np.float64) if coords else np.empty((0, 2))

    @staticmethod
    def _arc_length_param(arr: np.ndarray) -> np.ndarray:
        """Compute normalized arc-length parameterization for Nx2 points."""
        d = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        total = s[-1]
        return s / total if total > 0 else np.linspace(0, 1, len(arr))

    @property
    def baseline_path(self) -> PDMPath:
        """Compute centerline as midline between left and right boundaries.
        
        Uses arc-length parameterization to resample the shorter boundary
        to match the longer one, then averages for the centerline.
        """
        left_arr = self._parse_boundary(self._lane_dict.get('left_boundary', []))
        right_arr = self._parse_boundary(self._lane_dict.get('right_boundary', []))

        if len(left_arr) < 2 or len(right_arr) < 2:
            return PDMPath([StateSE2(0.0, 0.0, 0.0), StateSE2(1.0, 0.0, 0.0)])

        # Resample both boundaries to the same number of points via arc-length
        n_pts = max(len(left_arr), len(right_arr))
        t_uniform = np.linspace(0, 1, n_pts)

        t_left = self._arc_length_param(left_arr)
        left_resampled = np.column_stack([
            np.interp(t_uniform, t_left, left_arr[:, 0]),
            np.interp(t_uniform, t_left, left_arr[:, 1]),
        ])

        t_right = self._arc_length_param(right_arr)
        right_resampled = np.column_stack([
            np.interp(t_uniform, t_right, right_arr[:, 0]),
            np.interp(t_uniform, t_right, right_arr[:, 1]),
        ])

        centerline = (left_resampled + right_resampled) / 2.0

        # Remove duplicate consecutive points
        dists = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
        keep = np.concatenate([[True], dists > 1e-6])
        centerline = centerline[keep]

        if len(centerline) < 2:
            centerline = np.array([[0.0, 0.0], [1.0, 0.0]])

        # Build StateSE2 list with proper headings
        states: List[StateSE2] = []
        for i in range(len(centerline)):
            x, y = float(centerline[i, 0]), float(centerline[i, 1])
            if i < len(centerline) - 1:
                nx, ny = float(centerline[i + 1, 0]), float(centerline[i + 1, 1])
                heading = math.atan2(ny - y, nx - x)
            else:
                px, py = float(centerline[i - 1, 0]), float(centerline[i - 1, 1])
                heading = math.atan2(y - py, x - px)
            states.append(StateSE2(x, y, heading))

        return PDMPath(states)

    @property
    def speed_limit_mps(self) -> float:
        """Default speed limit for V2X-Real lanes (m/s)."""
        return 13.89  # 50 km/h

    @property
    def incoming_edges(self) -> List['V2XMapObject']:
        """If set, return stored edges (lane-connectors). Otherwise return []."""
        if self._incoming_edges_cache is not None:
            return self._incoming_edges_cache
        return []

    def set_incoming_edges(self, edges: Union[Any, Sequence[Any]]) -> None:
        if edges is None:
            self._incoming_edges_cache = []
        elif isinstance(edges, (list, tuple)):
            self._incoming_edges_cache = list(edges)
        else:
            self._incoming_edges_cache = [edges]

    @property
    def outgoing_edges(self) -> List['V2XMapObject']:
        """If set, return stored edges (lane-connectors). Otherwise return []."""
        if self._outgoing_edges_cache is not None:
            return self._outgoing_edges_cache
        return []

    def set_outgoing_edges(self, edges: Union[Any, Sequence[Any]]) -> None:
        if edges is None:
            self._outgoing_edges_cache = []
        elif isinstance(edges, (list, tuple)):
            self._outgoing_edges_cache = list(edges)
        else:
            self._outgoing_edges_cache = [edges]

    def contains_point(self, point) -> bool:
        """Check if point is contained in lane polygon."""
        from shapely.geometry import Point as ShapelyPoint
        if hasattr(point, 'x') and hasattr(point, 'y'):
            shapely_point = ShapelyPoint(point.x, point.y)
        else:
            shapely_point = ShapelyPoint(point[0], point[1])
        return self.polygon.contains(shapely_point)
    
    def get_roadblock_id(self) -> Optional[str]:
        """Get roadblock ID. V2X-Real has no roadblocks, return None."""
        return None
    
    @property
    def interior_edges(self) -> List['V2XMapObject']:
        """Get interior edges (lanes). For V2X-Real, return self as a single-item list."""
        return [self]


def load_test_tokens_from_pkl(pkl_path: Path) -> List[str]:
    """
    Load test sample tokens from V2X pkl file.
    :param pkl_path: path to spd_infos_temporal_test.pkl
    :return: list of token strings
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    tokens = [info['token'] for info in data['infos']]
    print(f"Loaded {len(tokens)} test tokens from {pkl_path}")
    return tokens


class V2XRealMapWrapper(AbstractMap):
    """
    Simple wrapper for V2X-Real map JSON data.
    Provides enough functionality for metric caching without full map implementation.
    """
    
    def __init__(self, map_root: Path):
        """
        Initialize with V2X-Real map.
        :param map_root: path to V2X-Real dataset root containing maps/expansion/v2x_real_map.json or v2x_real_map.json
        """
        self._map_name = "v2x_real"
        map_root = Path(map_root)
        
        # Try multiple possible locations for the map file
        possible_paths = [
            map_root / 'maps/expansion/v2x_real_map.json',
            map_root / 'v2x_real_map.json',
            map_root / 'map/cache/v2x_real_map.json',
        ]
        
        map_file = None
        for path in possible_paths:
            if path.exists():
                map_file = path
                break
        
        if map_file is None:
            raise FileNotFoundError(f"Map file not found in any of: {possible_paths}")
        
        # Load map data
        with open(map_file) as f:
            self._map_data = json.load(f)
        
        # Cache processed objects
        self._roadblocks_cache = None
        self._lanes_cache = None
        self._crosswalks_cache = None
        self._junctions_cache = None
        self._lane_connectors_cache: List[V2XLaneConnectorObject] = []
        self.road_block_ids_dict = {
            "rb0_6": ["1-15", "1-14", "1-36"],
            "rb6_0": ["1-35", "1-16", "1-17"],
            "rb1_0": ["1-5", "1-6"],
            "rb0_1": ["1-7", "1-8", "1-9"],
            "rb0_3": ["1-19", "1-18"],
            "rb3_0": ["1-26", "1-27", "1-28", "1-29"],
            "rb2_0": ["1-1", "1-2"],
            "rb0_2": ["1-3", "1-4"],
            "rb4_3": ["1-21", "1-20", "1-22"],
            "rb3_4": ["1-23", "1-24", "1-25"],
            "rb6_5": ["1-12", "1-13"],
            "rb5_6": ["1-11", "1-10"],
        }

        self.route_lane_ids = None
    
   
    def road_block_ids_dict(self) -> Dict:
        return self.road_block_ids_dict
    
    def set_route_lane_ids(self, route_lane_ids):
        self.route_lane_ids = route_lane_ids
    
    def get_route_lane_ids(self) -> List:
        return self.route_lane_ids

    @property
    def map_name(self) -> str:
        """Get map name."""
        return self._map_name
    
    def _get_roadblocks(self) -> List[Dict[str, Any]]:
        """Get cached roadblocks from map data."""
        if self._roadblocks_cache is None:
            # Try both lowercase and uppercase keys
            self._roadblocks_cache = self._map_data.get('road_block', self._map_data.get('ROAD_BLOCK', []))
        return self._roadblocks_cache
    
    def _get_lanes(self) -> List[V2XMapObject]:
        """Get cached lanes as wrapped objects with polygon property."""
        if self._lanes_cache is None:
            # Try both lowercase and uppercase keys
            lane_data = self._map_data.get('lane', self._map_data.get('LANE', {}))
            # Handle both list and dict formats
            if isinstance(lane_data, dict):
                # Pass lane ID (dict key) along with lane data
                self._lanes_cache = [V2XMapObject(lane_dict, self._map_data, lane_id=lane_id) 
                                     for lane_id, lane_dict in lane_data.items()]
            else:
                # List format - no explicit IDs
                self._lanes_cache = [V2XMapObject(lane, self._map_data) for lane in lane_data]
        return self._lanes_cache
    
    def _get_crosswalks(self) -> List[Polygon]:
        """Get crosswalk polygons for drivable area."""
        if self._crosswalks_cache is None:
            crosswalk_data = self._map_data.get('crosswalk', self._map_data.get('CROSSWALK', {}))
            if isinstance(crosswalk_data, dict):
                crosswalk_list = list(crosswalk_data.values())
            else:
                crosswalk_list = crosswalk_data
            
            self._crosswalks_cache = []
            for cw_dict in crosswalk_list:
                polygon_coords_str = cw_dict.get('polygon', [])
                coords = self._parse_polygon_coords(polygon_coords_str)
                if len(coords) >= 3:
                    try:
                        self._crosswalks_cache.append(Polygon(coords))
                    except:
                        pass
        return self._crosswalks_cache

    def _get_lane_connectors(self) -> List[V2XLaneConnectorObject]:
        return self._lane_connectors_cache

    def _get_junctions(self) -> List[Polygon]:
        """Get junction/intersection polygons for drivable area."""
        if self._junctions_cache is None:
            junction_data = self._map_data.get('junction', self._map_data.get('JUNCTION', {}))
            if isinstance(junction_data, dict):
                junction_list = list(junction_data.values())
            else:
                junction_list = junction_data
            
            self._junctions_cache = []
            for junc_dict in junction_list:
                polygon_coords_str = junc_dict.get('polygon', [])
                coords = self._parse_polygon_coords(polygon_coords_str)
                if len(coords) >= 3:
                    try:
                        self._junctions_cache.append(Polygon(coords))
                    except:
                        pass
        return self._junctions_cache
    
    @staticmethod
    def _parse_polygon_coords(coord_str_list: List[str]) -> List[tuple]:
        """Parse polygon coordinates from string list like ['(x, y)', ...]"""
        coords = []
        for coord_str in coord_str_list:
            try:
                coord_str = coord_str.strip('()')
                x, y = map(float, coord_str.split(','))
                coords.append((x, y))
            except:
                pass
        return coords
    
    def get_map_object(self, object_id: str, layer: SemanticMapLayer) -> Any:
        """Get map object by id and layer."""
        if layer == SemanticMapLayer.ROADBLOCK:
            for rb in self._get_roadblocks():
                if rb.get('token') == object_id:
                    return rb
        elif layer == SemanticMapLayer.LANE:
            for lane in self._get_lanes():
                if lane.id == object_id:  # Use .id property from V2XMapObject
                    return lane
        elif layer == SemanticMapLayer.LANE_CONNECTOR:
            for lc in self._get_lane_connectors():
                if lc.id == object_id:
                    return lc
        return None
    
    def get_map_objects(self, layer: SemanticMapLayer) -> List[Any]:
        """Get all map objects in a layer."""
        if layer == SemanticMapLayer.ROADBLOCK:
            return self._get_roadblocks()
        elif layer == SemanticMapLayer.LANE:
            return self._get_lanes()
        elif layer == SemanticMapLayer.LANE_CONNECTOR:
            return self._get_lane_connectors()
        return []
    
    def is_in_layer(self, x: float, y: float, layer: SemanticMapLayer) -> bool:
        """Check if point is in layer."""
        from shapely.geometry import Point as ShapelyPoint
        
        if layer == SemanticMapLayer.ROADBLOCK:
            # For V2X-Real, drivable area = lanes + crosswalks + junctions (intersections)
            point = ShapelyPoint(x, y)
            
            # Check lanes
            lanes = self._get_lanes()
            for lane in lanes:
                if lane.polygon.contains(point):
                    return True
            
            # Check crosswalks (pedestrian crossings are also drivable)
            crosswalks = self._get_crosswalks()
            for crosswalk in crosswalks:
                if crosswalk.contains(point):
                    return True
            
            # Check junctions (intersections)
            junctions = self._get_junctions()
            for junction in junctions:
                if junction.contains(point):
                    return True
            
            return False
        
        return False
    
    def get_nearest_lane_id(self, x: float, y: float) -> Optional[str]:
        """Get nearest lane id."""
        lanes = self._get_lanes()
        if lanes:
            return lanes[0].get('token')
        return None
    
    def get_layer_polygon_list(self, layer: SemanticMapLayer) -> List[Any]:
        """Get layer polygon list."""
        if layer == SemanticMapLayer.ROADBLOCK:
            return self._get_roadblocks()
        return []
    
    def get_discrete_driving_direction(self, x: float, y: float) -> Optional[float]:
        """Get discrete driving direction."""
        return None
    
    def get_all_map_objects(self, *args, **kwargs) -> List[Any]:
        """Get all map objects."""
        result = []
        result.extend(self._get_roadblocks())
        result.extend(self._get_lanes())
        return result
    
    def get_available_map_objects(self, *args, **kwargs) -> List[Any]:
        """Get available map objects."""
        return self.get_all_map_objects(*args, **kwargs)
    
    def get_available_raster_layers(self, *args, **kwargs) -> List[Any]:
        """Get available raster layers."""
        return []
    
    def get_distance_to_nearest_map_object(self, point, layer) -> Optional[tuple]:
        """Get distance to nearest map object. V2X-Real has no roadblocks, return None tuple."""
        # V2X-Real doesn't have roadblocks, return None tuple to skip route correction
        return (None, None)
    
    def get_distance_to_nearest_raster_layer(self, *args, **kwargs) -> Optional[float]:
        """Get distance to nearest raster layer."""
        return None
    
    def get_distances_matrix_to_nearest_map_object(self, *args, **kwargs) -> Any:
        """Get distances matrix to nearest map object."""
        return None
    
    def get_one_map_object(self, *args, **kwargs) -> Any:
        """Get one map object."""
        return None
    
    def get_proximal_map_objects(self, point, radius, layers) -> Dict[SemanticMapLayer, List[Any]]:
        """Get proximal map objects.
        
        For V2X-Real, ROADBLOCK queries return lanes + crosswalks + junctions (drivable areas).
        """
        result = {}
        for layer in layers:
            if layer == SemanticMapLayer.ROADBLOCK:
                # V2X-Real: Return lanes as roadblocks
                # PDMDrivableMap needs these as MapObject-like items with .polygon and .id
                result[layer] = self._get_lanes()
            elif layer == SemanticMapLayer.ROADBLOCK_CONNECTOR:
                result[layer] = []  # V2X-Real may not have connectors
            elif layer == SemanticMapLayer.LANE:
                result[layer] = self._get_lanes()
            elif layer == SemanticMapLayer.INTERSECTION:
                # Return junctions as intersection polygons
                # Wrap in V2XMapObject-like structure
                junctions = self._get_junctions()
                junction_objects = []
                for idx, poly in enumerate(junctions):
                    # Create a simple object with polygon and id attributes
                    class JunctionMapObject:
                        def __init__(self, polygon, obj_id):
                            self.polygon = polygon
                            self.id = f'junction_{obj_id}'
                    junction_objects.append(JunctionMapObject(poly, idx))
                result[layer] = junction_objects
            elif layer == SemanticMapLayer.CROSSWALK:
                # Return crosswalks
                crosswalks = self._get_crosswalks()
                crosswalk_objects = []
                for idx, poly in enumerate(crosswalks):
                    class CrosswalkMapObject:
                        def __init__(self, polygon, obj_id):
                            self.polygon = polygon
                            self.id = f'crosswalk_{obj_id}'
                    crosswalk_objects.append(CrosswalkMapObject(poly, idx))
                result[layer] = crosswalk_objects
            elif layer == SemanticMapLayer.CARPARK_AREA:
                result[layer] = []  # V2X-Real has no carpark areas
            elif layer == SemanticMapLayer.LANE_CONNECTOR:
                result[layer] = self._get_lane_connectors()
            else:
                result[layer] = []
        return result
    
    def get_raster_map(self, *args, **kwargs) -> Any:
        """Get raster map."""
        return None
    
    def get_raster_map_layer(self, *args, **kwargs) -> Any:
        """Get raster map layer."""
        return None
    
    def initialize_all_layers(self, *args, **kwargs) -> None:
        """Initialize all layers."""
        pass

    def _lane_id_to_obj(self) -> Dict[str, V2XMapObject]:
        lanes = self._get_lanes()
        return {l.id: l for l in lanes if getattr(l, "id", None)}

    def _get_connector_lane_ids(self, c: V2XLaneConnectorObject) -> tuple:
        in_ids = list(getattr(c, "_incoming_lane_ids", []) or [])
        out_ids = list(getattr(c, "_outgoing_lane_ids", []) or [])
        return in_ids, out_ids

    def set_lane_connectors(self, connectors: List[V2XLaneConnectorObject]) -> None:
        self._lane_connectors_cache = connectors or []
        self.update_lane_edges_from_connectors()

    def update_lane_edges_from_connectors(self) -> None:
        """
        For each lane mentioned by any connector's in/out, attach that connector
        into lane.incoming_edges / lane.outgoing_edges.
        """
        lanes_by_id = self._lane_id_to_obj()

        for c in self._lane_connectors_cache:
            in_ids, out_ids = self._get_connector_lane_ids(c)

            if len(in_ids) != 0:
                in_lane = lanes_by_id.get(in_ids[0])
                if in_lane is not None:
                    in_lane.set_outgoing_edges(c)

            if len(out_ids) != 0:
                out_lane = lanes_by_id.get(out_ids[0])
                if out_lane is not None:
                    out_lane.set_incoming_edges(c)

            c.resolve_edges(lanes_by_id)


def load_v2xreal_pkl(pkl_path: Path) -> List[Dict[str, Any]]:
    """
    Load V2X-Real pkl file which contains a dict with 'infos' key.
    :param pkl_path: path to V2X-Real pkl file
    :return: list of frame dictionaries
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and "infos" in data:
        return data["infos"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected pkl format: {type(data)}")


def convert_v2xreal_to_navsim_format(frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert V2X-Real frame format to NavSim-compatible format.
    :param frame: V2X-Real frame dictionary
    :return: NavSim-compatible frame dictionary
    """
    # V2X-Real is already similar to NuScenes, but we need to ensure all required fields exist
    navsim_frame = {
        "token": frame["token"],
        "log_name": frame.get("scene_token", "unknown"),  # Use scene_token as log_name
        "scene_name": frame.get("scene_token", "unknown"),
        "scene_token": frame.get("scene_token", "unknown"),  # Keep scene_token for Scene.from_scene_dict_list
        "map_location": "v2x_real",  # V2X-Real map name
        "timestamp": frame["timestamp"],
        "frame_idx": frame["frame_idx"],
        "ego2global_translation": frame["ego2global_translation"],
        "ego2global_rotation": frame["ego2global_rotation"],
        "lidar_path": frame.get("lidar_path"),
        "cams": frame.get("cams", {}),
    }
    
    # Handle ego status - V2X uses can_bus instead of ego_status
    # V2X-Real has no can_bus data (all zeros), so we'll calculate velocity from position changes later in Scene class
    # [vx_body, vy_body, ax, ay, yaw_rate]
    navsim_frame["ego_dynamic_state"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Handle driving command - default to 0 if not present
    navsim_frame["driving_command"] = frame.get("command", 0)
    
    # Handle roadblock_ids - may not exist in V2X
    navsim_frame["roadblock_ids"] = frame.get("roadblock_ids", [])
    
    # Handle traffic lights - may not exist in V2X
    navsim_frame["traffic_light_status"] = frame.get("traffic_light_status", [])
    
    # Handle annotations
    gt_boxes = frame.get("gt_boxes", [])
    gt_names = frame.get("gt_names", [])
    gt_velocity = frame.get("gt_velocity", [])
    gt_ins_tokens = frame.get("gt_ins_tokens", [])
    
    # V2X gt_boxes are 7-dim [x, y, z, w, l, h, heading] in EGO frame (lidar coordinate)
    # NavSim expects 9-dim [x, y, z, l, w, h, vx, vy, heading] in GLOBAL frame
    # Convert from ego to global
    import numpy as np
    if len(gt_boxes) > 0:
        gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
        gt_names_arr = np.asarray(gt_names, dtype=object)
        gt_ins_tokens_arr = np.asarray(gt_ins_tokens, dtype=object)
        gt_velocity = np.asarray(gt_velocity, dtype=np.float64) if len(gt_velocity) > 0 else np.zeros((len(gt_boxes), 2))

        # Remove ego-overlapping vehicle-like boxes (in ego frame) before conversion.
        if (
            V2X_EGO_OVERLAP_FILTER_RADIUS_M > 0.0
            and len(gt_names_arr) == len(gt_boxes)
            and len(gt_ins_tokens_arr) == len(gt_boxes)
        ):
            dist_sq = gt_boxes[:, 0] ** 2 + gt_boxes[:, 1] ** 2
            is_vehicle_like = np.array(
                [str(name).lower() in V2X_EGO_OVERLAP_FILTER_TYPES for name in gt_names_arr],
                dtype=bool,
            )
            drop_mask = is_vehicle_like & (dist_sq < V2X_EGO_OVERLAP_FILTER_RADIUS_M ** 2)
            if np.any(drop_mask):
                keep_mask = ~drop_mask
                gt_boxes = gt_boxes[keep_mask]
                gt_velocity = gt_velocity[keep_mask]
                gt_names_arr = gt_names_arr[keep_mask]
                gt_ins_tokens_arr = gt_ins_tokens_arr[keep_mask]

        gt_names = gt_names_arr.tolist()
        gt_ins_tokens = gt_ins_tokens_arr.tolist()
        
        # Get ego pose for transformation
        from pyquaternion import Quaternion
        ego_trans = np.array(frame["ego2global_translation"])
        ego_quat = Quaternion(*frame["ego2global_rotation"])
        ego_heading = ego_quat.yaw_pitch_roll[0]
        
        # Rotation matrix for ego to global
        cos_ego = np.cos(ego_heading)
        sin_ego = np.sin(ego_heading)
        
        gt_boxes_9d = np.zeros((len(gt_boxes), 9))
        for i, box in enumerate(gt_boxes):
            # Transform position from ego to global
            x_ego, y_ego = box[0], box[1]
            x_global = ego_trans[0] + x_ego * cos_ego - y_ego * sin_ego
            y_global = ego_trans[1] + x_ego * sin_ego + y_ego * cos_ego
            
            gt_boxes_9d[i, 0] = x_global
            gt_boxes_9d[i, 1] = y_global
            gt_boxes_9d[i, 2] = box[2] + ego_trans[2]  # z
            gt_boxes_9d[i, 3] = box[4]  # length (was at index 4 in V2X)
            gt_boxes_9d[i, 4] = box[3]  # width (was at index 3 in V2X)
            gt_boxes_9d[i, 5] = box[5]  # height
            
            # Transform velocity from ego to global
            vx_ego, vy_ego = gt_velocity[i, 0], gt_velocity[i, 1]
            vx_global = vx_ego * cos_ego - vy_ego * sin_ego
            vy_global = vx_ego * sin_ego + vy_ego * cos_ego
            gt_boxes_9d[i, 6] = vx_global
            gt_boxes_9d[i, 7] = vy_global
            
            # Transform heading from ego to global (standard rotation)
            heading_ego = -box[6] - np.pi/2

            heading_global = heading_ego + ego_heading
            gt_boxes_9d[i, 8] = heading_global
        
        gt_velocity_3d = np.zeros((len(gt_boxes), 3))
        gt_velocity_3d[:, :2] = gt_velocity[:, :2]
    else:
        gt_boxes_9d = np.array([])
        gt_velocity_3d = np.array([])
    
    navsim_frame["anns"] = {
        "gt_boxes": gt_boxes_9d,
        "gt_names": gt_names,
        "gt_velocity_3d": gt_velocity_3d,
        "instance_tokens": gt_ins_tokens,
        "track_tokens": gt_ins_tokens,  # Use same as instance tokens if no separate track tokens
        "is_v2xreal": True,  # Flag to indicate V2X-Real data (boxes already in global frame)
    }
    
    return navsim_frame


# ============== Shared utility functions ==============


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    import math
    return math.atan2(math.sin(angle), math.cos(angle))


def compute_body_frame_velocity(all_frames: List[Dict[str, Any]]) -> None:
    """
    Calculate ego velocity and yaw rate using central difference and convert global → body frame.
    Modifies frames in-place. Only updates frames where velocity is (0, 0).

    Uses central difference where possible (prev & next frames in same scene).
    Falls back to forward/backward difference at scene boundaries.
    Also computes yaw rate (angular velocity) from heading differences.

    nuPlan's EgoState.rear_axle_velocity_2d expects body frame (x=forward, y=left).
    V2X-Real has no CAN bus data, so velocity must be estimated from position changes.

    :param all_frames: list of NavSim-format frame dicts (must have ego2global_translation,
                       ego2global_rotation, ego_dynamic_state, scene_token, frame_idx)
    """
    import numpy as np
    from pyquaternion import Quaternion

    n = len(all_frames)
    for i in range(n):
        ego_dynamic = all_frames[i]["ego_dynamic_state"]
        if ego_dynamic[0] == 0.0 and ego_dynamic[1] == 0.0:  # vx and vy are zero
            same_prev = (i - 1 >= 0 and all_frames[i]["scene_token"] == all_frames[i - 1]["scene_token"])
            same_next = (i + 1 < n and all_frames[i]["scene_token"] == all_frames[i + 1]["scene_token"])

            if same_prev and same_next:
                # Central difference
                prev_pos = all_frames[i - 1]["ego2global_translation"]
                next_pos = all_frames[i + 1]["ego2global_translation"]
                dt = (all_frames[i + 1]["timestamp"] - all_frames[i - 1]["timestamp"]) / 1e6
                prev_heading = Quaternion(*all_frames[i - 1]["ego2global_rotation"]).yaw_pitch_roll[0]
                next_heading = Quaternion(*all_frames[i + 1]["ego2global_rotation"]).yaw_pitch_roll[0]
            elif same_next:
                # Forward difference (first frame of scene)
                prev_pos = all_frames[i]["ego2global_translation"]
                next_pos = all_frames[i + 1]["ego2global_translation"]
                dt = (all_frames[i + 1]["timestamp"] - all_frames[i]["timestamp"]) / 1e6
                prev_heading = Quaternion(*all_frames[i]["ego2global_rotation"]).yaw_pitch_roll[0]
                next_heading = Quaternion(*all_frames[i + 1]["ego2global_rotation"]).yaw_pitch_roll[0]
            elif same_prev:
                # Backward difference (last frame of scene)
                prev_pos = all_frames[i - 1]["ego2global_translation"]
                next_pos = all_frames[i]["ego2global_translation"]
                dt = (all_frames[i]["timestamp"] - all_frames[i - 1]["timestamp"]) / 1e6
                prev_heading = Quaternion(*all_frames[i - 1]["ego2global_rotation"]).yaw_pitch_roll[0]
                next_heading = Quaternion(*all_frames[i]["ego2global_rotation"]).yaw_pitch_roll[0]
            else:
                continue  # isolated frame, skip

            if dt > 0:
                vx_global = (next_pos[0] - prev_pos[0]) / dt
                vy_global = (next_pos[1] - prev_pos[1]) / dt
                yaw_rate = _normalize_angle(next_heading - prev_heading) / dt

                # Convert global frame velocity to body frame
                ego_quat = Quaternion(*all_frames[i]["ego2global_rotation"])
                heading = ego_quat.yaw_pitch_roll[0]
                cos_h = np.cos(heading)
                sin_h = np.sin(heading)
                vx_body = vx_global * cos_h + vy_global * sin_h
                vy_body = -vx_global * sin_h + vy_global * cos_h
                all_frames[i]["ego_dynamic_state"] = [
                    vx_body, vy_body, ego_dynamic[2], ego_dynamic[3], yaw_rate,
                ]

    # ── Second pass: compute body-frame acceleration from position second differences ──
    # Matches sparse_converter_w_map_parallel.py get_ego_status_no_canbus():
    #   v_bwd = (p_cur - p_prev) / dt0,  v_fwd = (p_next - p_cur) / dt1
    #   a_global = 2 * (v_fwd - v_bwd) / (dt0 + dt1)
    #   a_body  = R_global_to_ego @ a_global
    for i in range(n):
        ego_dynamic = all_frames[i]["ego_dynamic_state"]
        same_prev = (i - 1 >= 0 and all_frames[i]["scene_token"] == all_frames[i - 1]["scene_token"])
        same_next = (i + 1 < n and all_frames[i]["scene_token"] == all_frames[i + 1]["scene_token"])

        ax_body, ay_body = 0.0, 0.0
        if same_prev and same_next:
            p_pos = all_frames[i - 1]["ego2global_translation"]
            c_pos = all_frames[i]["ego2global_translation"]
            n_pos = all_frames[i + 1]["ego2global_translation"]
            dt0 = (all_frames[i]["timestamp"] - all_frames[i - 1]["timestamp"]) / 1e6
            dt1 = (all_frames[i + 1]["timestamp"] - all_frames[i]["timestamp"]) / 1e6
            if dt0 > 1e-6 and dt1 > 1e-6:
                vx_bwd = (c_pos[0] - p_pos[0]) / dt0
                vy_bwd = (c_pos[1] - p_pos[1]) / dt0
                vx_fwd = (n_pos[0] - c_pos[0]) / dt1
                vy_fwd = (n_pos[1] - c_pos[1]) / dt1
                dt_sum = dt0 + dt1
                ax_global = 2.0 * (vx_fwd - vx_bwd) / dt_sum
                ay_global = 2.0 * (vy_fwd - vy_bwd) / dt_sum

                ego_quat = Quaternion(*all_frames[i]["ego2global_rotation"])
                heading = ego_quat.yaw_pitch_roll[0]
                cos_h = np.cos(heading)
                sin_h = np.sin(heading)
                ax_body =  ax_global * cos_h + ay_global * sin_h
                ay_body = -ax_global * sin_h + ay_global * cos_h

        all_frames[i]["ego_dynamic_state"] = [
            ego_dynamic[0], ego_dynamic[1], ax_body, ay_body, ego_dynamic[4],
        ]


def compute_body_frame_velocity_from_positions(
    frames: List[Dict[str, Any]],
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> None:
    """
    Recalculate velocity and yaw rate for a range of frames using central difference.
    Converts global frame velocity to body frame. Modifies frames in-place.

    Unlike compute_body_frame_velocity(), this does NOT check for zero velocity —
    it always overwrites. Used after shifting ego positions (e.g., Hermite interpolation).

    Uses central difference where possible; falls back to forward/backward difference
    at the boundaries of [start_idx, end_idx].

    :param frames: list of NavSim-format frame dicts
    :param start_idx: first frame index to update (inclusive)
    :param end_idx: last frame index to update (inclusive). If None, updates up to len-1.
    """
    import numpy as np
    from pyquaternion import Quaternion

    if end_idx is None:
        end_idx = len(frames) - 1  # can't compute velocity for last frame with forward diff

    for i in range(start_idx, end_idx):
        has_prev = (i - 1 >= start_idx)
        has_next = (i + 1 < len(frames))

        if has_prev and has_next:
            # Central difference
            p_pos = frames[i - 1]["ego2global_translation"]
            n_pos = frames[i + 1]["ego2global_translation"]
            dt = (frames[i + 1]["timestamp"] - frames[i - 1]["timestamp"]) / 1e6
            p_heading = Quaternion(*frames[i - 1]["ego2global_rotation"]).yaw_pitch_roll[0]
            n_heading = Quaternion(*frames[i + 1]["ego2global_rotation"]).yaw_pitch_roll[0]
        elif has_next:
            # Forward difference (boundary)
            p_pos = frames[i]["ego2global_translation"]
            n_pos = frames[i + 1]["ego2global_translation"]
            dt = (frames[i + 1]["timestamp"] - frames[i]["timestamp"]) / 1e6
            p_heading = Quaternion(*frames[i]["ego2global_rotation"]).yaw_pitch_roll[0]
            n_heading = Quaternion(*frames[i + 1]["ego2global_rotation"]).yaw_pitch_roll[0]
        elif has_prev:
            # Backward difference (boundary)
            p_pos = frames[i - 1]["ego2global_translation"]
            n_pos = frames[i]["ego2global_translation"]
            dt = (frames[i]["timestamp"] - frames[i - 1]["timestamp"]) / 1e6
            p_heading = Quaternion(*frames[i - 1]["ego2global_rotation"]).yaw_pitch_roll[0]
            n_heading = Quaternion(*frames[i]["ego2global_rotation"]).yaw_pitch_roll[0]
        else:
            continue

        if dt > 0:
            vx_global = (n_pos[0] - p_pos[0]) / dt
            vy_global = (n_pos[1] - p_pos[1]) / dt
            yaw_rate = _normalize_angle(n_heading - p_heading) / dt

            # Convert global frame velocity to body frame
            ego_quat = Quaternion(*frames[i]["ego2global_rotation"])
            heading = ego_quat.yaw_pitch_roll[0]
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            vx_body = vx_global * cos_h + vy_global * sin_h
            vy_body = -vx_global * sin_h + vy_global * cos_h
            old_dyn = frames[i]["ego_dynamic_state"]
            frames[i]["ego_dynamic_state"] = [
                vx_body, vy_body, old_dyn[2], old_dyn[3], yaw_rate,
            ]

    # Second pass: compute body-frame acceleration from position second differences.
    # Uses central difference (v_fwd - v_bwd) / (dt1 + dt2) where v = dx/dt.
    # Position data is valid for all frames (including outside [start_idx, end_idx]).
    for i in range(start_idx, end_idx):
        has_prev = (i - 1 >= 0)
        has_next = (i + 1 < len(frames))

        ax_global, ay_global = 0.0, 0.0
        if has_prev and has_next:
            p_pos = frames[i - 1]["ego2global_translation"]
            c_pos = frames[i]["ego2global_translation"]
            n_pos = frames[i + 1]["ego2global_translation"]
            dt1 = (frames[i]["timestamp"] - frames[i - 1]["timestamp"]) / 1e6   # t_i   - t_{i-1}
            dt2 = (frames[i + 1]["timestamp"] - frames[i]["timestamp"]) / 1e6   # t_{i+1} - t_i
            if dt1 > 0 and dt2 > 0:
                # (v_fwd - v_bwd) / ((dt1+dt2)/2) with non-uniform spacing
                vx_fwd = (n_pos[0] - c_pos[0]) / dt2
                vy_fwd = (n_pos[1] - c_pos[1]) / dt2
                vx_bwd = (c_pos[0] - p_pos[0]) / dt1
                vy_bwd = (c_pos[1] - p_pos[1]) / dt1
                dt_sum = dt1 + dt2
                ax_global = 2.0 * (vx_fwd - vx_bwd) / dt_sum
                ay_global = 2.0 * (vy_fwd - vy_bwd) / dt_sum
        # boundary frames (has_prev XOR has_next): leave acceleration as 0.0

        # Rotate global acceleration to body frame
        ego_quat = Quaternion(*frames[i]["ego2global_rotation"])
        heading = ego_quat.yaw_pitch_roll[0]
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        ax_body =  ax_global * cos_h + ay_global * sin_h
        ay_body = -ax_global * sin_h + ay_global * cos_h

        old_dyn = frames[i]["ego_dynamic_state"]
        frames[i]["ego_dynamic_state"] = [old_dyn[0], old_dyn[1], ax_body, ay_body, old_dyn[4]]


def group_frames_by_scene(all_frames: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group frames by scene_token and sort each group by frame_idx.
    
    :param all_frames: list of NavSim-format frame dicts
    :return: dict mapping scene_token → sorted list of frame dicts
    """
    scenes_dict: Dict[str, List[Dict[str, Any]]] = {}
    for frame in all_frames:
        scene_token = frame["scene_name"]
        if scene_token not in scenes_dict:
            scenes_dict[scene_token] = []
        scenes_dict[scene_token].append(frame)

    for scene_token in scenes_dict:
        scenes_dict[scene_token].sort(key=lambda x: x["frame_idx"])

    return scenes_dict


def split_scene_into_subscenes(
    frame_list: List[Dict[str, Any]],
    num_frames: int,
    frame_interval: int,
) -> List[List[Dict[str, Any]]]:
    """
    Split a scene's frame list into overlapping sub-scenes.
    
    :param frame_list: sorted list of frame dicts for one scene
    :param num_frames: total frames per sub-scene (history + future)
    :param frame_interval: step size between sub-scene starts (None → non-overlapping)
    :return: list of sub-scene frame lists (only full-length sub-scenes)
    """
    if frame_interval is None:
        frame_interval = num_frames  # Non-overlapping
    subscenes = [
        frame_list[i : i + num_frames]
        for i in range(0, len(frame_list), frame_interval)
    ]
    # Filter out short sub-scenes
    return [s for s in subscenes if len(s) >= num_frames]


def filter_scenes_v2xreal(
    pkl_path: Path, scene_filter: SceneFilter
) -> Tuple[Dict[str, FrameList], List[str], Dict[str, FrameList]]:
    """
    Load a set of scenes from V2X-Real pkl file, while applying scene filter configuration.
    :param pkl_path: path to V2X-Real pkl file (e.g., spd_infos_temporal_test.pkl)
    :param scene_filter: scene filtering configuration class
    :return: dictionary of frame lists keyed by token, and list of final frame tokens
    """
    
    # Load V2X-Real pkl file
    print(f"Loading V2X-Real pkl from {pkl_path}...")
    all_frames = load_v2xreal_pkl(pkl_path)
    print(f"Loaded {len(all_frames)} frames")
    
    # Convert frames to NavSim format
    print("Converting frames to NavSim format...")
    all_frames = [convert_v2xreal_to_navsim_format(frame) for frame in tqdm(all_frames, desc="Converting frames")]
    
    # Calculate velocity from position differences (global → body frame)
    print("Calculating velocities from position differences...")
    compute_body_frame_velocity(all_frames)
    
    # Group frames by scene_token and sort
    scenes_dict = group_frames_by_scene(all_frames)
    print(f"Grouped into {len(scenes_dict)} scenes")
    
    # Now split each scene into sub-scenes according to scene_filter
    filtered_scenes: Dict[str, FrameList] = {}
    final_frame_tokens: List[str] = []
    
    filter_tokens = scene_filter.tokens is not None
    if filter_tokens:
        tokens = set(scene_filter.tokens)
    
    for scene_token, frame_list in tqdm(scenes_dict.items(), desc="Filtering scenes"):
        for sub_scene in split_scene_into_subscenes(frame_list, scene_filter.num_frames, scene_filter.frame_interval):
            
            # Filter scenes with no route (if required)
            if scene_filter.has_route and len(sub_scene[scene_filter.num_history_frames - 1]["roadblock_ids"]) == 0:
                # V2X-Real may not have roadblock_ids — silently skip route check
                pass
            
            # Get token from the current frame (history end)
            token = sub_scene[scene_filter.num_history_frames - 1]["token"]
            
            # Filter by token if specified
            if filter_tokens and token not in tokens:
                continue
            
            # Use scene_token + "_" + token as key to avoid collisions across scenes
            unique_key = scene_token + "_" + token
            filtered_scenes[unique_key] = sub_scene
            final_frame_token = sub_scene[scene_filter.num_frames - 1]["token"]
            final_frame_tokens.append(final_frame_token)
            
            # Stop if max_scenes reached
            if (scene_filter.max_scenes is not None) and (len(filtered_scenes) >= scene_filter.max_scenes):
                break
        
        if (scene_filter.max_scenes is not None) and (len(filtered_scenes) >= scene_filter.max_scenes):
            break
    
    print(f"Filtered to {len(filtered_scenes)} scenes")
    return filtered_scenes, final_frame_tokens, scenes_dict


####################### extract ego pose pkl for making lane connector
def extract_ego_polyline_global(scene_dict_list: List[Dict[str, Any]]) -> np.ndarray:
    """
    Return ego polyline as [T,2] global xy from frames.
    """
    xy = []
    for fr in scene_dict_list:
        t = fr["ego2global_translation"]
        xy.append([float(t[0]), float(t[1])])
    return np.asarray(xy, dtype=np.float64)


def extract_ego_yaws(scene_dict_list: List[Dict[str, Any]]) -> np.ndarray:
    yaws = []
    for fr in scene_dict_list:
        q = Quaternion(*fr["ego2global_rotation"])
        yaw = q.yaw_pitch_roll[0]
        yaws.append(float(yaw))
    return np.asarray(yaws, dtype=np.float64)


def _wrap_pi(a: float) -> float:
    """wrap angle to (-pi, pi]."""
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def _lane_tangent_heading(lane_obj, xy: np.ndarray, delta_m: float = 1.0) -> Optional[float]:
    """
    Estimate the tangent heading at the point on the lane centerline closest to xy.
    Returns None on failure.
    """
    from shapely.geometry import Point as ShapelyPoint
    try:
        path = lane_obj.baseline_path
        states = path.discrete_path
        if states is None or len(states) < 2:
            return None
        ls = LineString([(float(s.x), float(s.y)) for s in states])
    except Exception:
        return None

    if ls.length < 1e-6:
        return None

    s = float(ls.project(ShapelyPoint(float(xy[0]), float(xy[1]))))
    s0 = max(0.0, min(ls.length, s - delta_m))
    s1 = max(0.0, min(ls.length, s + delta_m))

    p0 = ls.interpolate(s0)
    p1 = ls.interpolate(s1)

    dx, dy = (p1.x - p0.x), (p1.y - p0.y)
    if dx * dx + dy * dy < 1e-9:
        return None
    return float(math.atan2(dy, dx))


def _nearest_lane_id(map_api: "V2XRealMapWrapper", xy: np.ndarray, search_radius: float = 8.0) -> Optional[str]:
    """
    Find nearest lane by polygon distance (cheap brute-force).
    xy: [2]
    """
    from shapely.geometry import Point as ShapelyPoint
    pt = ShapelyPoint(float(xy[0]), float(xy[1]))
    best = None
    best_d = float("inf")
    for lane in map_api._get_lanes():
        poly = lane.polygon
        if poly.is_empty:
            continue
        d = poly.distance(pt)
        if d < best_d:
            best_d = d
            best = lane.id
    if best is None:
        return None
    if best_d > search_radius:
        return best
    return best


def _nearest_lane_id_with_heading(
    map_api: "V2XRealMapWrapper",
    xy: np.ndarray,
    ego_yaw: float,
    search_radius: float = 20.0,
    top_k: int = 30,
    w_dist: float = 1.0,
    w_ang: float = 8.0,
) -> Optional[str]:
    """
    Find nearest lane using both:
      - distance from point to lane polygon
      - heading consistency between ego_yaw and lane centerline tangent

    Score = w_dist * dist + w_ang * |angle_diff|
    """
    from shapely.geometry import Point as ShapelyPoint
    pt = ShapelyPoint(float(xy[0]), float(xy[1]))

    candidates = []
    for lane in map_api._get_lanes():
        poly = lane.polygon
        if poly.is_empty:
            continue
        d = float(poly.distance(pt))
        if d <= search_radius:
            candidates.append((d, lane))

    if not candidates:
        # fallback: minimum distance over all lanes
        best, best_d = None, float("inf")
        for lane in map_api._get_lanes():
            poly = lane.polygon
            if poly.is_empty:
                continue
            d = float(poly.distance(pt))
            if d < best_d:
                best_d = d
                best = lane.id
        return best

    candidates.sort(key=lambda x: x[0])
    candidates = candidates[:top_k]

    best_id, best_score = None, float("inf")
    for d, lane in candidates:
        lane_h = _lane_tangent_heading(lane, xy)
        if lane_h is None:
            ang = math.pi
        else:
            ang = abs(_wrap_pi(lane_h - float(ego_yaw)))

        score = w_dist * d + w_ang * ang
        if score < best_score:
            best_score = score
            best_id = lane.id

    return best_id


def _buffer_linestring_as_polygon(xy: np.ndarray, width_m: float = 4.0) -> Polygon:
    """
    Make a drivable polygon for connector from centerline polyline.
    """
    if xy is None or len(xy) < 2:
        return Polygon()
    ls = LineString([(float(x), float(y)) for x, y in xy])
    return ls.buffer(width_m * 0.5, cap_style=2, join_style=2)


def _lines_from_intersection(geom):
    from shapely.geometry import LineString as ShpLineString, MultiLineString, GeometryCollection
    out = []
    if geom is None or geom.is_empty:
        return out
    if isinstance(geom, ShpLineString):
        out.append(geom)
    elif isinstance(geom, MultiLineString):
        out.extend(list(geom.geoms))
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            out.extend(_lines_from_intersection(g))
    return out


def build_lane_connectors_from_gt(
    scene_dict_list: List[Dict[str, Any]],
    map_api: "V2XRealMapWrapper",
    history_end_index: Optional[int],
    min_inside_len: int = 3,
    connector_width_m: float = 4.0,
) -> List[V2XLaneConnectorObject]:
    """
    - Intersect the ego polyline against every junction.
    - Create a connector for each LineString inside a junction (there may be several).
    - Estimate incoming/outgoing from lane ids near segment start-1 / end+1.
    """
    ego_xy = extract_ego_polyline_global(scene_dict_list)  # [T,2]
    if ego_xy is None or len(ego_xy) < 2:
        return []

    ego_xy_fut = ego_xy
    if len(ego_xy_fut) < 2:
        return []

    ego_ls = LineString([(float(x), float(y)) for x, y in ego_xy_fut])

    junction_polys = map_api._get_junctions()
    if not junction_polys:
        return []

    connectors: List[V2XLaneConnectorObject] = []
    cid = 0

    for j_idx, jpoly in enumerate(junction_polys):
        if jpoly is None or jpoly.is_empty:
            continue

        inter = ego_ls.intersection(jpoly)
        seg_lines = _lines_from_intersection(inter)

        for seg in seg_lines:
            if seg is None or seg.is_empty:
                continue
            coords = np.asarray(seg.coords, dtype=np.float64)
            if coords.shape[0] < min_inside_len:
                continue

            poly = _buffer_linestring_as_polygon(coords, width_m=connector_width_m)

            def nearest_ego_yaw(scene_dict_list, xy):
                yaws = extract_ego_yaws(scene_dict_list)
                d2 = np.sum((ego_xy - xy[None, :])**2, axis=1)
                return float(yaws[int(np.argmin(d2))])

            in_lane  = _nearest_lane_id_with_heading(map_api, coords[0],  nearest_ego_yaw(scene_dict_list, coords[0]))
            out_lane = _nearest_lane_id_with_heading(map_api, coords[-1], nearest_ego_yaw(scene_dict_list, coords[-1]))
            if in_lane == out_lane:
                in_lane = None

            conn = V2XLaneConnectorObject(
                _id=f"{j_idx}_{cid:04d}",
                _centerline_xy=coords,
                _polygon=poly,
                _incoming_lane_ids=[in_lane] if in_lane is not None else [],
                _outgoing_lane_ids=[out_lane] if out_lane is not None else [],
            )
            connectors.append(conn)
            cid += 1

    return connectors


import matplotlib.pyplot as plt


def visualize_lanes_connectors_and_gt(
    map_api: "V2XRealMapWrapper",
    scene_dict_list: list,
    out_path: Optional[str] = None,
    radius_m: float = 120.0,
    center_on_history_end: int = 0,
    show_ego_traj: bool = False,
    show_lane_polygons: bool = True,
    show_lane_centerlines: bool = True,
    show_connector_arrows: bool = True,
    show_inout_lanes: bool = True,
    show_inout_arrows: bool = True,
    label_connectors: bool = True,
    label_inout: bool = True,
    only_draw_inout_near_connector: bool = True,
    inout_max_dist_m: float = 40.0,
):
    def _plot_linestring(ax, ls: LineString, **kw):
        if ls is None:
            return
        if getattr(ls, "is_empty", False):
            return
        x, y = ls.xy
        ax.plot(x, y, **kw)

    def _arrow_on_linestring(ax, ls: LineString, frac=0.6, length=6.0, **kw):
        if ls is None or getattr(ls, "is_empty", False) or ls.length < 1e-6:
            return
        s = float(np.clip(frac, 0.05, 0.95) * ls.length)
        p0 = ls.interpolate(max(0.0, s - 0.01 * ls.length))
        p1 = ls.interpolate(min(ls.length, s + 0.01 * ls.length))
        dx, dy = (p1.x - p0.x), (p1.y - p0.y)
        n = (dx * dx + dy * dy) ** 0.5 + 1e-9
        dx, dy = dx / n * length, dy / n * length
        ax.quiver([p0.x], [p0.y], [dx], [dy],
                  angles="xy", scale_units="xy", scale=1,
                  width=0.006, headwidth=4.0, headlength=6.0, **kw)

    def _as_linestring_from_xy(xy: np.ndarray) -> Optional[LineString]:
        if xy is None:
            return None
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[0] < 2:
            return None
        return LineString(xy)

    def _connector_incoming_ids(conn):
        for key in ["incoming_lane_ids", "incoming_ids", "incoming_lanes", "incoming_lane_tokens"]:
            v = getattr(conn, key, None)
            if v:
                return list(v)
        return []

    def _connector_outgoing_ids(conn):
        for key in ["outgoing_lane_ids", "outgoing_ids", "outgoing_lanes", "outgoing_lane_tokens"]:
            v = getattr(conn, key, None)
            if v:
                return list(v)
        return []

    def _lane_centerline_ls(lane_obj):
        try:
            path = lane_obj.baseline_path
        except Exception:
            return None
        states = path.discrete_path
        xy = [(float(s.x), float(s.y)) for s in states]
        return LineString(xy)

    def _dist_ls_to_ls(ls1: LineString, ls2: LineString) -> float:
        if ls1 is None or ls2 is None:
            return 1e9
        try:
            return float(ls1.distance(ls2))
        except Exception:
            return 1e9

    ego_xy = extract_ego_polyline_global(scene_dict_list)
    cx, cy = float(ego_xy[center_on_history_end, 0]), float(ego_xy[center_on_history_end, 1])

    fig, ax = plt.subplots(figsize=(8, 8))

    lanes = list(map_api._get_lanes())
    lanes_by_id = {lane.id: lane for lane in lanes if hasattr(lane, "id")}

    if show_lane_polygons:
        for lane in lanes:
            poly = lane.polygon
            if poly.is_empty:
                continue
            x, y = poly.exterior.xy
            ax.plot(x, y, linewidth=0.7, alpha=0.35, color="k")

    if show_lane_centerlines:
        for lane in lanes:
            ls = _lane_centerline_ls(lane)
            if ls is None:
                continue
            _plot_linestring(ax, ls, linewidth=1.0, alpha=0.25, color="0.3")

    for j in map_api._get_junctions():
        if j.is_empty:
            continue
        x, y = j.exterior.xy
        ax.plot(x, y, linewidth=1.5, alpha=0.7, color="tab:pink")

    if show_ego_traj:
        ax.plot(
            ego_xy[:, 0], ego_xy[:, 1],
            linewidth=2.0, marker="o", markersize=2.0, alpha=0.9,
            color="tab:cyan",
        )

    for c in map_api._get_lane_connectors():
        cl = getattr(c, "_centerline_xy", None)
        cl = np.asarray(cl, dtype=float) if cl is not None else None

        if cl is not None and len(cl) >= 2:
            ax.plot(cl[:, 0], cl[:, 1], linewidth=3.0, alpha=0.9, color="red")

            conn_ls = _as_linestring_from_xy(cl)
            if show_connector_arrows and conn_ls is not None:
                _arrow_on_linestring(ax, conn_ls, frac=0.6, length=7.0, color="red")

            if label_connectors:
                ax.text(cl[0, 0], cl[0, 1], getattr(c, "id", "connector"), fontsize=9, color="red",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="none"))
        else:
            conn_ls = None

        incoming_ids = _connector_incoming_ids(c)
        outgoing_ids = _connector_outgoing_ids(c)

        if show_inout_lanes:
            for lid in incoming_ids:
                lane = lanes_by_id.get(lid, None)
                if lane is None:
                    continue
                ls = _lane_centerline_ls(lane)
                if ls is None:
                    continue
                if only_draw_inout_near_connector and conn_ls is not None:
                    if _dist_ls_to_ls(ls, conn_ls) > inout_max_dist_m:
                        continue
                _plot_linestring(ax, ls, linewidth=2.2, alpha=0.9, color="tab:blue", linestyle="--")
                if show_inout_arrows:
                    _arrow_on_linestring(ax, ls, frac=0.85, length=6.0, color="tab:blue")
                if label_inout:
                    p = ls.interpolate(0.9 * ls.length)
                    ax.text(p.x, p.y, f"in:{lid}", fontsize=7, color="tab:blue")

            for lid in outgoing_ids:
                lane = lanes_by_id.get(lid, None)
                if lane is None:
                    continue
                ls = _lane_centerline_ls(lane)
                if ls is None:
                    continue
                if only_draw_inout_near_connector and conn_ls is not None:
                    if _dist_ls_to_ls(ls, conn_ls) > inout_max_dist_m:
                        continue
                _plot_linestring(ax, ls, linewidth=2.2, alpha=0.9, color="tab:green", linestyle="-.")
                if show_inout_arrows:
                    _arrow_on_linestring(ax, ls, frac=0.15, length=6.0, color="tab:green")
                if label_inout:
                    p = ls.interpolate(0.1 * ls.length)
                    ax.text(p.x, p.y, f"out:{lid}", fontsize=7, color="tab:green")

    handles = [
        Line2D([0], [0], color="0.2", lw=0.7, alpha=0.25, label="Lane polygon"),
        Line2D([0], [0], color="0.3", lw=1.0, alpha=0.25, label="Lane centerline"),
        Line2D([0], [0], color="tab:pink", lw=1.5, alpha=0.7, label="Junction polygon"),
        Line2D([0], [0], color="tab:cyan", lw=2.0, marker="o", markersize=4, label="GT ego trajectory"),
        Line2D([0], [0], color="red", lw=3.0, label="GT-derived lane connector"),
        Line2D([0], [0], color="tab:blue", lw=2.2, ls="--", label="Incoming lane(s) to connector"),
        Line2D([0], [0], color="tab:green", lw=2.2, ls="-.", label="Outgoing lane(s) from connector"),
    ]
    ax.legend(handles=handles, loc="best", framealpha=0.9)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(cx - radius_m, cx + radius_m)
    ax.set_ylim(cy - radius_m, cy + radius_m)
    ax.set_title("Lanes + Junctions + GT ego traj + GT-derived lane connectors (+incoming/outgoing)")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def select_lane_id_for_ego_point(
    map_api: "V2XRealMapWrapper",
    ego_xy: np.ndarray | list | tuple,
    *,
    use_covers: bool = True,                 # prefer covers over contains
    fallback_to_nearest_centerline: bool = True,
    max_fallback_dist_m: float = 10.5,       # centerline distance limit when no candidate
    return_debug: bool = False,
):
    """
    Returns:
      - lane_id (str|None)  or  (lane_id, debug_dict) if return_debug=True

    Logic:
      1) Collect lane candidates where polygon.covers(pt) (or contains).
      2) Among candidates, pick the lane with minimum centerline distance.
      3) If no candidate (boundary / map hole), pick the lane with minimum
         centerline distance over all lanes (optional, distance-limited).
    """
    def _lane_centerline_ls(lane_obj) -> LineString | None:
        """Build a centerline LineString from lane_obj.baseline_path."""
        try:
            path = lane_obj.baseline_path
            states = path.discrete_path
            if states is None or len(states) < 2:
                return None
            xy = [(float(s.x), float(s.y)) for s in states]
            return LineString(xy)
        except Exception:
            return None
    ego_xy = np.asarray(ego_xy, dtype=float).reshape(-1)
    assert ego_xy.shape[0] >= 2
    pt = Point(float(ego_xy[0]), float(ego_xy[1]))

    lanes = list(map_api._get_lanes())
    hits = []
    for lane in lanes:
        poly = lane.polygon
        if poly is None or poly.is_empty:
            continue
        ok = poly.covers(pt) if use_covers else poly.contains(pt)
        if ok:
            hits.append(lane)

    def best_by_centerline(cands):
        best_id, best_d = None, float("inf")
        for lane in cands:
            ls = _lane_centerline_ls(lane)
            if ls is None or ls.is_empty:
                continue
            d = float(ls.distance(pt))
            if d < best_d:
                best_d = d
                best_id = lane.id
        return best_id, best_d

    chosen_id = None
    reason = None
    best_d = None

    if hits:
        chosen_id, best_d = best_by_centerline(hits)
        if chosen_id is not None:
            reason = "covers-hit + min centerline dist"
        else:
            # case where only lanes with failed centerline generation hit -> fallback to minimum polygon distance
            best_lane, best_pd = None, float("inf")
            for lane in hits:
                d = float(lane.polygon.distance(pt))
                if d < best_pd:
                    best_pd = d
                    best_lane = lane
            chosen_id = best_lane.id if best_lane is not None else None
            best_d = best_pd
            reason = "covers-hit + fallback min polygon dist"
    else:
        if fallback_to_nearest_centerline:
            chosen_id, best_d = best_by_centerline(lanes)
            if chosen_id is not None and best_d <= max_fallback_dist_m:
                reason = "no-hit + fallback nearest centerline"
            else:
                chosen_id = None
                reason = "no-hit + fallback too far/failed"
        else:
            chosen_id = None
            reason = "no-hit + no fallback"

    if not return_debug:
        return chosen_id

    debug = {
        "reason": reason,
        "best_distance": None if best_d is None else float(best_d),
        "num_hits": len(hits),
    }
    return chosen_id, debug

### for computing driving command
def compute_driving_command_from_gt_endpoint_y(
    s2_frames: list[dict],
    num_history: int,
    *,
    lookahead_idx: int = None,   # None uses the last frame as goal
    y_thresh_m: float = 2.0,            # body-frame y threshold (meters)
    require_forward_progress_m: float = 1.0,  # goal must be at least this far forward (x) to be valid
) -> int:
    """
    Transform the GT ego future trajectory into the current ego frame, then
    classify left/right/straight from the body-frame y of the goal endpoint.

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
    lookahead_idx: int = None,   # None uses the last frame
    y_thresh_m: float = 2.0,
    require_forward_progress_m: float = 1.0,
    cmd: int = None,             # pass in if already computed
    cmd_to_name: dict[int, str] = None,
    out_path: str = None,        # save path; None calls plt.show()
    figsize: tuple[int, int] = (12, 5),
    title: str = None,
    show_global_view: bool = False,      # left: global traj
    show_ego_view: bool = True,         # right: ego (frame) traj
    ego_view_samples: int = None # None uses cur..end entirely, otherwise cur..cur+N
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
            bbox_to_anchor=(0.00, 1.02),   # (x, y) in axes coords, y>1 => above top
            borderaxespad=0.0,
            framealpha=0.9,
            ncol=1,                        # set to 2 if there are many items
        )

        # 2) place text above top on the "right" (x=1.0, ha=right so it doesn't overlap the legend)
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
        out_path = f"exp_debug/driving_cmd_stage1/{fr0['token']}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

class SceneLoaderV2XReal:
    """Data loader for V2X-Real dataset scenes."""

    def __init__(
        self,
        pkl_path: Path,
        scene_filter: SceneFilter,
        sensor_config: SensorConfig = SensorConfig.build_no_sensors(),
        sensor_blob_path: Path = None,
        map_root: Path = None,
        connector_cache_dir: Optional[Path] = None,
        connector_force_recompute: bool = False,
    ):
        """
        Initializes the V2X-Real scene data loader.
        :param pkl_path: path to V2X-Real pkl file (e.g., spd_infos_temporal_test.pkl)
        :param scene_filter: dataclass for scene filtering specification
        :param sensor_config: dataclass for sensor loading specification, defaults to no sensors
        :param sensor_blob_path: root directory for sensor data (optional)
        :param map_root: root directory for V2X-Real map data (optional)
        :param connector_cache_dir: directory for precomputed lane connector caches (optional)
        :param connector_force_recompute: if True, recompute connectors even if cached
        """
        self.scene_frames_dicts, _, self.full_scenes_dicts = filter_scenes_v2xreal(pkl_path, scene_filter)
        self._sensor_config = sensor_config
        self._sensor_blob_path = sensor_blob_path
        self._scene_filter = scene_filter
        self._map_root = map_root
        self._connector_cache_dir = Path(connector_cache_dir) if connector_cache_dir is not None else None
        self._connector_force_recompute = connector_force_recompute
    
    @property
    def tokens(self) -> List[str]:
        """Return all available scene tokens."""
        return sorted(list(self.scene_frames_dicts.keys()))
    
    def __len__(self) -> int:
        """Return number of scenes."""
        return len(self.tokens)
    
    def __getitem__(self, idx: int) -> str:
        """Return token at index."""
        return self.tokens[idx]
    

    
    def get_scene_from_token(self, token: str) -> Scene:
        """
        Loads scene given a scene identifier string (token).
        :param token: scene identifier string.
        :return: scene dataclass
        """
        assert token in self.tokens, f"Token {token} not found in loaded scenes"
        
        scene_dict = self.scene_frames_dicts[token]

        ### compute driving_command
        cmd = compute_driving_command_from_gt_endpoint_y(
            s2_frames=scene_dict,
            num_history=self._scene_filter.num_history_frames,
            lookahead_idx=None,
            y_thresh_m=2.0,
        )

        # visualize_gt_endpoint_y_command(
        #     s2_frames=scene_dict,
        #     num_history=self._scene_filter.num_history_frames,
        #     lookahead_idx=None,
        #     y_thresh_m=2.0,
        #     cmd=cmd,
        #     out_path=None,  # display on screen
        # )



        for fr in scene_dict:
            fr["driving_command"] = int(cmd)
        
        #########################

        
        
        # Use custom creation method with actual V2X-Real map
        scene = self._create_scene_v2xreal(
            scene_dict,
            self._sensor_blob_path,
            num_history_frames=self._scene_filter.num_history_frames,
            num_future_frames=self._scene_filter.num_future_frames,
            sensor_config=self._sensor_config,
            map_root=self._map_root,
            connector_cache_dir=self._connector_cache_dir,
        )

        return scene
    
    #### connector precompute utils
    def _connector_path(self, scene_token: str) -> Optional[Path]:
        if self._connector_cache_dir is None:
            return None
        return self._connector_cache_dir / "lane_connectors" / f"{scene_token}.pkl"

    def _save_connectors(self, scene_token: str, connectors: List[V2XLaneConnectorObject]) -> None:
        out = self._connector_path(scene_token)
        if out is None:
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(connectors, f)
        os.replace(tmp, out)

    def _load_connectors(self, scene_token: str) -> List[V2XLaneConnectorObject]:
        p = self._connector_path(scene_token)
        if p is None or (not p.exists()):
            return []
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return []

    def precompute_from_scene(
        self,
        min_inside_len: int = 3,
        connector_width_m: float = 4.0,
        visualize_every: int = 0,
        visualize_out_dir: Optional[Path] = "exp2/precomputed_lane_connector",
    ) -> None:
        assert self._map_root is not None, "map_root must be set"
        assert self._connector_cache_dir is not None, "connector_cache_dir must be set"

        if visualize_out_dir is not None:
            visualize_out_dir = Path(visualize_out_dir)
            visualize_out_dir.mkdir(parents=True, exist_ok=True)

        map_api = V2XRealMapWrapper(map_root=Path(self._map_root))

        for i, (scene_token, full_frames) in enumerate(tqdm(self.full_scenes_dicts.items(), desc="precompute connectors")):
            out_path = self._connector_path(scene_token)
            if out_path is not None and out_path.exists() and (not self._connector_force_recompute):
                continue

            if full_frames is None or len(full_frames) < 2:
                self._save_connectors(scene_token, [])
                continue

            connectors = build_lane_connectors_from_gt(
                scene_dict_list=full_frames,
                map_api=map_api,
                history_end_index=None,
                min_inside_len=min_inside_len,
                connector_width_m=connector_width_m,
            )

            self._save_connectors(scene_token, connectors)

            if visualize_every and (i % visualize_every == 0):
                try:
                    map_api.set_lane_connectors(connectors)
                    out_img = None
                    if visualize_out_dir is not None:
                        out_img = str(visualize_out_dir / f"{scene_token}.png")
                    visualize_lanes_connectors_and_gt(
                        map_api=map_api,
                        scene_dict_list=full_frames,
                        out_path=out_img,
                        radius_m=120.0,
                        center_on_history_end=0,
                        show_ego_traj=True,
                    )
                except Exception as e:
                    print(f"[precompute] visualize failed: {scene_token}, err={e}")

    @staticmethod
    def _create_scene_v2xreal(
        scene_dict_list: List[Dict[str, Any]],
        sensor_blobs_path: Optional[Path],
        num_history_frames: int,
        num_future_frames: int,
        sensor_config: SensorConfig,
        map_root: Optional[Path] = None,
        connector_cache_dir: Optional[Path] = None,
    ) -> Scene:
        """
        Create Scene for V2X-Real data, bypassing standard map validation.
        :param scene_dict_list: list of frame dicts
        :param sensor_blobs_path: path to sensor blobs
        :param num_history_frames: number of history frames
        :param num_future_frames: number of future frames
        :param sensor_config: sensor config
        :param map_root: path to V2X-Real map root directory
        :return: Scene object
        """
        # Create scene metadata - use map_location as-is without validation
        scene_metadata = SceneMetadata(
            log_name=scene_dict_list[num_history_frames - 1]["log_name"],
            scene_token=scene_dict_list[num_history_frames - 1]["scene_token"],
            map_name=scene_dict_list[num_history_frames - 1]["map_location"],
            initial_token=scene_dict_list[num_history_frames - 1]["token"],
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
        )
        
        # Create V2X-Real map API wrapper
        effective_map_root = map_root or os.environ.get("V2XREAL_MAP_ROOT")
        if effective_map_root is None:
            raise ValueError("V2X-Real map root is required. Pass map_root or set V2XREAL_MAP_ROOT.")
        map_api = V2XRealMapWrapper(map_root=Path(effective_map_root))

        # Load and set lane connectors from cache if available
        connectors: List[V2XLaneConnectorObject] = []
        if connector_cache_dir is not None:
            scene_token = scene_dict_list[num_history_frames - 1]["scene_token"]
            p = Path(connector_cache_dir) / "lane_connectors" / f"{scene_token}.pkl"
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        connectors = pickle.load(f)
                except Exception:
                    connectors = []
        map_api.set_lane_connectors(connectors)
        if len(map_api._get_lane_connectors()) != 0:
            connector = map_api._get_lane_connectors()[0]
            connector_id = connector.id
            if len(connector.incoming_lane_ids) != 0:
                # try:
                incoming_lane_id = connector.incoming_lane_ids[0]
                # except:
            else:
                incoming_lane_id = connector.incoming_lane_ids
            outgoing_lane_id = connector.outgoing_lane_ids[0]

            route_lane_ids = [connector_id]

            for rb_id, lane_id_list in map_api.road_block_ids_dict.items():
                if len(incoming_lane_id) != 0:
                    if incoming_lane_id in lane_id_list:
                        route_lane_ids.extend(lane_id_list)
                if outgoing_lane_id in lane_id_list:
                    route_lane_ids.extend(lane_id_list)
            
        else:

            ego_xy = extract_ego_polyline_global(scene_dict_list)[num_history_frames - 1]  # relative to current frame

            lane_id, dbg = select_lane_id_for_ego_point(map_api, ego_xy, return_debug=True)

            if lane_id is None:
                ValueError("Can't select lane id")
            

            route_lane_ids = []
            for rb_id, lane_id_list in map_api.road_block_ids_dict.items():
                if lane_id in lane_id_list:
                    route_lane_ids.extend(lane_id_list)
        map_api.set_route_lane_ids(route_lane_ids)

       

        # Create frames without requiring map API
        frames: List[Scene.Frame] = []
        for frame_idx in range(len(scene_dict_list)):
            global_ego_status = Scene._build_ego_status(scene_dict_list[frame_idx])
            annotations = Scene._build_annotations(scene_dict_list[frame_idx])

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            
            # Build cameras and lidar only if sensors are requested
            cameras = None
            lidar = None
            if sensor_names:
                from navsim.common.dataclasses import Cameras, Lidar
                cameras = Cameras.from_camera_dict(
                    sensor_blobs_path=sensor_blobs_path,
                    camera_dict=scene_dict_list[frame_idx]["cams"],
                    sensor_names=sensor_names,
                )
                lidar = Lidar.from_lidar_dict(
                    sensor_blobs_path=sensor_blobs_path,
                    lidar_dict=scene_dict_list[frame_idx],
                    lidar_sensor_name="lidar",
                    sensor_names=sensor_names,
                )
            
            traffic_lights = scene_dict_list[frame_idx].get("traffic_light_status", [])
            
            frame = Frame(
                token=scene_dict_list[frame_idx]["token"],
                timestamp=scene_dict_list[frame_idx]["timestamp"],
                roadblock_ids=scene_dict_list[frame_idx].get("roadblock_ids", []),
                traffic_lights=traffic_lights,
                annotations=annotations,
                ego_status=global_ego_status,
                lidar=lidar,
                cameras=cameras,
            )
            frames.append(frame)
        # print(global_ego_status)
        # Create scene with V2X-Real map API
        scene = Scene(
            frames=frames,
            scene_metadata=scene_metadata,
            map_api=map_api,  # Use V2X-Real map API loaded via NuPlan
        )
        
        return scene
    
    def get_agent_input_from_token(self, token: str) -> AgentInput:
        """
        Loads agent input given a scene identifier string (token).
        :param token: scene identifier string.
        :return: agent input dataclass
        """
        scene = self.get_scene_from_token(token)
        return scene.get_agent_input()
    
    def get_tokens_list_per_log(self) -> Dict[str, List[str]]:
        """
        Collect tokens for each log/scene file.
        :return: dictionary of log names and tokens
        """
        # Group tokens by log_name (which is scene_token in V2X)
        tokens_per_log: Dict[str, List[str]] = {}
        
        for token, scene_dict in self.scene_frames_dicts.items():
            log_name = scene_dict[0]["log_name"]
            if log_name not in tokens_per_log:
                tokens_per_log[log_name] = []
            tokens_per_log[log_name].append(token)
        
        return tokens_per_log
