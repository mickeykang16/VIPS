from __future__ import annotations

import warnings
from typing import Any, Generator, List, Optional, Set, Tuple, Type, cast

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimeDuration, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import (
    TrafficLightStatusData,
    TrafficLightStatuses,
    TrafficLightStatusType,
    Transform,
)
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.database.maps_db.gpkg_mapsdb import MAP_LOCATIONS
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, SensorChannel, Sensors
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Scene
from navsim.planning.scenario_builder.navsim_scenario_utils import (
    annotations_to_detection_tracks,
    ego_status_to_ego_state,
    sample_future_indices,
    sample_past_indices,
)
import os

DUMMY_SCENARIO_TYPE = "unknown"
DUMMY_GOAL_STATE = StateSE2(0, 0, 0)


class NavSimScenario(AbstractScenario):
    """nuPlan interface for scenarios from NAVSIM logs."""

    def __init__(
        self,
        scene: Scene,
        map_root: str,
        map_version: str,
        ego_vehicle_parameters: VehicleParameters = get_pacifica_parameters(),
    ) -> None:
        """
        Initializes the NAVSIM scenario
        :param scene: dataclass describing scene in NAVSIM
        :param map_root: directory of maps
        :param map_version: string identifier of map version
        :param ego_vehicle_parameters: extend of ego vehicle, defaults to get_pacifica_parameters()
        """

        
        self._scene = scene
        
        # Calculate database_interval from frame timestamps
        # For V2X-Real, timestamps are in seconds (1, 2, 3, ...) with 10 Hz sampling
        # For NuPlan, timestamps are in microseconds
        if len(scene.frames) > 1:
            ts_diff = scene.frames[1].timestamp - scene.frames[0].timestamp
            if ts_diff > 1000:  # likely microseconds (NuPlan)
                self._database_interval = ts_diff / 1e6
            else:  # likely seconds (V2X-Real or already in seconds)
                self._database_interval = ts_diff
        else:
            self._database_interval = 0.5  # default fallback

        # map attributes
        self._map_root = map_root
        self._map_version = map_version

        self._scene_data = scene.scene_metadata
        self._map_name = self._scene_data.map_name
        self._map_name = self._map_name if self._map_name != "las_vegas" else "us-nv-las-vegas-strip"

        self._initial_frame_idx = self._scene_data.num_history_frames - 1

        self._initial_lidar_token = self._scene.frames[self._initial_frame_idx].token
        self._log_name = self._scene_data.log_name
        self._route_roadblock_ids = self._scene.frames[self._initial_frame_idx].roadblock_ids

        self._time_points = [TimePoint(int(frame.timestamp)) for frame in self._scene.frames]
        self._future_sampling = TrajectorySampling(num_poses=len(self._time_points) + 1, interval_length=self._database_interval)
        self._ego_vehicle_parameters = ego_vehicle_parameters

    def __reduce__(self) -> Tuple[Type[NavSimScenario], Tuple[Any, ...]]:
        """
        Hints on how to reconstruct the object when pickling.
        :return: Object type and constructor arguments to be used.
        """
        return (
            self.__class__,
            (
                self._scene,
                self._map_root,
                self._map_version,
                self._ego_vehicle_parameters,
            ),
        )

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        """Inherited, see superclass."""
        return self._ego_vehicle_parameters

    @property
    def token(self) -> str:
        """Inherited, see superclass."""
        return self._initial_lidar_token

    @property
    def log_name(self) -> str:
        """Inherited, see superclass."""
        # e.g. "2021.07.16.20.45.29_veh-35_01095_01486.db"
        return self._log_name

    @property
    def scenario_name(self) -> str:
        """Inherited, see superclass."""
        return self.token

    @property
    def scenario_type(self) -> str:
        """Inherited, see superclass."""
        return DUMMY_SCENARIO_TYPE  # TODO: avoid dummy

    @property
    def map_api(self) -> AbstractMap:
        """Inherited, see superclass."""
        # For V2X-Real and other custom datasets, use the map_api from Scene directly
        # This allows custom map implementations (like DummyMapV2XReal) to work
        if self._scene.map_api is not None:
            return self._scene.map_api
        
        # Fallback to standard NuPlan map loading for standard datasets
        assert self._map_name in MAP_LOCATIONS, f"Map location {self._map_name} not available!"
        map_api = get_maps_api(self._map_root, self._map_version, self._map_name)
        return map_api

    @property
    def map_root(self) -> str:
        """Get the map root folder."""
        return self._map_root

    @property
    def map_version(self) -> str:
        """Get the map version."""
        return self._map_version

    @property
    def database_interval(self) -> float:
        """Inherited, see superclass."""
        return self._database_interval

    def get_number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        return len(self._scene.frames)

    def get_lidar_to_ego_transform(self) -> Transform:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_mission_goal(self) -> Optional[StateSE2]:
        """Inherited, see superclass."""
        return DUMMY_GOAL_STATE

    def get_route_roadblock_ids(self) -> List[str]:
        """Inherited, see superclass."""
        return cast(List[str], self._route_roadblock_ids)

    def get_expert_goal_state(self) -> StateSE2:
        """Inherited, see superclass."""
        return DUMMY_GOAL_STATE

    def get_time_point(self, iteration: int) -> TimePoint:
        """Inherited, see superclass."""

        frame_idx = self._initial_frame_idx + iteration
        assert frame_idx >= 0
        if frame_idx < self.get_number_of_iterations():
            time_point = self._time_points[frame_idx]
            return self._time_points[frame_idx]
        else:
            last_frame_idx = self.get_number_of_iterations() - 1
            last_frame_time_point = self._time_points[last_frame_idx]
            time_point = last_frame_time_point + TimeDuration.from_s((frame_idx - last_frame_idx) * 0.5)
            return time_point

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """Inherited, see superclass."""

        frame_idx = self._initial_frame_idx + iteration
        if frame_idx >= self.get_number_of_iterations():
            warnings.warn(
                f"Iteration {frame_idx} out of bound of {self.get_number_of_iterations()} iterations! Using latest ego state instead"
            )
            frame_idx = self.get_number_of_iterations() - 1
        assert (
            0 <= frame_idx < self.get_number_of_iterations()
        ), f"Iteration {frame_idx} out of bound of {self.get_number_of_iterations()} iterations!"

        return ego_status_to_ego_state(
            ego_status=self._scene.frames[frame_idx].ego_status,
            vehicle_parameters=self._ego_vehicle_parameters,
            time_point=self.get_time_point(iteration),
        )
    
    ### get driving command  
    def get_driving_command_at_iteration(self, iteration: int):
        frame_idx = self._initial_frame_idx + iteration
        if frame_idx >= self.get_number_of_iterations():
            warnings.warn(
                f"Iteration {frame_idx} out of bound of {self.get_number_of_iterations()} iterations! Using latest ego state instead"
            )
            frame_idx = self.get_number_of_iterations() - 1
        assert (
            0 <= frame_idx < self.get_number_of_iterations()
        ), f"Iteration {frame_idx} out of bound of {self.get_number_of_iterations()} iterations!"

        return self._scene.frames[frame_idx].ego_status.driving_command


    def get_tracked_objects_at_iteration(
        self,
        iteration: int,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""

        frame_idx = iteration + self._initial_frame_idx
        assert frame_idx >= 0

        if future_trajectory_sampling:
            warnings.warn("NavSimScenario: TrajectorySampling in get_tracked_objects_at_iteration() not supported.")

        if frame_idx < self.get_number_of_iterations():
            ego_state = self.get_ego_state_at_iteration(iteration)
            return annotations_to_detection_tracks(self._scene.frames[frame_idx].annotations, ego_state)
        elif self._scene.extended_detections_tracks is not None and frame_idx - self.get_number_of_iterations() < len(
            self._scene.extended_detections_tracks
        ):
            return self._scene.extended_detections_tracks[frame_idx - self.get_number_of_iterations()]
        else:
            raise AssertionError(f"Iteration is out of scenario: {iteration}!")

    def get_tracked_objects_within_time_window_at_iteration(
        self,
        iteration: int,
        past_time_horizon: float,
        future_time_horizon: float,
        filter_track_tokens: Optional[Set[str]] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        """Inherited, see superclass."""
        assert 0 <= iteration < self.get_number_of_iterations(), f"Iteration is out of scenario: {iteration}!"
        raise NotImplementedError

    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]] = None) -> Sensors:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_future_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        indices = sample_future_indices(self._future_sampling, iteration, time_horizon, num_samples)
        for idx in indices:
            yield self.get_time_point(idx)

    def get_past_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        """Inherited, see superclass."""
        # FIXME:
        yield self.get_time_point(0)

    def _get_past_indices(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> List[int]:
        """
        Build past indices using scene-configured history length and database interval.
        The returned indices include the current iteration as the last element.
        """
        assert 0 <= iteration < self.get_number_of_iterations(), f"Iteration is out of scenario: {iteration}!"

        if self._database_interval <= 0.0:
            return [iteration]

        # Available past is constrained by scene history setup and current iteration.
        max_past_steps = self._initial_frame_idx + iteration

        # Convert horizon to steps on this scenario's native frequency.
        horizon_past_steps = max(0, int(time_horizon / self._database_interval))

        # Keep legacy behavior: num_samples refers to number of past steps (excluding current).
        requested_past_steps = num_samples if num_samples is not None else horizon_past_steps
        requested_past_steps = max(0, requested_past_steps)

        past_steps = min(max_past_steps, requested_past_steps)
        return list(range(iteration - past_steps, iteration + 1))

    def get_ego_past_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        indices = self._get_past_indices(iteration, time_horizon, num_samples)
        for idx in indices:
            yield self.get_ego_state_at_iteration(idx)

    def get_ego_future_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        """Inherited, see superclass."""
        indices = sample_future_indices(self._future_sampling, iteration, time_horizon, num_samples)
        for idx in indices:
            yield self.get_ego_state_at_iteration(idx)

    def get_past_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""
        indices = self._get_past_indices(iteration, time_horizon, num_samples)
        for idx in indices:
            yield self.get_tracked_objects_at_iteration(idx)

    def get_future_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        """Inherited, see superclass."""

        indices = sample_future_indices(self._future_sampling, iteration, time_horizon, num_samples)
        for idx in indices:
            yield self.get_tracked_objects_at_iteration(idx)

    def get_past_sensors(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        channels: Optional[List[SensorChannel]] = None,
    ) -> Generator[Sensors, None, None]:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        """Inherited, see superclass."""
        frame_idx = iteration + self._initial_frame_idx
        assert frame_idx >= 0
        if frame_idx < self.get_number_of_iterations():
            for lane_connector_id, is_red in self._scene.frames[frame_idx].traffic_lights:
                status = TrafficLightStatusType.RED if is_red else TrafficLightStatusType.GREEN
                yield TrafficLightStatusData(status, lane_connector_id, self.get_time_point(iteration))
        elif self._scene.extended_traffic_light_data is not None and frame_idx - self.get_number_of_iterations() < len(
            self._scene.extended_traffic_light_data
        ):
            traffic_light_data_at_iteration = self._scene.extended_traffic_light_data[
                frame_idx - self.get_number_of_iterations()
            ]
            yield from traffic_light_data_at_iteration.traffic_lights
        else:
            yield from []

    def get_past_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets past traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the past.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the past.
        """
        raise NotImplementedError

    def get_future_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        """
        Gets future traffic light status.

        :param iteration: iteration within scenario 0 <= scenario_iteration < get_number_of_iterations.
        :param time_horizon [s]: the desired horizon to the future.
        :param num_samples: number of entries in the future, if None it will be deduced from the DB.
        :return: Generator object for traffic light history to the future.
        """
        raise NotImplementedError

    def get_scenario_tokens(self) -> List[str]:
        """Return the list of lidarpc tokens from the DB that are contained in the scenario."""
        raise NotImplementedError
