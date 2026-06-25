# TODO: Move this file for common usage (not specific for PDM)

from __future__ import annotations

import warnings
from typing import Any, List, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import StateSE2
from scipy.interpolate import interp1d
from shapely.creation import linestrings
from shapely.geometry import LineString, Point
from shapely.ops import substring

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    array_to_states_se2,
    states_se2_to_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import SE2Index
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import calculate_progress, normalize_angle


class PDMPath:
    """Class representing a path to interpolate for PDM."""

    def __init__(self, discrete_path: List[StateSE2]):
        """
        Constructor for PDMPath
        :param discrete_path: list of (x,y,θ) values
        """

        if len(discrete_path) == 0:
            raise ValueError("PDMPath requires a non-empty discrete_path (no lane found near ego position)")

        # attribute
        self._discrete_path = discrete_path

        # loaded during initialization
        self._states_se2_array = states_se2_to_array(discrete_path)
        self._states_se2_array[:, SE2Index.HEADING] = np.unwrap(self._states_se2_array[:, SE2Index.HEADING], axis=0)
        self._progress = calculate_progress(discrete_path)
        self._linestring = linestrings(self._states_se2_array[:, : SE2Index.HEADING])
        ### NOTE hm debugging
        d = np.diff(self._progress)
        if np.any(d <= 0):
            print("[WARN] non-increasing progress:", (d <= 0).sum())
        self._interpolator = interp1d(self._progress, self._states_se2_array, axis=0)
        

    def __reduce__(self) -> Tuple[Type[PDMPath], Tuple[Any, ...]]:
        """Helper for pickling."""
        return self.__class__, (self._discrete_path,)

    @property
    def discrete_path(self):
        """Getter for discrete StateSE2 objects of path.

        Returns the original path with one extra "phantom" point appended,
        extrapolated 0.5 m past the last segment. This works around an
        off-by-one in nuPlan's `create_path_from_se2` (used by IDMAgent's
        `_convert_route_to_path`): the function does
            `zip(states, progress_list, repeated_states_mask)`
        where `repeated_states_mask = np.isclose(np.diff(progress_list), 0)`
        is one shorter than `states`, so the LAST state is silently dropped.

        Padding with a duplicate doesn't help because the duplicate's diff = 0
        triggers the `is_repeated` filter and the original last point is the
        one that gets removed. Appending a non-duplicate point ahead of the
        last segment makes `mask[-1]` False, so the original last point is
        retained while the phantom is dropped by the zip itself.

        PDMPath's own `_interpolator` and `linestring` are built from the
        un-padded path, so this is invisible to V2X-Real consumers.
        """
        if len(self._discrete_path) >= 2:
            last = self._discrete_path[-1]
            prev = self._discrete_path[-2]
            dx, dy = last.x - prev.x, last.y - prev.y
            norm = float(np.hypot(dx, dy))
            if norm > 1e-3:
                ext_dist = 0.5  # 50 cm phantom extension
                phantom = StateSE2(
                    last.x + dx / norm * ext_dist,
                    last.y + dy / norm * ext_dist,
                    last.heading,
                )
                return list(self._discrete_path) + [phantom]
        return self._discrete_path

    @property
    def length(self):
        """Getter for length of path."""
        return self._progress[-1]

    @property
    def linestring(self) -> LineString:
        """Getter for shapely's linestring of path."""
        return self._linestring

    def project(self, points: Any) -> Any:
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in line_locate_point", category=RuntimeWarning
        )
        return self._linestring.project(points)

    def interpolate(
        self,
        distances: Union[List[float], npt.NDArray[np.float64]],
        as_array=False,
    ) -> Union[npt.NDArray[np.object_], npt.NDArray[np.float64]]:
        """
        Calculates (x,y,θ) for a given distance along the path.
        :param distances: list of array of distance values
        :param as_array: whether to return in array representation, defaults to False
        :return: array of StateSE2 class or (x,y,θ) values
        """
        clipped_distances = np.clip(distances, 1e-5, self.length)
        interpolated_se2_array = self._interpolator(clipped_distances)
        interpolated_se2_array[..., 2] = normalize_angle(interpolated_se2_array[..., 2])
        interpolated_se2_array[np.isnan(interpolated_se2_array)] = 0.0

        if as_array:
            return interpolated_se2_array

        return array_to_states_se2(interpolated_se2_array)

    def get_nearest_arc_length_from_position(self, point: Any) -> float:
        """Compat with nuPlan's InterpolatedPath. Used by IDM `get_starting_segment`."""
        if isinstance(point, Point):
            shapely_point = point
        else:
            shapely_point = Point(float(point.x), float(point.y))
        return float(self.project(shapely_point))

    def get_nearest_pose_from_position(self, point: Any) -> StateSE2:
        """Compat with nuPlan's InterpolatedPath. Used by IDM `get_starting_segment`."""
        arc_length = self.get_nearest_arc_length_from_position(point)
        return self.interpolate([arc_length])[0]

    def substring(self, start_distance: float, end_distance: float) -> LineString:
        """
        Creates a sub-linestring between start and ending distances.
        :param start_distance: distance along the path to start [m]
        :param end_distance:  distance along the path to end [m]
        :return: LineString
        """

        # try faster method fist
        start_distance = np.clip(start_distance, 0.0, self.length)
        end_distance = np.clip(end_distance, 0.0, self.length)
        in_interval = np.logical_and(start_distance <= self._progress, self._progress <= end_distance)
        coordinates = self._states_se2_array[in_interval, :2]
        if len(coordinates) > 1:
            return LineString(coordinates)

        # fallback to slower method of shapely
        return substring(self.linestring, start_distance, end_distance)
