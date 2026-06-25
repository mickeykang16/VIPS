import os

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.simulation.planner.pdm_planner.simulation.batch_kinematic_bicycle import BatchKinematicBicycleModel
from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr import BatchLQRTracker
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from scipy.interpolate import CubicSpline

class PDMSimulator:
    """
    Re-implementation of nuPlan's simulation pipeline. Enables batch-wise simulation.
    """

    def __init__(self, proposal_sampling: TrajectorySampling):
        """
        Constructor of PDMSimulator.
        :param proposal_sampling: Sampling parameters for proposals
        """

        # time parameters
        self.proposal_sampling = proposal_sampling

        # simulation objects
        self._motion_model = BatchKinematicBicycleModel()
        
        # self._tracker = BatchLQRTracker(
        #     q_longitudinal=[20.0],
        #     q_lateral=[100.0, 50.0, 0.0],
        #     r_lateral=[0.1],
        #     stopping_proportional_gain=1.0,
        #     tracking_horizon=8,
        # )
        self._tracker = BatchLQRTracker(
            # q_longitudinal=[20.0],
            # q_lateral=[100.0, 50.0, 20.0],  # steering_angle cost 0->20: suppress overshoot
            # r_lateral=[2.0],                 # control cost 0.1->2.0: suppress excessive steering
            # stopping_proportional_gain=1.0,
            # # tracking_horizon=8,
            # tracking_horizon=10,
        )

    def simulate_proposals(
        self, states: npt.NDArray[np.float64], initial_ego_state: EgoState
    ) -> npt.NDArray[np.float64]:
        """
        Simulate all proposals over batch-dim
        :param initial_ego_state: ego-vehicle state at current iteration
        :param states: proposal states as array
        :return: simulated proposal states as array
        """
        N_SUBSTEPS = 5
        sub_interval = self.proposal_sampling.interval_length / N_SUBSTEPS

        self._motion_model._vehicle = initial_ego_state.car_footprint.vehicle_parameters
        self._tracker._discretization_time = sub_interval

        proposal_states = states[:, : self.proposal_sampling.num_poses + 1]
        num_outer = self.proposal_sampling.num_poses
        batch_size, num_orig, state_dim = proposal_states.shape

        # NOTE ver1 -> linear interpolation # Upsample proposal_states to sub-step resolution by linear interpolation
        num_sub = (num_orig - 1) * N_SUBSTEPS + 1
        sub_proposal_states = np.zeros((batch_size, num_sub, state_dim), dtype=np.float64)
        for i in range(num_orig - 1):
            for s in range(N_SUBSTEPS):
                alpha = s / N_SUBSTEPS
                sub_idx = i * N_SUBSTEPS + s
                sub_proposal_states[:, sub_idx] = (
                    (1 - alpha) * proposal_states[:, i] + alpha * proposal_states[:, i + 1]
                )
                # Heading requires angle-aware interpolation to handle wrap-around
                h0 = proposal_states[:, i, StateIndex.HEADING]
                h1 = proposal_states[:, i + 1, StateIndex.HEADING]
                dh = np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0))
                sub_proposal_states[:, sub_idx, StateIndex.HEADING] = h0 + alpha * dh
        sub_proposal_states[:, -1] = proposal_states[:, -1]

        # NOTE ver2 -> Upsample proposal_states to sub-step resolution by cubic spline interpolation
        # Cubic spline smoothly approximates the arc -> fixes curvature estimation errors caused by the chord problem
        # num_sub = (num_orig - 1) * N_SUBSTEPS + 1
        # sub_proposal_states = np.zeros((batch_size, num_sub, state_dim), dtype=np.float64)

        # t_orig = np.arange(num_orig, dtype=np.float64)
        # t_sub = np.linspace(0, num_orig - 1, num_sub)

        # # x, y: cubic spline (whole batch at once)
        # # proposal_states: (batch, T, state_dim) → transpose to (T, batch) for CubicSpline
        # cs_x = CubicSpline(t_orig, proposal_states[:, :, StateIndex.X].T)   # input: (T, B)
        # cs_y = CubicSpline(t_orig, proposal_states[:, :, StateIndex.Y].T)
        # sub_proposal_states[:, :, StateIndex.X] = cs_x(t_sub).T             # output: (B, num_sub)
        # sub_proposal_states[:, :, StateIndex.Y] = cs_y(t_sub).T

        # # heading: unwrap -> cubic spline -> re-wrap (prevents 2*pi jumps)
        # h_raw = proposal_states[:, :, StateIndex.HEADING]                     # (B, T)
        # h_unwrapped = np.unwrap(h_raw, axis=1)
        # cs_h = CubicSpline(t_orig, h_unwrapped.T)
        # h_interp = cs_h(t_sub).T                                              # (B, num_sub)
        # sub_proposal_states[:, :, StateIndex.HEADING] = np.arctan2(
        #     np.sin(h_interp), np.cos(h_interp)
        # )

        # # remaining channels (velocity, etc.): linear interpolation is sufficient
        # for ch in range(state_dim):
        #     if ch in (StateIndex.X, StateIndex.Y, StateIndex.HEADING):
        #         continue
        #     sub_proposal_states[:, :, ch] = np.interp(
        #         t_sub, t_orig, proposal_states[0, :, ch]  # per-channel scalar -> broadcast
        #     )
        #     # ideally interpolate per batch, but velocity/acceleration are recomputed by the LQR, so this approximation is sufficient
        #     for b in range(batch_size):
        #         sub_proposal_states[b, :, ch] = np.interp(
        #             t_sub, t_orig, proposal_states[b, :, ch]
        #         )

        self._tracker.update(sub_proposal_states)

        # state array representation for simulated vehicle states (original resolution output)
        simulated_states = np.zeros(proposal_states.shape, dtype=np.float64)
        simulated_states[:, 0] = ego_state_to_state_array(initial_ego_state)

        sub_delta = TimeDuration.from_s(sub_interval)
        current_time_point = initial_ego_state.time_point
        current_iteration = SimulationIteration(current_time_point, 0)
        next_iteration = SimulationIteration(current_time_point + sub_delta, 1)

        sub_state = simulated_states[:, 0].copy()

        for time_idx in range(1, num_outer + 1):
            for sub_idx in range(N_SUBSTEPS):
                global_sub_idx = (time_idx - 1) * N_SUBSTEPS + sub_idx
                sampling_time = next_iteration.time_point - current_iteration.time_point

                command_states = self._tracker.track_trajectory(
                    current_iteration,
                    next_iteration,
                    sub_state,
                )
                sub_state = self._motion_model.propagate_state(
                    states=sub_state,
                    command_states=command_states,
                    sampling_time=sampling_time,
                )

                current_iteration = next_iteration
                next_iteration = SimulationIteration(
                    current_iteration.time_point + sub_delta, global_sub_idx + 2
                )

            simulated_states[:, time_idx] = sub_state


                
        return simulated_states
