"""Adapter from the legacy Python PID surrogate to the v2 contracts.

The adapter deliberately reuses :class:`VectorPidSurrogate`; it does not copy
or claim to implement the upstream Gimbalrotor C++ equations.  Its identity is
always non-exact, so it can be used for smoke tests and proposals but can
never satisfy an exact replay gate.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.controller_replay import (
    AXIS_NAMES,
    ControllerFeedback,
    ControllerParameters,
    ControllerReference,
    PidLimits,
    VectorPidSurrogate,
)

from .contracts import (
    ControllerBackendIdentity,
    ControllerCoreInput,
    ControllerCoreOutput,
    ControllerCoreState,
    FIDELITY_PC_EXACT,
    PC_EXACT_REQUIRED_CAPABILITIES,
    PidCoreState,
)
from .snapshot import ControllerSnapshot


def _six_axis(
    values: Mapping[str, Any],
    names: Sequence[str],
    default: float,
) -> np.ndarray:
    for name in names:
        if name in values:
            array = np.asarray(values[name], dtype=float)
            if array.ndim == 0:
                return np.full(6, float(array))
            if array.shape == (6,):
                return np.array(array, copy=True)
            raise ValueError("{} must be scalar or a six-vector".format(name))
    per_axis = []
    for axis in AXIS_NAMES:
        axis_values = values.get(axis)
        selected = None
        if isinstance(axis_values, Mapping):
            for name in names:
                if name in axis_values:
                    selected = axis_values[name]
                    break
        per_axis.append(default if selected is None else float(selected))
    return np.asarray(per_axis, dtype=float)


def _parameters(snapshot: ControllerSnapshot) -> ControllerParameters:
    gains = snapshot.gains
    limits = snapshot.limits
    p_gain = _six_axis(gains, ("p_gain", "p"), 0.0)
    i_gain = _six_axis(gains, ("i_gain", "i"), 0.0)
    d_gain = _six_axis(gains, ("d_gain", "d"), 0.0)
    pid_limits = PidLimits(
        output=_six_axis(limits, ("output", "limit_sum"), 1.0e6),
        p_term=_six_axis(limits, ("p_term", "limit_p"), 1.0e6),
        i_term=_six_axis(limits, ("i_term", "limit_i"), 1.0e6),
        d_term=_six_axis(limits, ("d_term", "limit_d"), 1.0e6),
        p_error=_six_axis(limits, ("p_error", "limit_err_p"), 1.0e6),
        i_state=_six_axis(limits, ("i_state", "limit_err_i"), 1.0e6),
        d_error=_six_axis(limits, ("d_error", "limit_err_d"), 1.0e6),
    )
    inertia = np.asarray(snapshot.nominal_inertia, dtype=float)
    return ControllerParameters(
        p_gain=p_gain,
        i_gain=i_gain,
        d_gain=d_gain,
        limits=pid_limits,
        controller_mass=snapshot.nominal_mass,
        controller_inertia_diagonal=np.diag(inertia),
    )


class PythonSurrogateControllerBackend:
    """Contract adapter for low-fidelity tests and proposal generation."""

    def __init__(
        self,
        snapshot: Optional[ControllerSnapshot] = None,
        initial_state: Optional[ControllerCoreState] = None,
    ) -> None:
        self.identity = ControllerBackendIdentity(
            backend_id="python_vector_pid_surrogate_adapter/v2",
            fidelity=FIDELITY_PC_EXACT,
            is_exact=False,
            capabilities=PC_EXACT_REQUIRED_CAPABILITIES,
            implementation_language="Python",
            source_commit="surrogate",
            artifact_sha256="0" * 64,
        )
        self.snapshot: Optional[ControllerSnapshot] = None
        self._backend: Optional[VectorPidSurrogate] = None
        self._state = ControllerCoreState()
        if snapshot is not None:
            self.reset(
                snapshot,
                ControllerCoreState()
                if initial_state is None
                else initial_state,
            )

    @property
    def state(self) -> ControllerCoreState:
        return self._state

    def reset(
        self,
        snapshot: ControllerSnapshot,
        initial_state: ControllerCoreState,
    ) -> None:
        if not isinstance(snapshot, ControllerSnapshot):
            raise TypeError("snapshot must be ControllerSnapshot")
        if not isinstance(initial_state, ControllerCoreState):
            raise TypeError("initial_state must be ControllerCoreState")
        integral = np.asarray(
            [item.error_i for item in initial_state.pid], dtype=float
        )
        self.snapshot = snapshot
        self._backend = VectorPidSurrogate(
            _parameters(snapshot),
            initial_integral_state=integral,
        )
        self._state = initial_state

    def step(self, item: ControllerCoreInput) -> ControllerCoreOutput:
        if not isinstance(item, ControllerCoreInput):
            raise TypeError("item must be ControllerCoreInput")
        if self._backend is None or self.snapshot is None:
            raise RuntimeError("surrogate backend must be reset with a snapshot")
        if item.reset:
            self._backend.reset()

        # The legacy surrogate has a six-vector translational/Euler interface.
        # Orientation allocation remains intentionally absent: adding it here
        # would duplicate the upstream C++ controller being bounded.
        reference = ControllerReference(
            position=np.concatenate((item.target_position, np.zeros(3))),
            velocity=np.concatenate(
                (item.target_velocity, item.target_angular_velocity)
            ),
            acceleration=np.concatenate(
                (
                    item.target_acceleration,
                    item.target_angular_acceleration,
                )
            ),
        )
        feedback = ControllerFeedback(
            position=np.concatenate((item.position, np.zeros(3))),
            velocity=np.concatenate(
                (item.velocity, item.angular_velocity)
            ),
        )
        result = self._backend.step(
            reference,
            feedback,
            item.dt,
            integration_enabled=np.asarray(item.integration_enabled),
            control_mode=np.asarray(item.control_mode),
            mode=item.flight_state,
        )
        saturated_axes = np.flatnonzero(
            result.term_saturated | result.output_saturated
        )
        events = tuple(100 + int(axis) for axis in saturated_axes)
        pid_state = tuple(
            PidCoreState(
                error_p=result.p_error[axis],
                error_i=result.integral_state[axis],
                previous_error_i=self._state.pid[axis].error_i,
                error_d=result.d_error[axis],
                result=result.acceleration_command[axis],
                p_term=result.p_term[axis],
                i_term=result.i_term[axis],
                d_term=result.d_term[axis],
            )
            for axis in range(6)
        )
        self._state = ControllerCoreState(
            pid=pid_state,
            start_roll_pitch_integration=bool(
                self._state.start_roll_pitch_integration
                or item.integration_enabled[3]
                or item.integration_enabled[4]
            ),
            previous_stamp=item.stamp,
            previous_flight_state=item.flight_state,
            target_gimbal_angles=self._state.target_gimbal_angles,
            previous_control_mode=item.control_mode,
            previous_force_landing=item.force_landing,
        )
        wrench = tuple(float(value) for value in result.generalized_wrench_command)
        surrogate_base_thrust = tuple([wrench[2] / 4.0] * 4)
        return ControllerCoreOutput(
            pid_result=tuple(float(value) for value in result.acceleration_command),
            pid_p_term=tuple(float(value) for value in result.p_term),
            pid_i_term=tuple(float(value) for value in result.i_term),
            pid_d_term=tuple(float(value) for value in result.d_term),
            target_vectoring_force=wrench,
            # This is only a stable surrogate command carrier, not the
            # Gimbalrotor allocation or a claim of Newton-valued thrust.
            base_thrust=surrogate_base_thrust,
            gimbal_angle=(0.0, 0.0, 0.0, 0.0),
            torque_allocation_matrix_inverse=(),
            target_roll=0.0,
            target_pitch=0.0,
            candidate_yaw_term=float(result.acceleration_command[5]),
            events=events,
            stamp=item.stamp,
            saturated=bool(saturated_axes.size),
            four_axis_angles=(
                0.0,
                0.0,
                float(result.acceleration_command[5]),
            ),
            generalized_wrench=wrench,
            effective_target_acceleration=item.target_acceleration,
        )


PythonSurrogateAdapter = PythonSurrogateControllerBackend


__all__ = [
    "PythonSurrogateAdapter",
    "PythonSurrogateControllerBackend",
]
