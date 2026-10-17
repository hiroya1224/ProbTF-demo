"""Persistent second-order tracking driven by immutable asynchronous gradients.

The ordinary path integrates du/dt=v, dv/dt=-gamma*v-eta*g exactly for a
constant input g on each physical interval. There is no endpoint, servo,
velocity/acceleration clipping, objective evaluation, or probabilistic sampling
here. Explicit stops are exceptional fail-safe events, not the tracking law.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np


class FrozenMapping(Mapping):
    """Small recursively immutable and spawn-pickleable metadata mapping."""

    def __init__(self, values=()):
        items = values.items() if isinstance(values, Mapping) else values
        object.__setattr__(self, '_items', tuple((key, _freeze(value)) for key, value in items))

    def __getitem__(self, key):
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self):
        return len(self._items)

    def __setattr__(self, key, value):
        raise AttributeError('Gradient metadata is immutable.')

    def __reduce__(self):
        return (type(self), (self._items,))


def _freeze(value):
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, np.ndarray):
        return _freeze(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class GradientPacket:
    """A result keeps its evaluation point, time and candidate/K provenance.

    Failed evaluations may carry ``gradient=None`` and a non-ok status. They
    remain failure packets and are never turned into an accepted zero input.
    """

    gradient: object
    source_command: object
    source_stamp: float
    body_state_ids: object
    goal_state_ids: object
    parameter_density_id: object
    selected_K: object
    objective_at_source: object
    computation_seconds: float
    finite_difference_diagnostics: object
    status: str = 'ok'
    snapshot_update_count: int = 0
    command_interval_id: object = None
    observation_source_stamps: object = ()
    reason: object = None
    packet_id: object = None

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    def as_dict(self):
        return {name: _thaw(getattr(self, name)) for name in self.__dataclass_fields__}


def stable_damped_coefficients(gamma, dt):
    """Return exp(-gamma*dt), A, B without small-gamma cancellation."""
    gamma, dt = float(gamma), float(dt)
    if not math.isfinite(gamma) or not math.isfinite(dt) or gamma < 0. or dt < 0.:
        raise ValueError('Running-gradient gamma and dt must be finite and nonnegative.')
    if dt == 0.:
        return 1., 0., 0.
    if gamma == 0.:
        decay, a_coeff, b_coeff = 1., dt, .5*dt*dt
    else:
        x = gamma*dt
        decay = math.exp(-x)
        if x < 1e-3:
            # A/h and B/h^2, through x^6. The omitted relative term is <1e-25.
            phi1 = 1.+x*(-.5+x*(1./6+x*(-1./24+x*(1./120+x*(-1./720+x/5040)))))
            phi2 = .5+x*(-1./6+x*(1./24+x*(-1./120+x*(1./720+x*(-1./5040+x/40320)))))
            a_coeff, b_coeff = dt*phi1, (dt*dt)*phi2
        else:
            a_coeff = -math.expm1(-x)/gamma
            b_coeff = (dt-a_coeff)/gamma
    if not all(math.isfinite(item) for item in (decay, a_coeff, b_coeff)):
        raise FloatingPointError('Running-gradient integration coefficients overflowed.')
    return decay, a_coeff, b_coeff


def _vector(value, name, shape=None):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Running-gradient %s must be a finite joint vector.' % name) from error
    if (result.ndim != 1 or result.size == 0 or (shape is not None and result.shape != shape)
            or not np.all(np.isfinite(result))):
        raise ValueError('Running-gradient %s must be a finite matching joint vector.' % name)
    return result.copy()


def step_running_gradient(position, velocity, gradient, dt, eta, gamma):
    """Integrate a held gradient; the returned arrays never alias the inputs."""
    position = _vector(position, 'position')
    velocity = _vector(velocity, 'velocity', position.shape)
    gradient = _vector(gradient, 'gradient', position.shape)
    eta = float(eta)
    if not math.isfinite(eta) or eta <= 0.:
        raise ValueError('Running-gradient eta must be finite and positive.')
    decay, a_coeff, b_coeff = stable_damped_coefficients(gamma, dt)
    with np.errstate(over='ignore', invalid='ignore'):
        next_velocity = decay*velocity-eta*a_coeff*gradient
        next_position = position+a_coeff*velocity-eta*b_coeff*gradient
    if not np.all(np.isfinite(next_position)) or not np.all(np.isfinite(next_velocity)):
        raise FloatingPointError('Running-gradient state update produced NaN or infinity.')
    return next_position, next_velocity


@dataclass(frozen=True)
class RunningGradientStep:
    command: tuple
    velocity: tuple
    record: dict


class RunningGradientTracker:
    """Own persistent u/v and sample-and-hold the last accepted finite packet.

    Stale thresholds are disabled by default. When supplied, they reject newly
    arrived stale packets and explicitly stop if the held input becomes stale.
    A fail-safe preserves the last safe position, records the prior velocity and
    the rejected proposal, then announces a stop with velocity zero. It cannot
    silently resume or turn a failed gradient evaluation into a zero gradient.
    """

    def __init__(self, initial_command, *, eta, gamma, joint_lower, joint_upper,
                 initial_stamp=0., initial_velocity=None, max_source_age=None,
                 max_command_distance=None, max_snapshot_update_lag=None):
        self._position = _vector(initial_command, 'initial command')
        self._velocity = (np.zeros_like(self._position) if initial_velocity is None else
                          _vector(initial_velocity, 'initial velocity', self._position.shape))
        self.lower = _vector(joint_lower, 'joint lower', self._position.shape)
        self.upper = _vector(joint_upper, 'joint upper', self._position.shape)
        self.eta, self.gamma, self.stamp = float(eta), float(gamma), float(initial_stamp)
        if (not math.isfinite(self.eta) or self.eta <= 0. or not math.isfinite(self.gamma)
                or self.gamma < 0. or not math.isfinite(self.stamp)
                or np.any(self.lower >= self.upper)
                or np.any(self._position < self.lower) or np.any(self._position > self.upper)):
            raise ValueError('Running-gradient initialization needs finite gains, time and feasible joints.')
        self.max_source_age = self._threshold(max_source_age, 'source age')
        self.max_command_distance = self._threshold(max_command_distance, 'command distance')
        self.max_snapshot_update_lag = self._threshold(max_snapshot_update_lag, 'snapshot update lag')
        self.last_packet = None
        self.accepted_gradient_count = self.rejected_gradient_count = 0
        self.halted = False
        self.stop_notification = None
        self.last_event = None

    @staticmethod
    def _threshold(value, name):
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value) or value < 0.:
            raise ValueError('Running-gradient %s threshold must be finite and nonnegative or None.' % name)
        return value

    @property
    def position(self):
        return self._position.copy()

    @property
    def velocity(self):
        return self._velocity.copy()

    @property
    def gradient(self):
        return None if self.last_packet is None else np.asarray(self.last_packet.gradient, float).copy()

    def _metrics(self, packet, stamp, snapshot_update_count):
        source = np.asarray(packet.source_command, float)
        delta = self._position-source
        return dict(gradient_source_age_s=float(stamp-packet.source_stamp),
                    gradient_source_command_distance_l2=float(np.linalg.norm(delta)),
                    gradient_source_command_distance_inf=float(np.max(np.abs(delta))),
                    gradient_snapshot_update_count=int(packet.snapshot_update_count),
                    snapshot_update_count=(None if snapshot_update_count is None else int(snapshot_update_count)),
                    gradient_snapshot_update_lag=(None if snapshot_update_count is None else
                                                 int(snapshot_update_count)-int(packet.snapshot_update_count)))

    def _stale_reason(self, metrics):
        if self.max_source_age is not None and metrics['gradient_source_age_s'] > self.max_source_age:
            return 'source_age_exceeded'
        if (self.max_command_distance is not None
                and metrics['gradient_source_command_distance_l2'] > self.max_command_distance):
            return 'source_command_distance_exceeded'
        if (self.max_snapshot_update_lag is not None
                and metrics['gradient_snapshot_update_lag'] is not None
                and metrics['gradient_snapshot_update_lag'] > self.max_snapshot_update_lag):
            return 'snapshot_update_lag_exceeded'
        return None

    def accept(self, packet, adoption_stamp, snapshot_update_count=None):
        """Replace only g/provenance after validation; never reset u or v."""
        if not isinstance(packet, GradientPacket):
            raise TypeError('An immutable GradientPacket is required.')
        adoption_stamp = float(adoption_stamp)
        record = dict(event='gradient_result', accepted=False, adoption_stamp=adoption_stamp,
                      packet=packet.as_dict(), position=self._position.tolist(),
                      velocity=self._velocity.tolist())
        reason, metrics = None, {}
        try:
            _vector(packet.source_command, 'packet source command', self._position.shape)
            if packet.status != 'ok':
                reason = 'evaluation_'+str(packet.status)
            else:
                _vector(packet.gradient, 'packet gradient', self._position.shape)
                if (not math.isfinite(adoption_stamp) or not math.isfinite(float(packet.source_stamp))
                        or not math.isfinite(float(packet.computation_seconds))
                        or packet.computation_seconds < 0.
                        or not math.isfinite(float(packet.objective_at_source))):
                    raise ValueError('Non-finite gradient packet timing or objective.')
                if packet.selected_K is not None:
                    selected = _vector(packet.selected_K, 'selected K', self._position.shape)
                    if np.any(selected <= 0.):
                        raise ValueError('Selected K must be positive.')
                if int(packet.snapshot_update_count) != packet.snapshot_update_count or packet.snapshot_update_count < 0:
                    raise ValueError('Snapshot update count must be a nonnegative integer.')
                metrics = self._metrics(packet, adoption_stamp, snapshot_update_count)
                if metrics['gradient_source_age_s'] < 0.:
                    reason = 'source_time_is_in_future'
                elif (metrics['gradient_snapshot_update_lag'] is not None
                      and metrics['gradient_snapshot_update_lag'] < 0):
                    reason = 'source_snapshot_is_in_future'
                else:
                    reason = self._stale_reason(metrics)
        except (TypeError, ValueError, OverflowError) as error:
            reason = 'invalid_packet'
            record['validation_error'] = str(error)
        if self.halted:
            reason = 'tracker_halted'
        record.update(metrics)
        record['reason'] = reason
        if reason is None:
            self.last_packet = packet
            self.accepted_gradient_count += 1
            record['accepted'] = True
        else:
            self.rejected_gradient_count += 1
        record.update(accepted_gradient_count=self.accepted_gradient_count,
                      rejected_gradient_count=self.rejected_gradient_count)
        self.last_event = record
        return record

    def _result(self, record):
        record.update(command=self._position.tolist(), velocity=self._velocity.tolist(),
                      halted=self.halted, accepted_gradient_count=self.accepted_gradient_count,
                      rejected_gradient_count=self.rejected_gradient_count,
                      gradient_packet=None if self.last_packet is None else self.last_packet.as_dict())
        return RunningGradientStep(tuple(self._position), tuple(self._velocity), record)

    def _stop(self, reason, record, *, proposal=None):
        record.update(status='fail_safe_stop', reason=reason, stop_requested=True,
                      last_safe_command=self._position.tolist(), last_safe_velocity=self._velocity.tolist(),
                      proposed_command=None if proposal is None else np.asarray(proposal).tolist())
        self._velocity = np.zeros_like(self._velocity)
        self.halted = True
        self.stop_notification = dict(record)
        self.last_event = record
        return self._result(record)

    def _interval_extrema(self, gradient, dt, next_velocity):
        """Position at every interior velocity zero, including return paths."""
        points = []
        for index, (initial, final, force) in enumerate(zip(self._velocity, next_velocity, self.eta*gradient)):
            if initial == 0. or final == 0. or math.copysign(1., initial) == math.copysign(1., final):
                continue
            if force == 0.:
                continue
            if self.gamma == 0.:
                turning_time = initial/force
            else:
                ratio = self.gamma*initial/force
                turning_time = math.log1p(ratio)/self.gamma
            if 0. < turning_time < dt:
                point, _ = step_running_gradient(self._position, self._velocity, gradient,
                                                turning_time, self.eta, self.gamma)
                points.append((index, turning_time, point))
        return points

    def advance(self, dt, stamp=None, snapshot_update_count=None):
        """Advance one actual interval or return an explicit hold/stop record."""
        dt = float(dt)
        next_stamp = self.stamp+dt if stamp is None else float(stamp)
        record = dict(event='running_gradient_tick', dt=dt, interval_start_stamp=self.stamp,
                      interval_end_stamp=next_stamp, stop_requested=False)
        if (not math.isfinite(dt) or dt < 0. or not math.isfinite(next_stamp)
                or next_stamp < self.stamp):
            return self._stop('invalid_physical_interval', record)
        if self.halted:
            record.update(status='halted', reason=self.stop_notification['reason'], stop_requested=True)
            return self._result(record)
        if self.last_packet is None:
            self.stamp = next_stamp
            record.update(status='waiting_for_initial_gradient', gradient=None)
            return self._result(record)
        metrics = self._metrics(self.last_packet, next_stamp, snapshot_update_count)
        record.update(metrics)
        record['gradient_distance_evaluated_at_command'] = self._position.tolist()
        reason = self._stale_reason(metrics)
        if reason is not None:
            return self._stop('held_gradient_'+reason, record)
        try:
            command, velocity = step_running_gradient(self._position, self._velocity,
                self.last_packet.gradient, dt, self.eta, self.gamma)
            extrema = self._interval_extrema(np.asarray(self.last_packet.gradient), dt, velocity)
        except (ValueError, FloatingPointError, OverflowError) as error:
            record['integration_error'] = str(error)
            return self._stop('nonfinite_or_invalid_integration', record)
        if np.any(command < self.lower) or np.any(command > self.upper):
            return self._stop('joint_position_out_of_range', record, proposal=command)
        for joint, turning_time, point in extrema:
            if point[joint] < self.lower[joint] or point[joint] > self.upper[joint]:
                record.update(violating_joint=joint, violating_interval_offset_s=turning_time)
                return self._stop('joint_position_out_of_range_inside_interval', record, proposal=point)
        self._position, self._velocity, self.stamp = command, velocity, next_stamp
        record.update(self._metrics(self.last_packet, next_stamp, snapshot_update_count))
        record['gradient_distance_evaluated_at_command'] = command.tolist()
        record.update(status='tracking', gradient=list(self.last_packet.gradient),
                      acceleration=(-self.gamma*velocity-self.eta*np.asarray(self.last_packet.gradient)).tolist())
        return self._result(record)
