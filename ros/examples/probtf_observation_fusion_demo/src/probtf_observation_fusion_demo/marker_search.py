"""Bounded marker reacquisition around the command at which the arm stopped."""

from dataclasses import dataclass

import numpy as np


def marker_observed(result):
    return any(estimate.posterior is not None and estimate.detector_result is not None
               for estimate in result.local_estimates)


def _fixed_array(value):
    if value is None:
        return None
    value = np.array(value, dtype=float, copy=True)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class SearchPlan:
    mode: str
    target_command: object
    max_speed: object
    position_lower: object
    position_upper: object
    record: dict


class MarkerSearchPolicy:
    """Stop, clear the exposure history, then scan a small wrist neighborhood.

    This action policy receives observation validity and sent commands only.
    It does not replace a body/goal distribution or use physical pose truth.
    The common short-command controller enforces all motion constraints.
    """

    def __init__(self, settings, *, marker_enabled, exposure_time=0.):
        self.enabled = bool(settings.get('enabled', True)) and bool(marker_enabled)
        self.joints = tuple(int(index) for index in settings.get('joint_indices', (3, 4)))
        self.amplitude = float(settings.get('amplitude_rad', .02))
        self.speed = float(settings.get('max_joint_velocity', .025))
        self.settle_time = max(float(settings.get('settle_time_s', .25)), float(exposure_time))
        self.max_duration = float(settings.get('max_duration_s', 8.))
        values = (self.amplitude, self.speed, self.settle_time, self.max_duration)
        if (len(self.joints) != 2 or len(set(self.joints)) != 2 or min(self.joints) < 0
                or not all(np.isfinite(value) and value > 0. for value in values)):
            raise ValueError('Marker search requires two distinct joints and finite positive limits.')
        self.mode = 'reaching'
        self.anchor = None
        self.settle_started = self.search_started = None
        self.episode = self.waypoint = 0
        # Return to the stopped command between the positive/negative sweeps.
        self.pattern = ((1., 0.), (0., 0.), (-1., 0.), (0., 0.),
                        (0., 1.), (0., 0.), (0., -1.), (0., 0.))
        self.last_time = None

    def plan(self, result, velocity, time, *, joint_lower, joint_upper):
        time = float(time)
        command = np.asarray(result.prior.command, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        lower, upper = np.asarray(joint_lower), np.asarray(joint_upper)
        if (velocity.shape != command.shape or lower.shape != command.shape
                or upper.shape != command.shape or max(self.joints) >= command.size
                or not all(np.all(np.isfinite(value)) for value in (command, velocity, lower, upper))
                or not np.isfinite(time) or (self.last_time is not None and time <= self.last_time)):
            raise ValueError('Search planning needs increasing times and matching finite joint metadata.')
        self.last_time = time
        visible = marker_observed(result) if self.enabled else None
        target = position_lower = position_upper = max_speed = None
        event = 'marker_observation_not_enabled'
        stopped = bool(np.max(np.abs(velocity)) <= 1e-6)

        if self.enabled:
            event = 'marker_visible' if visible else 'marker_missing'
            if self.mode == 'reaching' and not visible:
                self.mode, self.anchor = 'braking', None
                self.episode += 1
                self.waypoint = 0
                self.settle_started = self.search_started = None
                event = 'braking_after_marker_loss'
            if self.mode == 'braking' and stopped:
                self.anchor = command.copy()
                self.mode, self.settle_started = 'settling', time
                event = 'stopped_for_exposure_settling'
            if self.mode == 'settling' and time-self.settle_started >= self.settle_time-1e-12:
                if visible:
                    self.mode = 'reaching'
                    event = 'marker_reacquired_after_stopping'
                else:
                    self.mode, self.search_started = 'searching', time
                    event = 'bounded_wrist_search_started'
            if self.mode in ('searching', 'search_exhausted') and visible:
                self.mode = 'reaching'
                event = 'marker_reacquired'

            if self.mode in ('searching', 'search_exhausted'):
                position_lower, position_upper = self.anchor.copy(), self.anchor.copy()
                position_lower[list(self.joints)] -= self.amplitude
                position_upper[list(self.joints)] += self.amplitude
                position_lower = np.maximum(position_lower, lower)
                position_upper = np.minimum(position_upper, upper)
                max_speed = self.speed
            if self.mode == 'searching':
                target = self._target(position_lower, position_upper)
                if np.max(np.abs(command-target)) <= 1e-4 and np.max(np.abs(velocity)) <= 1e-3:
                    self.waypoint += 1
                    target = (None if self.waypoint == len(self.pattern)
                              else self._target(position_lower, position_upper))
                if time-self.search_started >= self.max_duration or target is None:
                    self.mode, target = 'search_exhausted', None
                    event = 'search_limit_brake_and_hold'
            elif self.mode == 'settling':
                target = self.anchor

        record = dict(mode=self.mode, event=event, enabled=self.enabled,
            marker_visible=visible, episode=self.episode, time=time,
            source_stamp=float(result.prior.stamp), source_command=command.tolist(),
            source_prior_version=int(result.prior.version),
            stopped_before_search=bool(self.anchor is not None),
            stop_anchor_command=None if self.anchor is None else self.anchor.tolist(),
            target_command=None if target is None else np.asarray(target).tolist(),
            waypoint_index=self.waypoint, waypoint_count=len(self.pattern),
            joint_indices=list(self.joints), amplitude_rad=self.amplitude,
            max_joint_velocity=self.speed, settle_time_s=self.settle_time,
            max_duration_s=self.max_duration,
            search_elapsed_s=None if self.search_started is None else time-self.search_started,
            physical_truth_received=False, monte_carlo_used=False,
            reaching_criterion_used_for_selection=(self.mode == 'reaching'))
        return SearchPlan(self.mode, _fixed_array(target), max_speed,
                          _fixed_array(position_lower), _fixed_array(position_upper), record)

    def _target(self, lower, upper):
        result = self.anchor.copy()
        result[list(self.joints)] += self.amplitude*np.asarray(self.pattern[self.waypoint])
        return np.clip(result, lower, upper)
