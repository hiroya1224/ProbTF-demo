"""Bounded marker reacquisition around the command at which the arm stopped."""

from dataclasses import dataclass

import numpy as np


def marker_observed(result):
    return any(estimate.posterior is not None and estimate.detector_result is not None
               for estimate in result.local_estimates)


def _marker_observation_stamp(result):
    memory = getattr(result, 'records', {}).get('fixed_goal_memory', {})
    if memory:
        value = memory.get('observation_stamp') if memory.get('available', False) else None
        return (None if value is None else float(value)), 'fixed_goal_memory'
    stamps = [float(estimate.posterior.query.stamp) for estimate in result.local_estimates
        if estimate.posterior is not None and estimate.detector_result is not None
        and getattr(estimate.posterior, 'query', None) is not None]
    # The lightweight policy-only interface can omit a posterior query. Real
    # async callers supply the active, fused goal's original capture stamp.
    return (max(stamps), 'posterior_query') if stamps else (float(result.prior.stamp), 'input_prior_fallback')


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
    """Move between bounded wrist viewpoints, stopping to observe at each.

    This action policy receives observation validity and sent commands only.
    It does not replace a body/goal distribution or use physical pose truth.
    The common short-command controller enforces all motion constraints.
    """

    def __init__(self, settings, *, marker_enabled, exposure_time=0.):
        self.enabled = bool(settings.get('enabled', True)) and bool(marker_enabled)
        self.joints = tuple(int(index) for index in settings.get('joint_indices', (4, 3)))
        self.initial_amplitude = float(settings.get('amplitude_rad', .06))
        self.amplitude_increment = float(settings.get('amplitude_increment_rad', .03))
        self.max_amplitude = float(settings.get('max_amplitude_rad', .18))
        self.amplitude = self.initial_amplitude
        self.speed = float(settings.get('max_joint_velocity', .048))
        self.settle_time = max(float(settings.get('settle_time_s', .35)), float(exposure_time))
        values = (self.initial_amplitude, self.amplitude_increment,
                  self.max_amplitude, self.speed, self.settle_time)
        if (len(self.joints) != 2 or len(set(self.joints)) != 2 or min(self.joints) < 0
                or self.initial_amplitude > self.max_amplitude
                or not all(np.isfinite(value) and value > 0. for value in values)):
            raise ValueError('Marker search requires two distinct joints and finite positive limits.')
        self.mode = 'reaching'
        self.anchor = None
        self.settle_started = self.search_started = None
        self.settle_reason = None
        self.settle_target = None
        self.stop_source_stamp = None
        self.braking_reason = 'after_marker_loss'
        self.episode = self.waypoint = 0
        self.completed_roundtrips = self.completed_pattern_cycles = 0
        # Inspect the configured first wrist axis, then the second, with a full
        # stop and observation hold at both outward and centre viewpoints.
        self.pattern = ((1., 0.), (0., 0.), (0., 1.), (0., 0.),
                        (-1., 0.), (0., 0.), (0., -1.), (0., 0.))
        self.last_time = None

    def plan(self, result, velocity, time, *, joint_lower, joint_upper,
             marker_detected=None):
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
        detected = ((visible if marker_detected is None else bool(marker_detected))
                    if self.enabled else None)
        observation_stamp, observation_stamp_source = _marker_observation_stamp(result)
        if observation_stamp is not None and not np.isfinite(observation_stamp):
            raise ValueError('A marker observation source stamp must be finite.')
        target = position_lower = position_upper = max_speed = None
        event = 'marker_observation_not_enabled'
        stopped = bool(np.max(np.abs(velocity)) <= 1e-6)

        if self.enabled:
            event = 'marker_visible' if visible else 'marker_missing'
            if self.mode == 'reaching' and not (visible and detected):
                self.mode, self.anchor = 'braking', None
                self.episode += 1
                self.waypoint = 0
                self.amplitude = self.initial_amplitude
                self.completed_roundtrips = self.completed_pattern_cycles = 0
                self.settle_started = self.search_started = None
                self.settle_reason = None
                self.settle_target = None
                self.stop_source_stamp = None
                self.braking_reason = 'after_marker_loss'
                event = 'braking_after_marker_loss'
            # Pixels can stop a search immediately, but cannot authorize
            # reaching: wait at the stopped viewpoint for an actual fused goal.
            if self.mode == 'searching' and detected:
                self.mode = 'braking'
                self.braking_reason = 'after_search_reacquisition'
                event = 'braking_after_marker_reacquisition'
            if self.mode == 'braking' and stopped:
                if self.anchor is None:
                    self.anchor = command.copy()
                self.mode, self.settle_started = 'settling', time
                self.settle_reason = self.braking_reason
                self.settle_target = command.copy()
                self.stop_source_stamp = float(result.prior.stamp)
                event = 'stopped_for_exposure_settling'
            if self.mode == 'settling':
                fresh_since_stop = bool(detected and visible and observation_stamp is not None
                    and observation_stamp >= self.stop_source_stamp-1e-9)
                if time-self.settle_started >= self.settle_time-1e-12:
                    reason = self.settle_reason
                    if fresh_since_stop:
                        self.mode = 'reaching'
                        event = 'marker_reacquired_after_stopping'
                    elif detected:
                        # A valid preview must not cause repeated search
                        # motion merely because numerical inference is late.
                        event = 'marker_detected_waiting_for_fused_goal'
                    else:
                        self.mode = 'searching'
                        if reason in ('outward_observation', 'between_sweeps'):
                            self.waypoint = (self.waypoint+1) % len(self.pattern)
                        if self.search_started is None:
                            self.search_started = time
                            event = 'bounded_wrist_search_started'
                        elif reason == 'between_sweeps':
                            previous_amplitude = self.amplitude
                            self.amplitude = min(self.max_amplitude,
                                self.initial_amplitude
                                +self.amplitude_increment*self.completed_roundtrips)
                            event = ('search_amplitude_expanded'
                                if self.amplitude > previous_amplitude else 'bounded_wrist_search_repeated')
                        elif reason == 'outward_observation':
                            event = 'returning_to_search_center'
                        else:
                            event = 'bounded_wrist_search_resumed'
                    if self.mode != 'settling':
                        self.settle_reason = None
                        self.settle_target = None

            if self.mode in ('searching', 'settling') or (
                    self.mode == 'braking' and self.anchor is not None):
                position_lower, position_upper = self.anchor.copy(), self.anchor.copy()
                position_lower[list(self.joints)] -= self.amplitude
                position_upper[list(self.joints)] += self.amplitude
                position_lower = np.maximum(position_lower, lower)
                position_upper = np.minimum(position_upper, upper)
                max_speed = self.speed
            if self.mode == 'searching':
                target = self._target(position_lower, position_upper)
                if np.max(np.abs(command-target)) <= 1e-4 and stopped:
                    if self.waypoint % 2:
                        self.completed_roundtrips += 1
                        if self.waypoint == len(self.pattern)-1:
                            self.completed_pattern_cycles += 1
                        event = 'stopped_between_search_sweeps'
                        self.settle_reason = 'between_sweeps'
                    else:
                        self.settle_reason = 'outward_observation'
                        event = 'stopped_at_outward_viewpoint'
                    self.mode, self.settle_started = 'settling', time
                    self.settle_target = target.copy()
                    self.stop_source_stamp = float(result.prior.stamp)
            elif self.mode == 'settling':
                target = self.settle_target

        record = dict(mode=self.mode, event=event, enabled=self.enabled,
            marker_visible=visible, episode=self.episode, time=time,
            marker_detected=detected, marker_observed=visible,
            marker_detection_source=('current_observation' if marker_detected is None
                                     else 'raw_image_preview'),
            waiting_for_fused_marker_goal=bool(self.mode == 'settling' and detected
                and not (visible and observation_stamp is not None
                         and observation_stamp >= self.stop_source_stamp-1e-9)),
            source_stamp=float(result.prior.stamp), source_command=command.tolist(),
            source_prior_version=int(result.prior.version),
            stopped_before_search=bool(self.anchor is not None),
            stop_anchor_command=None if self.anchor is None else self.anchor.tolist(),
            target_command=None if target is None else np.asarray(target).tolist(),
            waypoint_index=self.waypoint, waypoint_count=len(self.pattern),
            joint_indices=list(self.joints), amplitude_rad=self.amplitude,
            initial_amplitude_rad=self.initial_amplitude,
            amplitude_increment_rad=self.amplitude_increment,
            max_amplitude_rad=self.max_amplitude,
            completed_roundtrips=self.completed_roundtrips,
            completed_pattern_cycles=self.completed_pattern_cycles,
            settle_reason=self.settle_reason,
            observation_hold_target=None if self.settle_target is None else self.settle_target.tolist(),
            observation_hold_started=self.settle_started,
            observation_hold_source_stamp=self.stop_source_stamp,
            marker_observation_stamp=observation_stamp,
            marker_observation_stamp_source=observation_stamp_source,
            marker_observation_since_stop=bool(visible and observation_stamp is not None
                and self.stop_source_stamp is not None
                and observation_stamp >= self.stop_source_stamp-1e-9),
            search_strategy='bounded step, full stop, observation hold at every outward and return viewpoint',
            waypoint_stop_velocity_tolerance_rad_s=1e-6,
            max_joint_velocity=self.speed, settle_time_s=self.settle_time,
            max_duration_s=None,
            search_elapsed_s=None if self.search_started is None else time-self.search_started,
            physical_truth_received=False, monte_carlo_used=False,
            reaching_criterion_used_for_selection=(self.mode == 'reaching'))
        return SearchPlan(self.mode, _fixed_array(target), max_speed,
                          _fixed_array(position_lower), _fixed_array(position_upper), record)

    def _target(self, lower, upper):
        result = self.anchor.copy()
        result[list(self.joints)] += self.amplitude*np.asarray(self.pattern[self.waypoint])
        return np.clip(result, lower, upper)
