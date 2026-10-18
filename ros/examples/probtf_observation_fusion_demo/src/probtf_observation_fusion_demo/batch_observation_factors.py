"""Original-acquisition likelihoods pulled into one frozen batch chart.

The affine likelihoods retain every native image mode. The common factor
clock is a coordinate-reference clock, not a new acquisition timestamp.
Only the actually executed ledger supplies command actions; no physical
state, future IK trial, Monte Carlo, or intermediate body approximation is
used here.
"""
from dataclasses import dataclass, replace
from bisect import bisect_left
import hashlib
import json

import numpy as np

from probtf.geometry import axis_angle_to_quat, quat_to_rotmat
from probtf.spherical_law.fusion import FrozenGaussianQuery, Gaussian, SensorPosterior

from .async_state import ActionStep, _action_signature, _recorded_action
from .immutable_body_cache import freeze_json_record
from .sensors import ADAPTERS, PreparedMarkerObservation, prepare_marker_observation


def _readonly(value):
    array = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError('Batch likelihood parameters must be finite.')
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class GaussianLikelihoodComponent:
    label: str
    affine: np.ndarray
    offset: np.ndarray
    observed: np.ndarray
    covariance: np.ndarray
    log_weight: float
    source_ids: tuple = ()

    def __post_init__(self):
        for name in ('affine', 'offset', 'observed', 'covariance'):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        dimension = self.observed.size
        if (self.observed.shape != (dimension,) or self.offset.shape != (dimension,)
                or self.affine.ndim != 2 or self.affine.shape[0] != dimension
                or self.covariance.shape != (dimension, dimension)):
            raise ValueError('Batch likelihood affine Gaussian dimensions disagree.')
        # This is the unchanged observation covariance, never a floor/clip.
        Gaussian(np.zeros(dimension), self.covariance)
        if not np.isfinite(self.log_weight):
            raise ValueError('Every retained observation mode needs its finite native log weight.')
        object.__setattr__(self, 'log_weight', float(self.log_weight))
        object.__setattr__(self, 'source_ids', tuple(self.source_ids))


@dataclass(frozen=True)
class BatchObservationFactor:
    factor_id: str
    observation_ids: tuple
    components: tuple
    record: object
    acquisition_estimate: object
    common_posterior: object
    prepared_marker: object
    common_state_id: str
    common_stamp: float
    common_command: tuple
    acquisition_state_id: str
    acquisition_stamp: float
    acquisition_command: tuple

    @property
    def effective(self):
        return bool(self.components)


def prepare_batch_images(events, common):
    """Construct each distinct image-only likelihood once, before body priors.

    The returned prepared image is shared by Keep/Reset and by every other
    body hypothesis. Its individual native mixtures and central-board record
    remain the existing detector outputs.
    """
    prepared, signatures = {}, {}
    for event in events:
        if event.kind != 'image':
            continue
        image = np.asarray(event.payload['image'])
        signature = (float(event.stamp), tuple(event.command), image.shape,
                     str(image.dtype), hashlib.sha256(image.tobytes()).hexdigest())
        identifier = str(event.event_id)
        if identifier in signatures:
            if signatures[identifier] != signature:
                raise ValueError('A repeated batch image ID has changed acquisition data.')
            continue
        if float(event.payload.get('image_stamp', event.stamp)) != float(event.stamp):
            raise ValueError('Batch image acquisition and pixel timestamps disagree.')
        prepared[identifier] = prepare_marker_observation(common, image, float(event.stamp))
        signatures[identifier] = signature
    return prepared


class AcquisitionTransport:
    """Validate one executed ledger; reuse its exact prefix actions.

    Store ``ledger_record`` once with the batch. Individual factor records
    refer to its ID and prefix range/digest instead of embedding all earlier
    steps again. No prefix is reduced to the endpoint command difference.
    """
    def __init__(self, common_state, steps, body_ids, actions):
        self.common_state = common_state
        self.steps = tuple(steps)
        self.signature = _action_signature(body_ids, actions, len(common_state.command))
        if any(name not in common_state.edge_ids for name in body_ids):
            raise ValueError('A batch action refers to an absent body variable.')
        command = tuple(common_state.command)
        stamp = float(common_state.stamp)
        increment = np.zeros(len(command))
        self._stamps, self._commands = [stamp], [command]
        increments = [increment.copy()]
        chain = hashlib.sha256(b'ProbTF joint-batch actual-command ledger v1').digest()
        self._prefix_hashes = [chain.hex()]
        step_records, previous_index = [], None
        for step in self.steps:
            if (not isinstance(step, ActionStep) or step.start_stamp != stamp
                    or step.start_command != command
                    or previous_index is not None and step.index != previous_index+1):
                raise ValueError('A batch acquisition requires its contiguous actual command prefix.')
            increment += np.asarray(step.joint_increment)
            if not np.all(np.isfinite(increment)):
                raise ValueError('A batch ledger prefix action became nonfinite.')
            stamp, command, previous_index = step.end_stamp, step.end_command, step.index
            self._stamps.append(stamp); self._commands.append(command)
            increments.append(increment.copy())
            record = step.record(); step_records.append(record)
            encoded = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            chain = hashlib.sha256(chain+len(encoded).to_bytes(8, 'big')+encoded).digest()
            self._prefix_hashes.append(chain.hex())
        self._increments = _readonly(increments)
        record = dict(kind='shared complete actual command ledger for joint batch',
            source_state_id=common_state.state_id, source_stamp=common_state.stamp,
            source_command=list(common_state.command), target_stamp=stamp,
            target_command=list(command), steps=step_records,
            prefix_chain_sha256=self._prefix_hashes[-1],
            prefix_chain_convention='joint-batch v1 seed; sha256(previous digest + 8-byte big-endian canonical JSON length + canonical ActionStep JSON)',
            physical_truth_received=False, body_reconstructed_from_stiffness=False,
            axes=[dict(edge_id=name, joint_index=joint, axis=list(axis))
                  for name, joint, axis in self.signature])
        self.ledger_id = hashlib.sha256(json.dumps(record, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.ledger_record = freeze_json_record(dict(record, ledger_id=self.ledger_id))
        self._current_prefix = 0

    def _index(self, stamp, command):
        stamp = float(stamp)
        index = bisect_left(self._stamps, stamp)
        if (index >= len(self._stamps) or self._stamps[index] != stamp
                or tuple(command) != self._commands[index]):
            raise ValueError('The actual ledger endpoint differs from the batch observation acquisition.')
        return index

    def at(self, acquisition_stamp, acquisition_command, *, version=None):
        index = self._index(acquisition_stamp, acquisition_command)
        increment = self._increments[index]
        state = _recorded_action(self.common_state, self._commands[index], increment, self.signature)
        state = replace(state, stamp=self._stamps[index],
            version=self.common_state.version if version is None else int(version))
        chart = np.eye(self.common_state.mean.size)
        for name, joint, axis in self.signature:
            if increment[joint] != 0.:
                edge = self.common_state.edge_ids.index(name)
                chart[6*edge+3:6*edge+6, 6*edge+3:6*edge+6] = quat_to_rotmat(
                    axis_angle_to_quat(axis, increment[joint])).T
        chart.setflags(write=False)
        first = None if not self.steps else self.steps[0].index
        record = dict(kind='compact prefix of shared actual joint-batch ledger',
            ledger_id=self.ledger_id, source_state_id=self.common_state.state_id,
            source_stamp=self.common_state.stamp, source_command=list(self.common_state.command),
            acquisition_state_id=state.state_id, acquisition_stamp=self._stamps[index],
            acquisition_command=list(self._commands[index]),
            start_index=first, end_index_exclusive=first if not index else self.steps[index-1].index+1,
            step_count=index, source_prefix_chain_sha256=self._prefix_hashes[0],
            prefix_chain_sha256=self._prefix_hashes[index], joint_increment=increment.tolist(),
            prefix_chain_convention='joint-batch v1; complete definition in referenced shared ledger',
            complete_steps_stored_once='batch shared_actual_command_ledger identified by ledger_id',
            physical_truth_received=False, body_reconstructed_from_stiffness=False,
            future_trial_family=False)
        return state, chart, freeze_json_record(record)

    def advance_to(self, acquisition_stamp, acquisition_command, *, version=None):
        index = self._index(acquisition_stamp, acquisition_command)
        if index < self._current_prefix:
            raise ValueError('Chronological batch acquisition transport cannot move backwards.')
        self._current_prefix = index
        return self.at(acquisition_stamp, acquisition_command, version=version)


def transport_state_to_acquisition(common_state, steps, body_ids, actions, *,
                                   acquisition_stamp, acquisition_command, version=None):
    """Return an actual-ledger push-forward and its exact affine chart action.

    Use a prefix of the immutable executed ledger from the batch start.
    The same G acts on all cross blocks: P_acq=G P_common G.T. The marker's
    root slot and all other uncommanded variables have the identity action.
    """
    steps = tuple(steps)
    command = tuple(float(value) for value in common_state.command)
    stamp = float(common_state.stamp)
    signature = _action_signature(body_ids, actions, len(command))
    increment = np.zeros(len(command))
    previous_index = None
    for step in steps:
        if (not isinstance(step, ActionStep) or step.start_stamp != stamp
                or step.start_command != command
                or previous_index is not None and step.index != previous_index+1):
            raise ValueError('A batch acquisition requires its contiguous actual command prefix.')
        increment += np.asarray(step.joint_increment)
        stamp, command, previous_index = step.end_stamp, step.end_command, step.index
    if (float(acquisition_stamp) != stamp or tuple(acquisition_command) != command
            or not np.all(np.isfinite(increment))):
        raise ValueError('The actual ledger endpoint differs from the batch observation acquisition.')
    state = _recorded_action(common_state, command, increment, signature)
    state = replace(state, stamp=stamp, version=common_state.version if version is None else int(version))
    chart = np.eye(common_state.mean.size)
    for name, joint, axis in signature:
        index = common_state.edge_ids.index(name)
        if increment[joint] != 0.:
            rotation = quat_to_rotmat(axis_angle_to_quat(axis, increment[joint]))
            chart[6*index+3:6*index+6, 6*index+3:6*index+6] = rotation.T
    chart.setflags(write=False)
    record = dict(kind='actual executed command prefix for batch acquisition',
        source_state_id=common_state.state_id, source_stamp=common_state.stamp,
        source_command=list(common_state.command), acquisition_state_id=state.state_id,
        acquisition_stamp=stamp, acquisition_command=list(command),
        joint_increment=increment.tolist(), steps=[step.record() for step in steps],
        chart_action=chart.tolist(), physical_truth_received=False,
        body_reconstructed_from_stiffness=False, future_trial_family=False)
    return state, chart, freeze_json_record(record)


def build_observation_factor(common_state, acquisition_state, observation, paths, common, image, *,
                             common_to_acquisition=None, event_metadata=None,
                             ledger_record=None, prepared_marker=None):
    """Reuse the existing adapter, then expose its exact affine likelihood.

    If xi_acq=G xi_common, the likelihood uses H_common=H_acq G and the
    original query intercept. Detector image weights, chart-volume constants,
    observation covariance normalizers and every native component survive.
    The adapters' original local posterior is retained for acquisition
    diagnostics and fresh-goal construction; it is not the joint-batch result.
    """
    sensor = observation['sensor']
    if sensor not in ('imu', 'marker'):
        raise ValueError('A batch observation factor requires an IMU or marker likelihood.')
    if acquisition_state.edge_ids != common_state.edge_ids:
        raise ValueError('Batch common and acquisition charts need the same joint variables.')
    chart = np.eye(common_state.mean.size) if common_to_acquisition is None else np.asarray(common_to_acquisition, dtype=float)
    if chart.shape != common_state.covariance.shape or not np.all(np.isfinite(chart)):
        raise ValueError('A batch observation needs its finite complete ledger chart action.')
    if not np.allclose(chart @ common_state.covariance @ chart.T,
                       acquisition_state.covariance, rtol=1e-9, atol=1e-12):
        raise ValueError('The acquisition body law differs from the declared frozen-chart ledger push-forward.')
    event = dict(event_metadata or {})
    if (event.get('acquisition_stamp', acquisition_state.stamp) != acquisition_state.stamp
            or tuple(event.get('sent_command', acquisition_state.command)) != acquisition_state.command
            or event.get('observation_id', observation['observation_id']) != observation['observation_id']):
        raise ValueError('Batch observation metadata differs from its original acquisition.')
    if prepared_marker is not None and sensor != 'marker':
        raise ValueError('An IMU event cannot use a prepared image likelihood.')
    if sensor == 'marker':
        if prepared_marker is None:
            prepared_marker = prepare_marker_observation(common, image, acquisition_state.stamp)
        if not isinstance(prepared_marker, PreparedMarkerObservation):
            raise TypeError('A batch image needs its actual prepared marker observation.')
        estimate = ADAPTERS[sensor](acquisition_state, observation, paths, common, image,
                                    prepared=prepared_marker)
    else:
        estimate = ADAPTERS[sensor](acquisition_state, observation, paths, common, image,
                                    shared_observation_frame=True)
    source_ids = (str(observation['observation_id']), acquisition_state.state_id, common_state.state_id)
    components = []
    common_posterior = None
    if estimate.posterior is not None:
        frozen = estimate.posterior.query
        affine = frozen.jacobian @ chart
        offset = frozen.distribution.mean-affine @ common_state.mean
        # The common timestamp explicitly names the linearization reference.
        # Original acquisition metadata is separately immutable below.
        pulled = FrozenGaussianQuery(Gaussian(common_state.mean, common_state.covariance),
            frozen.distribution, affine, 'batch-pullback:'+frozen.query_id,
            common_state.version, common_state.stamp, common_state.command,
            common_state.component_id, frozen.frame, state_id=common_state.state_id)
        common_posterior = SensorPosterior(pulled, estimate.posterior.sensor_id,
            estimate.posterior.observation_id, estimate.posterior.components)
        if sensor == 'imu':
            basis = np.asarray(estimate.diagnostics['tangent_basis'])
            observed = basis.T @ np.asarray(estimate.diagnostics['observed_direction'])
            covariance = float(estimate.diagnostics['measurement_variance'])*np.eye(2)
            components.append(GaussianLikelihoodComponent('gravity', affine, offset,
                observed, covariance, 0., source_ids))
        else:
            # These details are the exact native likelihood Gaussians used
            # inside marker_estimate, before body-prior overlap weighting.
            for item in estimate.diagnostics['components']:
                components.append(GaussianLikelihoodComponent(item['label'], affine, offset,
                    item['image_only_mean'], item['image_only_covariance'],
                    item['image_log_weight']+item['chart_log_volume'], source_ids))
    record = dict(kind='original-acquisition affine likelihood in frozen common batch chart',
        sensor=sensor, sensor_id=observation['sensor_id'], observation_id=observation['observation_id'],
        effective=bool(components), common_state_id=common_state.state_id,
        common_factor_stamp=common_state.stamp, common_factor_command=list(common_state.command),
        common_factor_clock_role='frozen coordinate reference; never an acquisition timestamp',
        acquisition_state_id=acquisition_state.state_id, acquisition_stamp=acquisition_state.stamp,
        acquisition_command=list(acquisition_state.command), acquisition_event=event,
        actual_command_ledger={} if ledger_record is None else ledger_record,
        common_to_acquisition_chart=chart.tolist(),
        native_component_count=len(components), native_mixture_preserved=True,
        likelihood_weights='native image log weights plus scalar chart-volume Jacobian; no posterior overlap weights' if sensor == 'marker' else 'single observed-direction S2 tangent likelihood',
        components=[dict(label=item.label, affine=item.affine.tolist(), offset=item.offset.tolist(),
            observed=item.observed.tolist(), covariance=item.covariance.tolist(),
            log_weight=item.log_weight, source_ids=list(item.source_ids)) for item in components],
        adapter_diagnostics=estimate.diagnostics, intermediate_body_projection_used=False,
        physical_truth_received=False, monte_carlo_used=False)
    frozen_record = freeze_json_record(record)
    identity = hashlib.sha256(json.dumps(frozen_record, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return BatchObservationFactor(identity, (str(observation['observation_id']),) if components else (),
        tuple(components), frozen_record, estimate, common_posterior, prepared_marker,
        common_state.state_id, common_state.stamp, common_state.command,
        acquisition_state.state_id, acquisition_state.stamp, acquisition_state.command)
