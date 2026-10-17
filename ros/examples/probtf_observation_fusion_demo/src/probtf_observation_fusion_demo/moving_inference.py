"""Retained transform-candidate inference on the moving switch scene.

The directly supplied family initializes the body and defines the explicit
body-redraw change kernel. Each new observation compares that kernel with
the retained joint proxy pushed by its declared command action. Current
observation/change modes remain explicit goal components. No physical truth
or invented process covariance enters inference.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import rospkg
from scipy.spatial.transform import Rotation
import yaml
from scipy.special import logsumexp

from probtf.geometry import axis_angle_to_quat, quat_right_matrix, quat_to_rotmat, rotmat_to_quat
from probtf.spherical_law.first_order import BinghamTransform, JointTransformState, point_query
from probtf.spherical_law.fusion import PosteriorComponent
from symaware_grasp.probtf_config import bingham_parameter_from_mode

from .model import (
    FrozenBodyBranch, Stage, WeightedState, _joint_record, component_log_weight, run,
)
from .body_projection import project_update, retain_goal_components
from .moving_control import command_state
from .bayes_factor import transition_priors, projection_input, retain_transition_goals
from .fresh_goal import fresh_marker_goals
from .sensor_event_store import _encode


@dataclass(frozen=True)
class CandidateMeasureView:
    """Unnormalized candidate scale and the actual normalized joint proxy.

    The proxy objects are the ones consumed by control and display. Their
    normalization is a computational view, not a unit-mass declaration for
    the underlying candidate measure. Log mass uses an explicit arbitrary
    initial reference and the recorded sensor-proxy evidence measure.
    """
    components: tuple
    log_total_mass: float
    evidence_measure: str


def _origin(joint):
    element = joint.find("origin")
    if element is None:
        return np.zeros(3), np.array([1., 0., 0., 0.])
    translation = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    return translation, rotmat_to_quat(Rotation.from_euler("xyz", rpy).as_matrix())


def _fixed_edge(joint):
    if joint.get("type") != "fixed":
        raise ValueError("An explicitly known attachment must be a fixed URDF joint.")
    translation, quaternion = _origin(joint)
    return BinghamTransform(joint.find("child").get("link"), translation,
                            rotation_wxyz=quaternion)


class MovingInference:
    """All sensor lists share the same laws, geometry and frozen-time update.

    ``robot_description`` is known attachment geometry, not a physical pose.
    ``observation_templates`` determines only which observations are enabled.
    IMU vectors and camera pixels are read solely from ``SensorInputs``.
    """

    def __init__(self, robot_description, observation_templates, config_path=None):
        packages = rospkg.RosPack()
        config_path = Path(config_path or Path(packages.get_path(
            "probtf_observation_fusion_demo")) / "config/moving_estimation.yaml")
        settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        law_path = (Path(packages.get_path(settings["body_law_package"]))
                    / settings["body_law_file"])
        self.law = yaml.safe_load(law_path.read_text(encoding="utf-8"))
        if self.law.get("independent") is not True:
            raise ValueError("The initial product law must be explicitly declared.")
        if self.law["root_frame"] != settings["root_frame"]:
            raise ValueError("The direct law and inference root frames differ.")
        self.reference_command = np.asarray(self.law["reference_command"], dtype=float)
        self.templates = tuple(deepcopy(item) for item in observation_templates)
        self.settings = settings
        self.surprise_settings = dict(settings.get('bayes_factor_surprise', {}))
        self.surprise_enabled = bool(self.surprise_settings.get('enabled', False))
        self.change_probability = float(self.surprise_settings.get('change_probability', .25))
        if not np.isfinite(self.change_probability) or not 0. <= self.change_probability < 1.:
            raise ValueError('Bayes-factor change probability must lie in [0, 1).')
        self.last_key = None
        self.last_result = None
        self.last_observation_signature = None
        self.components = ()
        self.goal_components = ()
        self.fixed_goal_memory = None
        self.last_observations = ()
        self.candidate_measure = None
        self._sensor_events = {}
        self._last_sensor_event_id = None
        self._last_event_image = None

        # URDF is used only for declared geometry.  The direct uncertain laws
        # supply each active edge; masses/inertia and physics are never read.
        tree = ET.fromstring(robot_description)
        joints = {joint.find("child").get("link"): joint
                  for joint in tree.findall("joint")}
        self.fixed = tuple(_fixed_edge(joints[name]) for name in (
            "tool0", "press_rod", "rod_tip", "camera_optical_frame"))
        expected_parent = settings["root_frame"]
        self.base_parameters = []
        self.actions = []
        self.body_ids = []
        for edge in self.law["edges"]:
            if edge["parent"] != expected_parent:
                raise ValueError("The directly supplied laws must form one ordered chain.")
            expected_parent = edge["child"]
            if edge["child"] == "tool0":
                fixed = self.fixed[0]
                if not np.array_equal(fixed.translation, edge["translation"]["mean"]):
                    raise ValueError("The tool attachment differs from its direct input law.")
                if edge["orientation"]["kind"] != "dirac":
                    raise ValueError("This declared tool attachment must be exact.")
                continue
            orientation = edge["orientation"]
            if orientation["kind"] != "bingham":
                raise ValueError("All six active transform candidates must be finite Bingham laws.")
            joint = joints[edge["child"]]
            translation, quaternion = _origin(joint)
            if (not np.array_equal(translation, edge["translation"]["mean"])
                    or not np.allclose(quaternion, [1., 0., 0., 0.])):
                raise ValueError("The direct family geometry differs from the shared robot.")
            coupling = edge["translation"].get("rotation_coupling", np.zeros((3, 9)))
            if np.any(np.asarray(coupling) != 0):
                raise ValueError("This moving input declares independent local translation/rotation.")
            matrix = orientation.get("parameter_matrix")
            if matrix is None:
                matrix = bingham_parameter_from_mode(
                    orientation["mode_wxyz"], orientation["eigenvalues"])
            self.base_parameters.append(np.asarray(matrix, dtype=float))
            self.body_ids.append(edge["child"])
            self.actions.append(edge["right_joint_action"])
        if len(self.body_ids) != 6:
            raise ValueError("The shared moving arm requires all six local transform candidates.")

        marker_data = settings["marker_prior"]
        marker_quaternion = rotmat_to_quat(Rotation.from_euler(
            "xyz", marker_data["orientation_rpy"]).as_matrix())
        self.marker = BinghamTransform(
            marker_data["edge_id"], marker_data["translation"],
            bingham=bingham_parameter_from_mode(
                marker_quaternion, marker_data["bingham_eigenvalues"]),
            translation_covariance=float(marker_data["translation_standard_deviation"])**2*np.eye(3))
        body_path = tuple((name, False) for name in self.body_ids)
        flange_path = body_path + (("tool0", False),)
        camera_path = flange_path + (("camera_optical_frame", False),)
        marker_path = ((self.marker.edge_id, False),)
        self.paths = {
            "links": {name: body_path[:index+1] for index, name in enumerate(self.body_ids)},
            "camera": camera_path,
            "hand": flange_path + (("press_rod", False), ("rod_tip", False)),
            "marker": marker_path,
            "camera_marker": tuple((name, not inverse) for name, inverse in reversed(camera_path))
                             + marker_path,
        }
        self.paths["links"]["tool0"] = flange_path
        self.common = {
            "root_frame": settings["root_frame"],
            "update": {},
            "prior": {"independence": "initial supplied product; explicit body-redraw branch independent of the retained environment; full joint moments retained in the internal approximation",
                      "body": [{"edge_id": name} for name in self.body_ids],
                      "marker": {"status": marker_data["status"]}},
            "known_geometry": {"source": "fixed attachment geometry parsed from shared URDF; masses and physical poses excluded"},
            "camera": settings["camera"], "marker": settings["marker"],
            "marker_board": settings.get("marker_board", {}),
            "display": settings["display"], "control": settings["control"],
        }

    def prior_at(self, command, stamp, frame_index):
        """Query the directly supplied initial family, with no pose truth.

        Besides initialization, this is the specified reset branch of the
        change model. The retained posterior remains the competing branch;
        this method never unconditionally replaces an accumulated posterior.
        """
        command = np.asarray(command, dtype=float)
        if command.shape != self.reference_command.shape:
            raise ValueError("Command dimension differs from the direct input family.")
        transforms = []
        for edge, matrix, action in zip(self.law["edges"], self.base_parameters, self.actions):
            index = int(action["joint_index"])
            quaternion = axis_angle_to_quat(action["axis"], command[index]-self.reference_command[index])
            right = quat_right_matrix(quaternion)
            transforms.append(BinghamTransform(
                edge["child"], edge["translation"]["mean"],
                bingham=right @ matrix @ right.T,
                translation_covariance=edge["translation"]["covariance"]))
        transforms += list(self.fixed) + [self.marker]
        return JointTransformState.from_independent(
            transforms, version=3*int(frame_index)+1, stamp=float(stamp),
            command=tuple(command), component_id="common_body_prior")

    def update(self, sensor_inputs, frame_index, *, command_map=None):
        """Run one same-time common update; duplicate ticks return that snapshot."""
        return self._update_common(sensor_inputs, frame_index, command_map=command_map)

    def update_sensor_event(self, event, event_index=None, *, command_map=None,
                            probability_unit=None):
        """Assimilate only this acquired sensor into the retained joint law.

        Distinct sensors may share an acquisition stamp; their immutable DB
        sequence orders the separate updates. Joint events are command-context
        notifications, never a measured-angle likelihood or a surprise event.
        The physical frame index comes from explicit ``event_index`` or the
        acquisition metadata, falling back to sequence for standalone callers.
        The immediate identical replay returns the same object. Replaying an
        older consumed ID raises instead of rolling the current law backwards.
        """
        from .moving_scene import SensorInputs

        if event.kind not in ('joint', 'imu', 'image'):
            raise ValueError('Unsupported sensor event kind.')
        if event.status not in ('pending', 'processing', 'completed'):
            raise ValueError('A sensor event must have a ready, chronologically released payload.')
        payload = event.payload
        allowed = {'time', 'frame_index'} | ({'image', 'image_stamp', 'image_path',
            'image_sha256'} if event.kind == 'image' else
            {'gravity_in_sensor', 'imu_target_link', 'imu_sigma_rad'} if event.kind == 'imu'
            else {'command_velocity'})
        if set(payload)-allowed:
            raise ValueError('Undeclared sensor-event fields: '+', '.join(sorted(set(payload)-allowed)))
        if 'frame_index' in payload and (isinstance(payload['frame_index'], bool)
                or not isinstance(payload['frame_index'], int) or payload['frame_index'] < 0):
            raise ValueError('Acquisition frame_index must be a nonnegative integer.')
        if 'image_path' in payload and (not isinstance(payload['image_path'], str)
                or not payload['image_path'] or '\x00' in payload['image_path']):
            raise ValueError('Acquisition image_path must be a nonempty path string.')
        if 'image_sha256' in payload and (not isinstance(payload['image_sha256'], str)
                or re.fullmatch(r'[0-9a-fA-F]{64}', payload['image_sha256']) is None):
            raise ValueError('Acquisition image_sha256 must be a 64-digit hexadecimal SHA256.')
        stamp = float(event.stamp)
        command = np.asarray(event.command, dtype=float)
        if (not np.isfinite(stamp) or command.shape != self.reference_command.shape
                or not np.all(np.isfinite(command))):
            raise ValueError('Sensor acquisition stamp and sent command must be finite and valid.')
        requested_index = payload.get('frame_index', event.sequence) if event_index is None else event_index
        if (isinstance(requested_index, (bool, np.bool_)) or
                not isinstance(requested_index, (int, np.integer)) or requested_index < 0):
            raise ValueError('The event index must be a nonnegative integer.')
        index = int(requested_index)
        if 'frame_index' in payload and index != payload['frame_index']:
            raise ValueError('Requested event index and acquired physical frame_index disagree.')
        _, _, payload_hash = _encode(payload)
        signature = (event.kind, stamp, tuple(command), payload_hash)
        if event.event_id in self._sensor_events:
            if self._sensor_events[event.event_id] != signature:
                raise ValueError('A repeated sensor event has different acquisition data.')
            if event.event_id != self._last_sensor_event_id:
                raise ValueError('An already consumed older sensor event requires checkpoint replay.')
            return self.last_result
        if event.kind == 'image':
            if 'image' not in payload:
                raise ValueError('A ready image event must contain acquired pixels.')
            if float(payload.get('image_stamp', stamp)) != stamp:
                raise ValueError('Image payload and acquisition timestamps disagree.')
            image = np.asarray(payload['image'])
            if (image.dtype != np.uint8 or image.ndim not in (2, 3)
                    or (image.ndim == 3 and image.shape[2] not in (1, 3, 4))):
                raise ValueError('An image event requires uint8 camera pixels.')
            if ('image_sha256' in payload and hashlib.sha256(image.tobytes()).hexdigest()
                    != payload['image_sha256'].lower()):
                raise ValueError('Acquisition image_sha256 disagrees with the acquired pixels.')
        else:
            image = (np.zeros((int(self.common['camera']['height']),
                               int(self.common['camera']['width']), 3), dtype=np.uint8)
                     if self._last_event_image is None else self._last_event_image)
        if event.kind == 'imu':
            if 'gravity_in_sensor' not in payload:
                raise ValueError('An IMU event requires its acquired direction.')
            gravity = np.asarray(payload['gravity_in_sensor'], dtype=float)
            target = str(payload.get('imu_target_link', 'link_3'))
            sigma = float(payload.get('imu_sigma_rad', .018))
            if (gravity.shape != (3,) or not np.all(np.isfinite(gravity))
                    or np.linalg.norm(gravity) == 0. or not np.isfinite(sigma) or sigma <= 0.):
                raise ValueError('An IMU event needs a finite nonzero direction and positive sigma.')
            if target not in self.body_ids and target not in ('tool0',):
                raise ValueError('The acquired IMU target is outside the known body.')
        else:
            gravity, target, sigma = np.array([0., 0., -1.]), 'link_3', .018
        sensor = {'imu': 'imu', 'image': 'marker', 'joint': None}[event.kind]
        templates = tuple(item for item in self.templates if item['sensor'] == sensor)
        if len(templates) > 1:
            raise ValueError('One acquired sensor event must identify exactly one enabled sensor template.')
        probability_unit = (self.surprise_settings.get('event_probability_unit')
                            if probability_unit is None else probability_unit)
        if (self.surprise_enabled and self.change_probability > 0. and templates
                and probability_unit != 'per_effective_sensor_event'):
            raise ValueError('Sensor-event Bayes surprise requires an explicitly configured '
                             'event_probability_unit=per_effective_sensor_event.')
        inputs = SensorInputs(time=float(payload.get('time', stamp)), stamp=stamp,
            command=command, image=image, gravity_in_sensor=gravity,
            imu_target_link=target, imu_sigma_rad=sigma)
        if not np.isfinite(inputs.time):
            raise ValueError('Sensor event simulation time must be finite.')
        event_record = dict(event_id=str(event.event_id), kind=event.kind,
            sequence=int(event.sequence), event_index=index, acquisition_stamp=stamp,
            sent_command=command.tolist(), enabled=bool(templates),
            observation_id=str(event.event_id) if templates else None,
            probability_unit=probability_unit,
            chronological_separate_sensor_update=True, simultaneous_sensor_wait=False,
            joint_notification_is_observation=False, payload_hash=payload_hash,
            acquisition_metadata={key: payload[key] for key in
                ('frame_index', 'image_path', 'image_sha256') if key in payload},
            acquisition_metadata_is_observation=False,
            law_metadata_index='physical acquisition frame index; may repeat across separate sensor events',
            update_identity='event_id and sequence plus content-derived state_id; numeric version alone is not an observation identifier',
            physical_truth_received=False, monte_carlo_used=False)
        result = self._update_common(inputs, index, command_map=command_map,
            observation_templates=templates, sensor_event=event_record)
        self._sensor_events[event.event_id] = signature
        self._last_sensor_event_id = event.event_id
        if event.kind == 'image':
            self._last_event_image = np.array(image, copy=True)
            self._last_event_image.setflags(write=False)
        return result

    def _update_common(self, sensor_inputs, frame_index, *, command_map=None,
                       observation_templates=None, sensor_event=None):
        key = (int(frame_index), float(sensor_inputs.stamp), tuple(sensor_inputs.command))
        if sensor_event is not None:
            key += (sensor_event['event_id'],)
        signature = hashlib.sha256(np.asarray(sensor_inputs.image).tobytes()
            + np.asarray(sensor_inputs.gravity_in_sensor).tobytes()
            + repr((sensor_inputs.imu_target_link, sensor_inputs.imu_sigma_rad)).encode()
            + (b'' if command_map is None else json.dumps(command_map.record(),
                sort_keys=True, separators=(',', ':')).encode())).hexdigest()
        if key == self.last_key:
            if signature != self.last_observation_signature:
                raise ValueError("A repeated physical tick contains different observation data.")
            return self.last_result
        if self.last_key is not None and (int(frame_index) < self.last_key[0]
                or (sensor_event is None and int(frame_index) == self.last_key[0])):
            raise ValueError("Frame indices must increase; construct a fresh inference on replay.")
        if self.last_key is not None and (float(sensor_inputs.stamp) < self.last_key[1]
                or (sensor_event is None and float(sensor_inputs.stamp) == self.last_key[1])):
            raise ValueError("A new observation tick must have an increasing timestamp.")
        prediction = []
        goal_prediction = []
        predicted_goals = ()
        if not self.components:
            prior = self.prior_at(sensor_inputs.command, sensor_inputs.stamp, frame_index)
            priors = (WeightedState(prior.component_id, 1., prior, 0.),)
        else:
            priors = []
            for index, source in enumerate(self.components):
                predicted = (command_state(source.state, sensor_inputs.command,
                                           self.body_ids, self.actions) if command_map is None
                             else command_map.push(source.state, sensor_inputs.command,
                                                   self.body_ids, self.actions))
                label = "predicted:{}:{}".format(int(frame_index), index)
                predicted = replace(predicted, version=3*int(frame_index)+1,
                    stamp=float(sensor_inputs.stamp), component_id=label)
                priors.append(WeightedState(label, source.weight, predicted,
                                            component_log_weight(source)))
                prediction.append(dict(source_label=source.label,
                    source_state_id=source.state.state_id, source_version=source.state.version,
                    source_stamp=source.state.stamp, source_command=list(source.state.command),
                    transported_label=label, transported_state_id=predicted.state_id,
                    transported_version=predicted.version, transported_stamp=predicted.stamp,
                    transported_command=list(predicted.command), weight=source.weight,
                    log_weight=component_log_weight(source)))
            priors = tuple(priors)
            goals = []
            for index, source in enumerate(self.goal_components):
                predicted = (command_state(source.state, sensor_inputs.command,
                                           self.body_ids, self.actions) if command_map is None
                             else command_map.push(source.state, sensor_inputs.command,
                                                   self.body_ids, self.actions))
                label = "predicted_goal:{}:{}".format(int(frame_index), index)
                predicted = replace(predicted, version=priors[0].state.version,
                    stamp=float(sensor_inputs.stamp), component_id=label)
                goals.append(WeightedState(label, source.weight, predicted,
                                            component_log_weight(source)))
                goal_prediction.append(dict(source_label=source.label,
                    source_state_id=source.state.state_id, source_version=source.state.version,
                    source_stamp=source.state.stamp, source_command=list(source.state.command),
                    transported_label=label, transported_state_id=predicted.state_id,
                    transported_version=predicted.version, transported_stamp=predicted.stamp,
                    transported_command=list(predicted.command), weight=source.weight,
                    log_weight=component_log_weight(source)))
            predicted_goals = tuple(goals)
        observations = []
        for template in (self.templates if observation_templates is None else observation_templates):
            observation = deepcopy(template)
            if observation["sensor"] == "imu":
                observation["sensor_id"] = "imu_" + sensor_inputs.imu_target_link
            observation["observation_id"] = "{}:frame:{:05d}".format(
                observation["sensor_id"], int(frame_index))
            if sensor_event is not None:
                observation['observation_id'] = sensor_event['event_id']
            if observation["sensor"] == "imu":
                observation.update(target_link=sensor_inputs.imu_target_link,
                    gravity_in_root=[0., 0., -1.],
                    gravity_in_sensor=np.asarray(sensor_inputs.gravity_in_sensor).tolist(),
                    direction_sigma_rad=float(sensor_inputs.imu_sigma_rad))
            observations.append(observation)
        transition = False
        surprise = dict(enabled=self.surprise_enabled, applied=False,
            change_probability=self.change_probability,
            prior_change_odds=self.change_probability/(1.-self.change_probability),
            probability_unit='one fresh nonempty valid observation packet; common across sensors',
            reason=('disabled' if not self.surprise_enabled else
                    'zero change probability: ordinary observation fusion'
                    if self.change_probability == 0. else 'no valid new observations'),
            log_bayes_factor=None, posterior_change_probability=None,
            log_keep_evidence=None, log_reset_evidence=None,
            monte_carlo_used=False, physical_truth_received=False)
        if self.surprise_enabled and self.change_probability > 0. and observations:
            native_priors, transition_record = transition_priors(priors[0],
                self.prior_at(sensor_inputs.command, sensor_inputs.stamp, frame_index),
                self.change_probability, marker_id=self.marker.edge_id)
            result = run(self.common, tuple(observations), sensor_inputs.image,
                prior_components=native_priors, paths=self.paths,
                joint_evidence=True, shared_observation_frame=True)
            if result.sensor_posteriors:
                transition = True
                evidences = result.records['joint_observation_evidence']
                keep_log, reset_log = (item['log_evidence'] for item in evidences['parents'])
                log_bf = reset_log-keep_log
                reset_mass = float(logsumexp([component_log_weight(item)
                    for item in result.frozen_body_branches[1].posterior_components]))
                keep_mass = float(logsumexp([component_log_weight(item)
                    for item in result.frozen_body_branches[0].posterior_components]))
                if not np.isclose(reset_mass-keep_mass,
                    np.log(self.change_probability)-np.log1p(-self.change_probability)+log_bf,
                    atol=1e-9, rtol=1e-9):
                    raise ValueError('Fused change odds and normalized joint evidence disagree.')
                surprise.update(applied=True, reason='fresh valid observation packet',
                    log_bayes_factor=float(log_bf),
                    posterior_change_probability=float(np.exp(reset_mass)),
                    posterior_keep_log_probability=keep_mass,
                    posterior_change_log_probability=reset_mass,
                    log_keep_evidence=keep_log, log_reset_evidence=reset_log,
                    log_observation_evidence=evidences['log_evidence'],
                    observation_ids=list(result.records['observation_ids']),
                    transition=transition_record,
                    prior_source_state_ids=[item.state.state_id for item in native_priors],
                    initial_packet=not bool(self.components))
            else:
                # Missing images are not evidence for a change. Preserve the
                # ordinary prior bit for bit and retain the detector result.
                missing = result
                result = run(self.common, (), sensor_inputs.image,
                    prior_components=priors, paths=self.paths)
                result = replace(result, local_estimates=missing.local_estimates,
                                 detection_image=missing.detection_image)
                result.records['sensor_estimates'] = missing.records['sensor_estimates']
        else:
            result = run(self.common, tuple(observations), sensor_inputs.image,
                         prior_components=priors, paths=self.paths,
                         joint_evidence=sensor_event is not None)
        result.records['bayes_factor_surprise'] = surprise
        result = self._project_internal_update(result, predicted_goals)
        surprise['internal_state_id'] = result.stages[-1].components[0].state.state_id
        surprise['goal_source_state_ids'] = [item.state.state_id
                                            for item in result.stages[-1].goal_components]
        self._record_fixed_goal_memory(result)
        if sensor_event is not None:
            result.records['sensor_event'] = sensor_event
            if sensor_event['kind'] != 'image' and result.records['fixed_goal_memory_event'] == 'image_missing_hold':
                result.records['fixed_goal_memory_event'] = 'no_image_event_hold'
            surprise['probability_unit'] = (sensor_event['probability_unit'] if observations
                                             else 'not applied: no enabled observation in this event')
            self._record_candidate_measure(result)
        result.records["prediction"] = prediction
        result.records["goal_prediction"] = goal_prediction
        result.records["prediction_command_correction"] = (
            None if command_map is None else command_map.record())
        result.records["prediction_command_action"] = (
            None if command_map is None else command_map.selection_record(sensor_inputs.command))
        recorded_history = (command_map is not None and
            result.records['prediction_command_correction'].get('kind') == 'recorded_executed_command_history')
        result.records["moving_input_family"] = {
            "law_id": self.law["law_id"], "revision": self.law["revision"],
            "command_family": ("actual executed command history with each interval's frozen K action; all intervals retained across observation scheduling" if recorded_history else
                "configured Bingham right action; translation law unchanged"
                if command_map is None else "same posterior right action by the previous cycle's frozen estimated-K equilibrium increment; translation law unchanged"),
            "temporal_policy": ("Bayes-factor weighted continuation and direct-family body redraw on fresh valid observation packets; retained marker marginal; single analytic joint moment proxy and all current goal modes" if transition else
                "retain one full joint moment approximation, including unknown-marker memory and all cross blocks; preserve the current observation mixture in the goal criterion and display"),
            "initialization": not bool(self.components),
            "frame_index": int(frame_index), "time": float(sensor_inputs.time),
            "physical_truth_received": False,
        }
        self.last_key, self.last_result = key, result
        self.last_observation_signature = signature
        self.components = tuple(result.stages[-1].components)
        self.goal_components = tuple(result.stages[-1].goal_components)
        self.last_observations = tuple(observations)
        return result

    def _record_candidate_measure(self, result):
        previous = 0. if self.candidate_measure is None else self.candidate_measure.log_total_mass
        evidence = result.records.get('joint_observation_evidence')
        valid = bool(result.records.get('valid_observation_count', 0))
        increment = 0. if not valid else float(evidence['log_evidence'])
        if not np.isfinite(increment):
            raise ValueError('Candidate mass requires finite comparable observation evidence.')
        measure = ('arbitrary initial log-scale zero; normalized linearized Bingham-derived '
            'sensor Gaussian proxy evidence with declared native image chart-volume correction')
        self.candidate_measure = CandidateMeasureView(tuple(result.stages[-1].components),
                                                      previous+increment, measure)
        result.records['candidate_measure'] = dict(
            kind='unnormalized retained joint transform candidate measure',
            normalized_shape_view=True, log_total_mass=self.candidate_measure.log_total_mass,
            previous_log_total_mass=previous, log_mass_increment=increment,
            initial_global_log_scale=0., global_scale_is_arbitrary=True,
            evidence_measure=measure, observation_evidence=deepcopy(evidence),
            shape_source_state_ids=[item.state.state_id for item in self.candidate_measure.components],
            shape_source_versions=[item.state.version for item in self.candidate_measure.components],
            shape_source_stamps=[item.state.stamp for item in self.candidate_measure.components],
            shape_source_commands=[list(item.state.command) for item in self.candidate_measure.components],
            shape_log_weights=[component_log_weight(item) for item in self.candidate_measure.components],
            no_observation_mass_is_identity=not valid,
            body_edge_ids=list(self.body_ids),
            joint_proxy_includes_retained_environment_and_all_cross_covariances=True,
            body_candidate_is_joint_proxy_marginal_with_the_same_total_mass=True,
            bayes_factor_hypothesis_scale=dict(normalized_proxy_views=True,
                keep_log_total_mass=previous, reset_log_total_mass=previous,
                reset_kernel_preserves_total_mass=True),
            approximation='same analytic joint moment proxy; not exact raw-pixel Bayesian evidence',
            monte_carlo_used=False)

    def _project_internal_update(self, result, predicted_goals):
        """Keep one recursive internal law and the explicit current goal mixture.

        Moment matching occurs only after sensor-specific updates and common-
        prior-corrected fusion in their single frozen input chart. The same
        projected body is used by stiffness, control, display and prediction.
        A fresh image constructs task goals by composing its native relative
        marker modes through this captured common camera/body law. The
        retained unknown-marker posterior remains a separate internal state.
        Without a new marker observation their root marker laws remain fixed;
        only the body and its side of shared cross blocks update.
        """
        transition = bool(result.records.get('bayes_factor_surprise', {}).get('applied'))
        native_branches = result.frozen_body_branches
        projected_input = projection_input(result) if transition else result
        projection = project_update(projected_input)
        raw_stage = result.stages[-1]
        marker_observed = any(estimate.posterior is not None
                              and estimate.detector_result is not None
                              for estimate in result.local_estimates)
        if marker_observed:
            goals, construction = fresh_marker_goals(result, projection.state,
                                                     marker_id=self.marker.edge_id)
            result.records['fresh_goal_construction'] = construction
        else:
            goals = (projection.goal_components if not predicted_goals else
                     retain_transition_goals(predicted_goals, result, projection) if transition else
                     retain_goal_components(predicted_goals, result.prior,
                                            projection.posterior, projection.state))
        internal = WeightedState(projection.state.component_id, 1., projection.state, 0.)
        stage = Stage('fused', (internal,), raw_stage.sensor_ids, tuple(goals))
        effective_raw = PosteriorComponent(internal.label, projection.posterior, 1.,
            lineage=tuple(item.label for item in result.frozen_body_posterior.components),
            log_weight=0.)
        frozen = replace(result.frozen_body_posterior, components=(effective_raw,))
        original_branch = native_branches[0]
        branch = FrozenBodyBranch(original_branch.label, 1., original_branch.reference_state,
                                  original_branch.prior, (effective_raw,), 0.)
        if transition:
            stages = result.stages[:-1] + (
                replace(raw_stage, label='fused_change_candidates'),
                replace(projected_input.stages[-1], label='fused_observation_candidates'), stage)
        else:
            stages = result.stages[:-1] + (
                replace(raw_stage, label='fused_observation_candidates'), stage)
        records = result.records
        if transition:
            records['bayes_factor_transport'] = projected_input.records['bayes_factor_transport']
        records['internal_projection'] = dict(projection.record,
            diagnostic_conditional_goal_state_ids=projection.record['goal_state_ids'],
            conditional_goal_diagnostics_used_for_task=not marker_observed and not predicted_goals,
            goal_state_ids=[item.state.state_id for item in goals],
            goal_labels=[item.label for item in goals],
            goal_log_weights=[component_log_weight(item) for item in goals],
            goal_construction=('fresh image-relative task composition; old-map posterior conditionals are projection diagnostics only'
                               if marker_observed else
                               'retained root-fixed task goal; update body-side cross only; raw conditional goals are projection diagnostics only'
                               if predicted_goals else projection.record['goal_construction']),
            goal_policy=('all native fresh image modes composed through the shared captured camera/body; capture the root-fixed switch mixture'
                         if marker_observed else
                         'root-fixed marker marginals and mixture masses; posterior-weighted keep-branch Schmidt cross and independent reset cross' if transition else
                         'root-fixed marker marginals and mixture masses; Schmidt body-side cross update only'),
            goal_retention_is_full_posterior_conditioning=False,
            goal_source_state_ids=[item.state.state_id for item in goals],
            goal_source_labels=[item.label for item in goals],
            goal_source_weights=[item.weight for item in goals],
            goal_source_log_weights=[component_log_weight(item) for item in goals],
            goal_parent_internal_state_id=projection.state.state_id,
            predicted_goal_source_state_ids=[item.state.state_id for item in predicted_goals])
        records['mixture_policy'] = (
            'one recursive joint internal moment approximation; all current observation/goal modes and log masses retained; no temporal Cartesian-product enumeration')
        records['stages'][-1]['label'] = 'fused_change_candidates' if transition else 'fused_observation_candidates'
        if transition:
            records['stages'].append(dict(label='fused_observation_candidates',
                sensor_ids=list(raw_stage.sensor_ids), components=[_joint_record(item,
                    self.paths, result.root_frame) for item in projected_input.stages[-1].components]))
        records['stages'].append(dict(label=stage.label, sensor_ids=list(stage.sensor_ids),
            components=[_joint_record(internal, self.paths, result.root_frame)],
            goal_components=[_joint_record(item, self.paths, result.root_frame) for item in goals]))
        records['frozen_internal_charts'] = [dict(prior_label=branch.label,
            reference_state_id=branch.reference_state.state_id,
            posterior_labels=[effective_raw.label],
            approximation='full joint moment projection after fusion; density ratio includes this declared approximation')]
        return replace(result, stages=stages, frozen_body_posterior=frozen,
                       frozen_body_branches=native_branches if transition else (branch,),
                       projected_body_numerator=branch if transition else None)

    def _record_fixed_goal_memory(self, result):
        """Capture a root-frame task law only on a valid marker observation.

        This record describes the exact marker variables used by both IK and
        display, followed by the known marker-to-switch action.  It is not a
        display-only replacement distribution.  During image loss, its source
        time, command, version, all mode parameters and all masses are retained
        independently of the current body's metadata.
        """
        estimates = tuple(estimate for estimate in result.local_estimates
            if estimate.posterior is not None and estimate.detector_result is not None)
        if estimates:
            goals = tuple(result.stages[-1].goal_components)
            offset = np.asarray(self.settings['control']['goal_offset'], dtype=float)
            orientation = np.asarray(self.settings['control']['goal_orientation_wxyz'], dtype=float)
            rotation = quat_to_rotmat(orientation)
            length = float(self.settings['display']['axis_length'])
            endpoints = np.vstack((np.zeros(3), length*np.eye(3)))
            components = []
            for component in goals:
                state = component.state
                marker_index = state.edge_ids.index(self.marker.edge_id)
                marker = state.transforms[marker_index]
                block = np.arange(6*marker_index, 6*marker_index+6)
                queries = []
                for index, endpoint in enumerate(endpoints):
                    local_point = offset+rotation @ endpoint
                    query = point_query(state, result.marker_path, local_point,
                        query_id='fixed-goal-capture:{}'.format(index), frame=result.root_frame)
                    queries.append(dict(local_point=local_point.tolist(),
                        mean=query.mean.tolist(), covariance=query.covariance.tolist(),
                        source_state_id=query.state_id))
                components.append(dict(label=component.label, weight=component.weight,
                    log_weight=component_log_weight(component), source_state_id=state.state_id,
                    source_stamp=state.stamp, source_command=list(state.command),
                    source_version=state.version,
                    root_marker_parameters=dict(edge_id=marker.edge_id,
                        translation=marker.translation.tolist(),
                        rotation_wxyz=marker.rotation_wxyz.tolist(),
                        bingham=None if marker.bingham is None else marker.bingham.tolist(),
                        joint_six_coordinate_covariance=state.covariance[np.ix_(block, block)].tolist()),
                    root_goal_point_queries=queries))
            source = goals[0].state
            self.fixed_goal_memory = dict(available=True, root_frame=result.root_frame,
                observation_stamp=source.stamp, observation_command=list(source.command),
                observation_version=source.version,
                observation_ids=list(dict.fromkeys(item.posterior.observation_id for item in estimates)),
                observation_sensor_ids=list(dict.fromkeys(item.posterior.sensor_id for item in estimates)),
                observation_source_state_ids=result.records['internal_projection']['raw_source_state_ids'],
                construction=result.records['fresh_goal_construction']['construction']+', then fixed known marker-to-switch action',
                construction_approximation='task-only independent fresh measurement cut; not an exact joint Bayesian posterior',
                image_relative_source=deepcopy(result.records['fresh_goal_construction']),
                marker_to_switch=dict(translation=offset.tolist(), rotation_wxyz=orientation.tolist()),
                goal_components=components, sampling_used=False,
                holding_policy=('fixed root marker marginal and all mixture masses; posterior-weighted continuation Schmidt cross and independent body-redraw cross' if self.surprise_enabled else
                    'fixed root marker marginal and all mixture masses; body-side Schmidt cross update without conditioning the recorded goal'))
            event = 'marker_observation_update'
        elif self.fixed_goal_memory is None:
            event = 'unobserved_prior'
        else:
            event = 'image_missing_hold'
        result.records['fixed_goal_memory'] = (dict(available=False,
            root_frame=result.root_frame, reason='no valid marker observation has been captured')
            if self.fixed_goal_memory is None else deepcopy(self.fixed_goal_memory))
        result.records['fixed_goal_memory_event'] = event
