"""Retained transform-candidate inference on the moving switch scene.

The directly supplied family initializes the body and defines the explicit
body-redraw change kernel. Each new observation compares that kernel with
the retained joint proxy pushed by its declared command action. Current
observation/change modes remain explicit goal components. No physical truth
or invented process covariance enters inference.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
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
        if not np.isfinite(self.change_probability) or not 0. < self.change_probability < 1.:
            raise ValueError('Bayes-factor change probability must be strictly between zero and one.')
        self.last_key = None
        self.last_result = None
        self.last_observation_signature = None
        self.components = ()
        self.goal_components = ()
        self.fixed_goal_memory = None
        self.last_observations = ()

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
        key = (int(frame_index), float(sensor_inputs.stamp), tuple(sensor_inputs.command))
        signature = hashlib.sha256(np.asarray(sensor_inputs.image).tobytes()
            + np.asarray(sensor_inputs.gravity_in_sensor).tobytes()
            + repr((sensor_inputs.imu_target_link, sensor_inputs.imu_sigma_rad)).encode()
            + (b'' if command_map is None else json.dumps(command_map.record(),
                sort_keys=True, separators=(',', ':')).encode())).hexdigest()
        if key == self.last_key:
            if signature != self.last_observation_signature:
                raise ValueError("A repeated physical tick contains different observation data.")
            return self.last_result
        if self.last_key is not None and int(frame_index) <= self.last_key[0]:
            raise ValueError("Frame indices must increase; construct a fresh inference on replay.")
        if self.last_key is not None and float(sensor_inputs.stamp) <= self.last_key[1]:
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
        for template in self.templates:
            observation = deepcopy(template)
            if observation["sensor"] == "imu":
                observation["sensor_id"] = "imu_" + sensor_inputs.imu_target_link
            observation["observation_id"] = "{}:frame:{:05d}".format(
                observation["sensor_id"], int(frame_index))
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
            reason='disabled' if not self.surprise_enabled else 'no valid new observations',
            log_bayes_factor=None, posterior_change_probability=None,
            log_keep_evidence=None, log_reset_evidence=None,
            monte_carlo_used=False, physical_truth_received=False)
        if self.surprise_enabled and observations:
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
                         prior_components=priors, paths=self.paths)
        result.records['bayes_factor_surprise'] = surprise
        result = self._project_internal_update(result, predicted_goals)
        surprise['internal_state_id'] = result.stages[-1].components[0].state.state_id
        surprise['goal_source_state_ids'] = [item.state.state_id
                                            for item in result.stages[-1].goal_components]
        self._record_fixed_goal_memory(result)
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
