"""Retained transform-candidate inference on the moving switch scene.

The direct independent family initializes the body once. Every later prior
is the full preceding posterior, pushed by its declared command action. No
physical pose, stiffness or invented process covariance enters this update.
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

from probtf.geometry import axis_angle_to_quat, quat_right_matrix, rotmat_to_quat
from probtf.spherical_law.first_order import BinghamTransform, JointTransformState
from symaware_grasp.probtf_config import bingham_parameter_from_mode

from .model import WeightedState, component_log_weight, run
from .moving_control import command_state


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
        self.last_key = None
        self.last_result = None
        self.last_observation_signature = None
        self.components = ()
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
            "prior": {"independence": "initial product over supplied local transform candidates; all posterior components and cross blocks retained between updates",
                      "body": [{"edge_id": name} for name in self.body_ids],
                      "marker": {"status": marker_data["status"]}},
            "known_geometry": {"source": "fixed attachment geometry parsed from shared URDF; masses and physical poses excluded"},
            "camera": settings["camera"], "marker": settings["marker"],
            "display": settings["display"],
        }

    def prior_at(self, command, stamp, frame_index):
        """Query the directly supplied initial family, with no pose truth.

        Sequential inference uses this only at initialization. It never uses
        this method to replace a body posterior on later ticks.
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
        result = run(self.common, tuple(observations), sensor_inputs.image,
                     prior_components=priors, paths=self.paths)
        result.records["prediction"] = prediction
        result.records["prediction_command_correction"] = (
            None if command_map is None else command_map.record())
        result.records["prediction_command_action"] = (
            None if command_map is None else command_map.selection_record(sensor_inputs.command))
        result.records["moving_input_family"] = {
            "law_id": self.law["law_id"], "revision": self.law["revision"],
            "command_family": ("configured Bingham right action; translation law unchanged"
                if command_map is None else "same posterior right action by the previous cycle's frozen estimated-K equilibrium increment; translation law unchanged"),
            "temporal_policy": "initial independent family once; retain every posterior component and push full joint law through the declared right action at the next command",
            "initialization": not bool(self.components),
            "frame_index": int(frame_index), "time": float(sensor_inputs.time),
            "physical_truth_received": False,
        }
        self.last_key, self.last_result = key, result
        self.last_observation_signature = signature
        self.components = tuple(result.stages[-1].components)
        self.last_observations = tuple(observations)
        return result
