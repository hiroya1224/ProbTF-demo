"""Actuate the configured initial joint law of all local link transforms."""
from hashlib import sha256
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from probtf.joint import JointTransformLaw
from probtf_switch_press_demo.beliefs import BODY_FRAMES, JointMeasure
from probtf_switch_press_demo.body import clearance, shape_displacement
from probtf_switch_press_demo.scene import inverse
from symaware_grasp.grasp_collision import BottleCollisionModel


class CandidateKinematics:
    """Push the initial SE(3) tuple law through the known actuator motion.

    A candidate row is the complete initial body, not six independently chosen
    poses. Neither the source transforms nor their dependence are replaced by
    an angular error model. URDF axes describe only the commanded right action.
    """
    def __init__(self, urdf, source, rod_length, reference_command):
        self.urdf = Path(urdf)
        self.source_law = source
        if tuple(source.frame_names) != BODY_FRAMES:
            raise ValueError("initial transform law must cover the complete ordered body frames")
        records = getattr(source, "edge_records", ())
        if records and records[0].parent_frame_id != "base_link":
            raise ValueError("initial transform law must be rooted at base_link")
        self.reference_command = self._command(reference_command).copy()
        self.reference_command.setflags(write=False)
        source_reference = getattr(source, "reference_command", ())
        if source_reference and not np.array_equal(source_reference, self.reference_command):
            raise ValueError("configured transform law and actuator reference commands must agree")
        self.tip = np.array([float(rod_length), 0., 0.])
        if not np.isfinite(self.tip[0]) or self.tip[0] <= 0:
            raise ValueError("rod length must be positive and finite")
        root = ET.parse(str(self.urdf)).getroot()
        self.axes, lower, upper = [], [], []
        for index, frame in enumerate(BODY_FRAMES[:6]):
            joint = root.find("./joint[@name='joint_{}']".format(index+1))
            if (joint is None or joint.get("type") != "revolute"
                    or joint.find("child").get("link") != frame):
                raise ValueError("known actuator joints must match the six configured link frames")
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) == 0:
                raise ValueError("actuator axis must contain three finite nonzero coordinates")
            self.axes.append(axis / np.linalg.norm(axis))
            limit = joint.find("limit")
            lower.append(float(limit.get("lower")))
            upper.append(float(limit.get("upper")))
        self.axes = np.asarray(self.axes)
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        if (not np.all(np.isfinite(self.lower)) or not np.all(np.isfinite(self.upper))
                or np.any(self.lower >= self.upper)):
            raise ValueError("actuator command limits must be finite and ordered")
        self._within_limits(self.reference_command)
        initial, weights = source.quadrature()
        self.ids = tuple(source.node_ids)
        self.weights = np.asarray(weights)
        if np.shape(initial) != (len(self.ids), len(BODY_FRAMES), 4, 4):
            raise ValueError("initial candidate rows must contain the complete transform tuple")
        self.geometry = BottleCollisionModel(self.urdf)
        # Collision geometry follows its source link; the remaining URDF
        # attachments (camera/rod body) have known fixed maps from those links.
        attachments = {"base_link": (-1, np.eye(4))}
        attachments.update((frame, (index, np.eye(4))) for index, frame in enumerate(BODY_FRAMES))
        pending = [joint for joint in self.geometry.joints if joint[1] == "fixed"]
        while pending:
            progressed = False
            for joint in pending[:]:
                _name, _kind, parent, child, origin, _axis = joint
                if child in attachments:
                    pending.remove(joint)
                    progressed = True
                elif parent in attachments:
                    anchor, offset = attachments[parent]
                    attachments[child] = anchor, offset @ origin
                    pending.remove(joint)
                    progressed = True
            if not progressed:
                raise ValueError("fixed collision attachments cannot be resolved from candidate link frames")
        self._collision_bodies = []
        for link, _name, local, half in self.geometry.bodies:
            if link not in attachments:
                raise ValueError("collision body has no configured link transform: " + link)
            index, offset = attachments[link]
            self._collision_bodies.append((link, index, offset @ local, half))
        fixed = tuple(getattr(source, "deterministic_local_indices", ()))
        model_id = "configured_transform_right_actuation_v2_" + sha256(
            self.urdf.read_bytes() + self.reference_command.tobytes()).hexdigest()
        self.law = JointTransformLaw(source, self.generate, BODY_FRAMES, model_id,
            fixed_local_indices=fixed, relative_generator=self.relative_generate,
            relative_generator_id="known_joint_right_action_prefix_cancellation_v1")
        self.current_measure = None
        self.command = None

    @staticmethod
    def _command(command):
        value = np.asarray(command, dtype=float)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("command must contain six finite actuator coordinates")
        return value

    def _within_limits(self, command):
        if np.any(command < self.lower) or np.any(command > self.upper):
            raise ValueError("actuator command exceeds known joint limits")

    def generate(self, initial_local_transforms, command):
        """Apply T_i(u)=T_i(ref) R_axis_i(u_i-u_ref_i) to every shared row."""
        command = self._command(command)
        self._within_limits(command)
        local = np.array(initial_local_transforms, dtype=float, copy=True)
        if local.ndim != 4 or local.shape[1:] != (len(BODY_FRAMES), 4, 4):
            raise ValueError("actuation requires complete initial local-transform rows")
        rotations = Rotation.from_rotvec(self.axes * (command-self.reference_command)[:, None]).as_matrix()
        local[:, :6, :3, :3] = local[:, :6, :3, :3] @ rotations
        return local

    def current(self, command):
        command = self._command(command)
        if self.command is None or not np.array_equal(command, self.command):
            self.current_measure = JointMeasure.from_query(self.law.at(command).evaluate())
            self.command = command.copy()
        return self.current_measure

    def relative_generate(self, initial_local_transforms, before_command, after_command,
                          reference_index, index=None):
        """Cancel the shared first initial transform before multiplying rows.

        Writing a root pose as L0 D0(u) P(u) gives
        P(before)^-1 D0(after-before) P(after). The shared L0 disappears
        algebraically, so equal relative candidates remain bitwise equal.
        """
        before_command = self._command(before_command)
        after_command = self._command(after_command)
        before = self.generate(initial_local_transforms, before_command)
        after = self.generate(initial_local_transforms, after_command)
        reference_index %= len(BODY_FRAMES)
        tail = np.broadcast_to(np.eye(4), (len(before), 4, 4)).copy()
        for edge in range(1, reference_index+1):
            tail = tail @ before[:, edge]
        first_action = np.eye(4)
        first_action[:3, :3] = Rotation.from_rotvec(
            self.axes[0]*(after_command[0]-before_command[0])).as_matrix()
        pose = np.linalg.inv(tail) @ first_action
        relative = np.empty_like(after)
        for edge in range(len(BODY_FRAMES)):
            if edge:
                pose = pose @ after[:, edge]
            relative[:, edge] = pose
        if np.array_equal(before_command, after_command):
            relative[:, reference_index] = np.eye(4)
            if reference_index+1 < len(BODY_FRAMES):
                relative[:, reference_index+1] = after[:, reference_index+1]
        return relative if index is None else relative[:, index % len(BODY_FRAMES)]

    def predict(self, command, increment, stamp=0.):
        from probtf_switch_press_demo.control import Prediction
        increment = self._command(increment)
        before = self.current(command)
        after = (before if not np.any(increment) else
                 JointMeasure.from_query(self.law.at(np.asarray(command)+increment).evaluate()))
        return Prediction(increment.copy(), before, after, tuple(after.local_transforms), float("-inf"), stamp)

    def tip_jacobians(self, command, prediction):
        """Exact actuator derivatives for each propagated initial candidate."""
        before = self.current(command)
        root = prediction.joint_after.root_transforms
        axes = np.einsum("snij,nj->sni", root[:, :6, :3, :3], self.axes)
        displacements = root[:, -1, None, :3, 3] - root[:, :6, :3, 3]
        world = np.cross(axes, displacements).transpose(0, 2, 1)
        return before.root_transforms[:, -2, :3, :3].transpose(0, 2, 1) @ world

    def boxes(self, command_or_measure, reference=None):
        measure = (command_or_measure if hasattr(command_or_measure, "root_transforms") else
                   JointMeasure.from_query(self.law.at(command_or_measure).evaluate()))
        roots = measure.root_transforms
        reference = (np.broadcast_to(np.eye(4), (len(roots), 4, 4)) if reference is None else
                     np.broadcast_to(reference, (len(roots), 4, 4)))
        bodies = []
        for sample, row in enumerate(roots):
            if self.weights[sample] == 0.:
                continue
            for link, index, local, half in self._collision_bodies:
                pose = local if index == -1 else row[index] @ local
                bodies.append((self.ids[sample] + ":" + link, reference[sample] @ pose, half))
        return bodies

    def safe_path(self, command, prediction, marker_modes, obstacles, settings):
        safe, smallest = self.safe_motion(command, np.asarray(command)+prediction.increment,
                                         command, marker_modes, obstacles, settings)
        prediction.minimum_clearance = smallest
        return safe

    def safe_motion(self, start, end, observed_command, marker_modes, obstacles, settings):
        start, end = self._command(start), self._command(end)
        if np.any(np.minimum(start, end) < self.lower) or np.any(np.maximum(start, end) > self.upper):
            return False, float("-inf")
        observed = self.current(observed_command)
        references = np.array([inverse(pose) for pose in observed.root_transforms[:, -2]])

        def shapes(command):
            return self.boxes(JointMeasure.from_query(self.law.at(command).evaluate()), references)

        def segment(c0, boxes0, c1, boxes1, depth=0):
            if shape_displacement(boxes0, boxes1) <= settings["spatial_resolution"]:
                yield c1, boxes1
                return
            if depth >= 12:
                raise RuntimeError("candidate motion subdivision did not converge")
            middle = (c0+c1)/2
            boxes_middle = shapes(middle)
            yield from segment(c0, boxes0, middle, boxes_middle, depth+1)
            yield from segment(middle, boxes_middle, c1, boxes1, depth+1)

        count = max(1, int(np.ceil(np.max(np.abs(end-start))/settings["joint_resolution"])))
        previous, previous_boxes = start, shapes(start)

        def path():
            nonlocal previous, previous_boxes
            yield previous, previous_boxes
            for fraction in np.linspace(0., 1., count+1)[1:]:
                command = start+fraction*(end-start)
                boxes = shapes(command)
                yield from segment(previous, previous_boxes, command, boxes)
                previous, previous_boxes = command, boxes

        smallest = float("inf")
        for _command, boxes in path():
            for marker in marker_modes:
                gap, _pair = clearance(boxes, marker, obstacles)
                smallest = min(smallest, gap)
                if gap < settings["clearance"]:
                    return False, smallest
        return True, smallest
