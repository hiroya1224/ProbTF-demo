"""Select commands by the full deterministic Gaussian--Bingham overlap."""
from dataclasses import dataclass, field, replace
import xml.etree.ElementTree as ET
import json

import numpy as np
from scipy.optimize import minimize

from probtf.distributions import compose_with_deterministic_right
from probtf.geometry import DeterministicTransform, quat_to_rotmat
from probtf_switch_press_demo.body import clearance, shape_displacement
from probtf_switch_press_demo.model import Model
from probtf_switch_press_demo.objective import PoseOverlapObjective
from probtf_switch_press_demo.scene import apply, inverse, transform
from probtf_switch_press_demo.vision import ImageObservation


class GoalLikelihood:
    """One frozen observation-time target, shared by optimization and display."""
    def __init__(self, model, marker_record, observed_command, goal_offset, goal_orientation_wxyz):
        self.summary = model.target(observed_command, marker_record, goal_offset, goal_orientation_wxyz)
        self.objective = PoseOverlapObjective(self.summary, model.integration_steps)
        self.record = self.summary.to_record("switch_goal")
        self.marker_summary = model.marker(observed_command, marker_record)
        # Display the comparison law itself in marker coordinates. Rebuilding
        # an independently projected marker law would change the source law.
        attachment = DeterministicTransform(goal_offset, goal_orientation_wxyz)
        query = compose_with_deterministic_right(self.record, attachment.inverse(),
            child_frame_id="observed_ar_marker", edge_id="goal_comparison_marker_coordinate_query")
        context = json.loads(query.provenance.detail)
        context["coordinate_query"] = "goal_comparison_marker_coordinate_query"
        self.marker_record = replace(query, provenance=replace(query.provenance,
            method="goal_comparison_marker_coordinate_query",
            detail=json.dumps(context, sort_keys=True)))
        self.sensor_record = marker_record
        self.mean = self.summary.mean
        self.marker_modes = []
        for branch in self.marker_summary.branch_summaries:
            pose = np.eye(4)
            pose[:3, :3] = quat_to_rotmat(branch.orientation.mode_wxyz)
            pose[:3, 3] = branch.mean
            self.marker_modes.append(pose)


@dataclass
class Prediction:
    increment: np.ndarray
    command: np.ndarray
    hand: object
    score: float
    diagnostics: dict
    minimum_clearance: float = float("inf")
    ik: dict = field(default_factory=dict)
    records: tuple = ()
    rods: object = None

    @property
    def tips(self):
        return self.hand.mean[None, :]


class GeometryGuard:
    """Nominal geometry feasibility; no stiffness or simulated equilibrium.

    This is a declared geometric constraint on representatives of the input
    law, not a guarantee for the unbounded support of Gaussian candidates.
    The physical simulator separately checks actual motion for penetration.
    """
    def __init__(self, model):
        self.model = model

    def boxes(self, command):
        roots = self.model.nominal_roots(command)
        return [(link, roots[link] @ local, half)
                for link, _name, local, half in self.model.knowngeometry.bodies]

    def safe_motion(self, start, end, marker_modes, obstacles, settings):
        start, end = np.asarray(start), np.asarray(end)
        count = max(1, int(np.ceil(np.max(np.abs(end-start))/settings["joint_resolution"])))
        smallest = float("inf")

        def segment(c0, boxes0, c1, boxes1, depth=0):
            if shape_displacement(boxes0, boxes1) <= settings["spatial_resolution"]:
                yield boxes1
                return
            if depth >= 12:
                raise RuntimeError("nominal geometry path subdivision did not converge")
            middle = (c0+c1)/2
            boxes_middle = self.boxes(middle)
            yield from segment(c0, boxes0, middle, boxes_middle, depth+1)
            yield from segment(middle, boxes_middle, c1, boxes1, depth+1)

        previous, boxes0 = start, self.boxes(start)
        for boxes in (boxes0,):
            for marker in marker_modes:
                gap, _pair = clearance(boxes, marker, obstacles)
                smallest = min(smallest, gap)
                if gap < settings["clearance"]:
                    return False, smallest
        for fraction in np.linspace(0, 1, count+1)[1:]:
            current = start+fraction*(end-start)
            boxes1 = self.boxes(current)
            for boxes in segment(previous, boxes0, current, boxes1):
                for marker in marker_modes:
                    gap, _pair = clearance(boxes, marker, obstacles)
                    smallest = min(smallest, gap)
                    if gap < settings["clearance"]:
                        return False, smallest
            previous, boxes0 = current, boxes1
        return True, smallest


class Controller:
    """Only pixels, commands and directly configured transform laws enter IK."""
    def __init__(self, urdf, control, motion, marker, camera, rod_length, obstacles, seed):
        self.goal_offset = np.asarray(marker["goal_offset"])
        self.goal_orientation = np.asarray(marker["goal_orientation_wxyz"])
        settings = {key: marker[key] for key in
                    ("id", "size", "corner_sigma_px", "translation_prior_variance", "adaptive_covariance")}
        self.observation = ImageObservation(settings, camera)
        self.hand_camera = transform(camera["translation"], camera["rpy"])
        self.model = Model.from_config(urdf, control.get("transform_law_config"), rod_length,
                                      self.hand_camera, control.get("integration_steps", 480))
        self.geometry = GeometryGuard(self.model)
        limits = [joint.find("limit") for joint in ET.parse(str(urdf)).getroot().findall("joint")
                  if joint.get("type") == "revolute"]
        self.lower = np.array([float(limit.get("lower")) for limit in limits])
        self.upper = np.array([float(limit.get("upper")) for limit in limits])
        self.settings, self.motion, self.obstacles = control, motion, obstacles
        self.marker_size = float(marker["size"])
        self.camera_size = np.array([camera["width"], camera["height"]], dtype=float)
        self.focal_length = float(camera["focal_length"])

    def propose(self, command, goal):
        # Mean differential IK supplies a seed only. Selection uses all J terms.
        settings = self.settings
        eps, limit = settings["jacobian_step"], settings["max_joint_step"]
        mean, _covariance = self.model.position_moments(command)
        jacobian = np.column_stack([(self.model.position_moments(command+axis*eps)[0]-mean)/eps
                                    for axis in np.eye(6)])
        error = goal-mean
        error *= min(1., settings["proposal_distance"]/max(np.linalg.norm(error), 1e-12))
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T+settings["damping"]**2*np.eye(3), error)
        step *= min(1., limit/max(np.max(np.abs(step)), 1e-12))
        return [np.clip(value, self.lower-command, self.upper-command)
                for value in (np.zeros(6), step)]

    def choose(self, image, command, stamp, on_iteration=None):
        result, debug = self.observation.estimate(image, stamp)
        if result is None:
            return None, None, None, debug
        command = np.asarray(command, dtype=float).copy()
        goal = GoalLikelihood(self.model, result.record, command, self.goal_offset, self.goal_orientation)
        cache, trace = {}, []
        best_sent_score = float("-inf")
        source_mean = self.model.position_moments(command)[0]
        source_distance_squared = float(np.sum((source_mean-goal.mean)**2))
        # Shorten the trust box near the target so a new image arrives before
        # compliance-model error can exhaust the observed image margin.
        limit = min(self.settings["max_joint_step"],
                    max(.1*self.settings["max_joint_step"], .5*np.sqrt(source_distance_squared)))
        observed_camera = self.model.nominal_roots(command)["camera_optical_frame"]
        half = self.marker_size/2
        corners = np.array([[-half, half, 0], [half, half, 0],
                            [half, -half, 0], [-half, -half, 0]])
        observed_corners = []
        for branch in result.record.distribution.normalized_components():
            component = branch.component
            quaternion = component.orientation.reference_quaternion_wxyz
            pose = np.eye(4)
            pose[:3, :3] = quat_to_rotmat(quaternion)
            pose[:3, 3] = component.conditional_translation_mean(quaternion)
            observed_corners.append(apply(observed_camera @ pose, corners))

        source_view_bounds = []
        source_inverse = inverse(observed_camera)
        for branch in observed_corners:
            points = apply(source_inverse, branch)
            pixels = self.focal_length*points[:, :2]/points[:, 2, None]+self.camera_size/2
            normalized = pixels/self.camera_size
            source_view_bounds.append((np.minimum(.15, normalized+.02), np.minimum(.15, 1-normalized+.02)))

        def task_constraints(increment):
            # Explicit U_t conditions for this reaching task, using only the
            # frozen observed law and nominal representatives of the same model.
            q = command+np.asarray(increment)
            cached = cache.get(tuple(np.asarray(increment, dtype=float)))
            mean = cached.hand.mean if cached is not None else self.model.position_moments(q)[0]
            progress = (.95**2)*source_distance_squared-float(np.sum((mean-goal.mean)**2))
            camera_inverse = inverse(self.model.nominal_roots(q)["camera_optical_frame"])
            margins = [progress]
            for branch, (lower_margin, upper_margin) in zip(observed_corners, source_view_bounds):
                points = apply(camera_inverse, branch)
                depth = points[:, 2]
                pixels = self.focal_length*points[:, :2]/np.maximum(depth[:, None], 1e-9)+self.camera_size/2
                # Keep 15% border margin where available. If the new real
                # observation is already outside it, require two percentage
                # points of recovery (up to 15%) before approaching further.
                margins.extend((depth-.02).tolist())
                margins.extend((pixels/self.camera_size-lower_margin).ravel().tolist())
                margins.extend((1-upper_margin-pixels/self.camera_size).ravel().tolist())
            return np.asarray(margins)

        def evaluate(increment):
            key = tuple(np.asarray(increment, dtype=float))
            if key not in cache:
                q = command+np.asarray(increment)
                hand = self.model.hand(q, stamp)
                score, diagnostics = goal.objective.evaluate(hand)
                cache[key] = Prediction(np.array(increment, copy=True), q, hand, score, diagnostics)
            return cache[key]

        proposals = [evaluate(np.clip(value, -limit, limit)) for value in self.propose(command, goal.mean)]
        seed = max(proposals, key=lambda item: item.score)
        scale = max(1., abs(seed.score))

        def feasible(item):
            if np.max(np.abs(item.increment)) > limit+1e-10 or np.min(task_constraints(item.increment)) < -1e-8:
                return False
            safe, gap = self.geometry.safe_motion(command, item.command, goal.marker_modes,
                                                  self.obstacles, self.motion)
            item.minimum_clearance = gap
            return safe

        def publish(item):
            if not item.records:
                item.records = self.model.display_records(item.command, stamp, hand_summary=item.hand)
                roots = self.model.nominal_roots(item.command)
                item.rods = np.array([[roots["tool0"][:3, 3], item.hand.mean]])
            if on_iteration is not None:
                on_iteration(item, goal, result, debug, len(trace)-1)

        def iteration(increment):
            nonlocal best_sent_score
            item = evaluate(increment)
            proposals.append(item)
            trace.append(dict(iteration=len(trace), increment=item.increment.copy(),
                              log_expected_density=item.score, mean=item.hand.mean.copy(),
                              covariance=item.hand.covariance.copy(), diagnostics=item.diagnostics))
            if item.score > best_sent_score and feasible(item):
                best_sent_score = item.score
                publish(item)

        iteration(seed.increment)
        bounds = list(zip(np.maximum(self.lower-command, -limit), np.minimum(self.upper-command, limit)))
        solution = minimize(lambda increment: -evaluate(increment).score/scale, seed.increment,
            method="SLSQP", bounds=bounds, callback=iteration,
            constraints=[dict(type="ineq", fun=task_constraints)],
            options=dict(maxiter=self.settings["ik_max_iterations"], ftol=1e-8,
                         eps=self.settings["jacobian_step"]))
        proposals.extend(evaluate(solution.x*fraction) for fraction in (1., .5, .25))
        info = dict(method="SLSQP Gaussian-Bingham expected-likelihood IK",
            converged=bool(solution.success), termination=str(solution.message),
            iterations=int(solution.nit), evaluations=int(solution.nfev),
            initial_log_expected_density=seed.score,
            optimized_log_expected_density=-float(solution.fun)*scale,
            optimized_increment=solution.x.copy(), trace=trace,
            geometry_constraint="nominal representative path; not a Gaussian-support certificate",
            task_constraints="at least 5% Hand mean distance reduction and observed-marker visibility recovery",
            source_mean_distance_squared=source_distance_squared, joint_trust_limit=limit)
        for item in sorted(proposals, key=lambda item: item.score, reverse=True):
            if feasible(item):
                item.ik = info
                if not item.records:
                    publish(item)
                return item, goal, result, debug
        goal.failure = dict(source_constraints=task_constraints(np.zeros(6)),
            candidate_constraints=[dict(increment=item.increment, minimum=float(np.min(task_constraints(item.increment))),
                                         clearance=(item.minimum_clearance if np.isfinite(item.minimum_clearance) else None)) for item in proposals])
        return None, goal, result, debug


class CommandServo:
    """Follow absolute command targets without resetting velocity at replans."""
    def __init__(self, settings):
        self.settings = settings
        self.velocity = np.zeros(6)

    def step(self, command, target, dt):
        error = np.asarray(target) - command
        speed = self.settings["max_joint_velocity"]
        acceleration = self.settings["max_joint_acceleration"]
        desired = error / self.settings["tracking_time"]
        desired *= min(1., speed / max(np.max(np.abs(desired)), 1e-12))
        change = desired - self.velocity
        change *= min(1., acceleration * dt / max(np.max(np.abs(change)), 1e-12))
        self.velocity += change
        return np.asarray(command) + self.velocity * dt
