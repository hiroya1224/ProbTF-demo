"""Expected goal density under native shared body laws, with image feedback."""
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

from probtf.distributions import (
    DistributionStatus, compose_with_deterministic_right,
)
from probtf.geometry import DeterministicTransform, quat_to_rotmat
from probtf.kernels import ForwardEdgeKernel, KernelEvaluator, KernelRepresentation
from probtf.joint import PointQuadrature
from probtf.candidates import load_initial_transform_law
from probtf.probability.point_density import DensityIntegrationError

from probtf_switch_press_demo.kinematics import CandidateKinematics
from probtf_switch_press_demo.beliefs import BODY_FRAMES, JointMeasure, finite_pose_record, left_compose_record
from probtf_switch_press_demo.scene import transform
from probtf_switch_press_demo.vision import ImageObservation


class GoalLikelihood:
    """Expected-density kernel; preserve Bingham and conditional translation."""
    def __init__(self, marker_record, hand_camera, goal_offset, order=9):
        camera_goal = compose_with_deterministic_right(
            marker_record, DeterministicTransform(np.asarray(goal_offset), [1, 0, 0, 0]),
            child_frame_id="switch_goal", edge_id="observed_switch_goal")
        self.record = left_compose_record(camera_goal, hand_camera, "observation_hand")
        self.kernel = ForwardEdgeKernel(self.record)
        self.evaluator = KernelEvaluator(point_density_order=order)
        result = self.evaluator.apply_to_point(self.kernel, np.zeros(3), KernelRepresentation.NUMERICAL_LAW)
        if result.status is not DistributionStatus.OK or not hasattr(result.value, "log_density_and_gradient"):
            raise DensityIntegrationError("goal density query unavailable: {}".format(result.diagnostics))
        self.density = result.value
        self.metadata = self.density.metadata
        # The proposal uses exact component point moments, not a sampled mode.
        moments = self.evaluator.apply_to_point(self.kernel, np.zeros(3), KernelRepresentation.MOMENTS)
        if moments.status is not DistributionStatus.OK:
            raise DensityIntegrationError("goal point moments unavailable")
        self.mean = moments.value.mean
        self.marker_modes = []
        for item in marker_record.distribution.normalized_components():
            component = item.component
            quaternion = component.orientation.reference_quaternion_wxyz
            pose = np.eye(4)
            pose[:3, :3] = quat_to_rotmat(quaternion)
            pose[:3, 3] = component.conditional_translation_mean(quaternion)
            self.marker_modes.append(hand_camera @ pose)

    def value_and_tip_gradient(self, hand_record):
        if hand_record.parent_frame_id != self.record.parent_frame_id:
            raise ValueError("hand and goal must use the same observation frame")
        components = hand_record.distribution.normalized_components()
        if any(not item.component.is_deterministic for item in components):
            raise ValueError("expected density requires the original shared-law query, not a marginal pose fit")
        tips = np.array([item.component.translation.mean_at_reference for item in components])
        weights = np.array([item.weight for item in components])
        return self._value_and_tip_gradient(tips, weights)

    def value_and_prediction_gradient(self, prediction):
        cached = getattr(prediction, "_expected_density_query", None)
        if cached is not None and cached[0] is self:
            return cached[1]
        expression = prediction.joint_after.expression
        if expression is None:
            value = self._value_and_tip_gradient(prediction.tips, prediction.joint_after.weights)
        else:
            kernel = expression.relative(prediction.joint_before.expression, -1)
            representation = (KernelRepresentation.SAMPLES if prediction.joint_after.metadata.get("integration_method") == "monte_carlo"
                              else KernelRepresentation.QUADRATURE)
            query = self.evaluator.apply_to_point(kernel, np.zeros(3), representation)
            if query.status is not DistributionStatus.OK:
                raise DensityIntegrationError("shared hand law query unavailable: {}".format(query.diagnostics))
            value = query.value.expected_log_density(self.density)
        prediction._expected_density_query = (self, value, self.density.last_diagnostics)
        return value

    def expected_density_diagnostics(self, prediction):
        metadata = dict(prediction.joint_after.metadata)
        if metadata.get("integration_method") != "monte_carlo":
            return dict(method="exact_finite_input_law", node_count=len(prediction.tips),
                        relative_standard_error=0., source_law_id=metadata.get("source_law_id"))
        values = self.density.log_density(prediction.tips)
        scaled = np.exp(values - np.max(values))
        mean = np.mean(scaled)
        relative = (float(np.std(scaled, ddof=1) / np.sqrt(len(scaled)) / mean)
                    if len(scaled) > 1 and mean > 0 else float("inf"))
        return dict(method="monte_carlo", sample_count=len(values),
                    relative_standard_error=relative, error_bound=None,
                    source_law_id=metadata.get("source_law_id"),
                    source_revision=metadata.get("source_revision"),
                    sampling_method=metadata.get("integration_method"), sampling_seed=metadata.get("seed"))

    def _value_and_tip_gradient(self, tips, weights):
        return PointQuadrature(np.asarray(tips), np.asarray(weights), {}).expected_log_density(self.density)

    def log_expected_density(self, hand_record):
        return self.value_and_tip_gradient(hand_record)[0]


@dataclass
class Prediction:
    increment: np.ndarray
    joint_before: JointMeasure
    joint_after: JointMeasure
    poses: tuple
    score: float
    stamp: float = 0.
    minimum_clearance: float = float("inf")
    ik: dict = field(default_factory=dict)
    _records: dict = field(default_factory=dict, repr=False)

    @cached_property
    def relative_transforms(self):
        return self.joint_after.relative_transforms(self.joint_before)

    def link_record(self, index):
        index %= len(BODY_FRAMES)
        if index not in self._records:
            self._records[index] = finite_pose_record("observation_hand", "predicted/" + BODY_FRAMES[index],
                self.stamp, self.relative_transforms[:, index], self.joint_after.body_ids, self.joint_after.weights,
                metadata=dict(self.joint_after.metadata, reference_query_id=self.joint_before.metadata.get("query_id")))
        return self._records[index]

    @property
    def link_records(self):
        return tuple(self.link_record(index) for index in range(len(BODY_FRAMES)))

    @cached_property
    def root_records(self):
        return self.joint_after.root_records(self.stamp)

    @property
    def hand_record(self):
        return self.link_record(-1)

    @property
    def tips(self):
        return self.relative_transforms[:, -1, :3, 3]

    @property
    def rods(self):
        return self.relative_transforms[:, -2:, :3, 3]


class Controller:
    """No access to the plant, world marker pose, true joints, or display TF."""
    def __init__(self, urdf, control, motion, marker, camera, rod_length, obstacles, seed):
        # This argument contains known marker geometry and the image prior only.
        self.goal_offset = np.array(marker["goal_offset"])
        observation_settings = {key: marker[key] for key in
                                ("id", "size", "corner_sigma_px", "translation_prior_variance", "adaptive_covariance")}
        self.observation = ImageObservation(observation_settings, camera)
        self.hand_camera = transform(camera["translation"], camera["rpy"])
        path = Path(control["initial_transform_law"])
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        law = load_initial_transform_law(path)
        if law.reference_command is None:
            raise ValueError("initial transform law must declare its reference command")
        self.candidates = CandidateKinematics(urdf, law, rod_length, law.reference_command)
        self.settings, self.motion, self.obstacles = control, motion, obstacles

    def propose(self, command, goal):
        settings = self.settings
        limit = settings["max_joint_step"]
        current = self.candidates.predict(command, np.zeros(6))
        jacobians = self.candidates.tip_jacobians(command, current)
        weights = current.joint_after.weights
        jacobian = np.einsum("s,sij->ij", weights, jacobians)
        error = goal - weights @ current.tips
        error *= min(1.0, settings["proposal_distance"] / max(np.linalg.norm(error), 1e-12))
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + settings["damping"] ** 2 * np.eye(3), error)
        step *= min(1.0, limit / max(np.max(np.abs(step)), 1e-12))
        return [value for value in (np.zeros(6), step)
                if np.all(command + value >= self.candidates.lower)
                and np.all(command + value <= self.candidates.upper)]

    def choose(self, image, command, stamp, on_iteration=None):
        result, debug = self.observation.estimate(image, stamp)
        if result is None:
            return None, None, None, debug
        goal = GoalLikelihood(result.record, self.hand_camera, self.goal_offset,
                              self.settings["orientation_quadrature_order"])
        proposals = []
        for increment in self.propose(np.asarray(command), goal.mean):
            try:
                prediction = self.candidates.predict(command, increment, stamp)
                prediction.score = goal.value_and_prediction_gradient(prediction)[0]
                if np.isfinite(prediction.score):
                    proposals.append(prediction)
            except RuntimeError:
                continue
        if not proposals:
            return None, goal, result, debug

        # Solve a bounded nonlinear IK step with the same expected likelihood
        # used for the final decision. Differential IK only supplies its seed.
        command = np.asarray(command, dtype=float)
        seed = max(proposals, key=lambda item: item.score)
        candidates = self.candidates
        objective_scale = max(1.0, abs(seed.score))
        cache = {tuple(item.increment): item for item in proposals}
        feasibility = {}
        precision = {}
        trace = []

        def safe(item):
            key = tuple(item.increment)
            if key not in precision:
                precision[key] = goal.expected_density_diagnostics(item)
            estimate = precision[key]
            if estimate.get("relative_standard_error", 0.) > self.settings["max_monte_carlo_relative_error"]:
                raise DensityIntegrationError("joint-sample expected likelihood does not meet the requested precision", estimate)
            if key not in feasibility:
                feasibility[key] = candidates.safe_path(command, item, goal.marker_modes,
                                                        self.obstacles, self.motion)
            return feasibility[key]

        def evaluate(increment):
            key = tuple(np.asarray(increment, dtype=float))
            if key not in cache:
                try:
                    item = candidates.predict(command, np.asarray(increment), stamp)
                    item.score = goal.value_and_prediction_gradient(item)[0]
                    cache[key] = item if np.isfinite(item.score) else None
                except RuntimeError:
                    cache[key] = None
            return cache[key]

        def objective(increment):
            item = evaluate(increment)
            return 1e100 if item is None else -item.score / objective_scale

        def gradient(increment):
            item = evaluate(increment)
            if item is None:
                return np.zeros(6)
            _score, tip_gradient = goal.value_and_prediction_gradient(item)
            jacobians = candidates.tip_jacobians(command, item)
            return -np.einsum("ij,ijk->k", tip_gradient, jacobians) / objective_scale

        def iteration(increment):
            item = evaluate(increment)
            if item is None:
                return
            trace.append(dict(iteration=len(trace), increment=item.increment.copy(),
                              log_expected_density=item.score, predicted_tips=item.tips.copy()))
            if on_iteration is not None and safe(item):
                item.ik = dict(expected_density_diagnostics=precision[tuple(item.increment)],
                               expected_density_query=dict(item.joint_after.metadata),
                               goal_density_diagnostics=item._expected_density_query[2])
                on_iteration(item, goal, result, debug, len(trace) - 1)

        iteration(seed.increment)
        limit = self.settings["max_joint_step"]
        bounds = list(zip(np.maximum(candidates.lower - command, -limit),
                          np.minimum(candidates.upper - command, limit)))
        solution = minimize(objective, seed.increment, jac=gradient, method="L-BFGS-B", bounds=bounds,
            callback=iteration, options=dict(maxiter=self.settings["ik_max_iterations"],
                ftol=1e-10, gtol=1e-6, maxls=20))
        for fraction in (1.0, 0.5, 0.25):
            item = evaluate(solution.x * fraction)
            if item is not None:
                proposals.append(item)
        info = dict(method="L-BFGS-B expected-likelihood IK", converged=bool(solution.success),
                    termination=str(solution.message), iterations=int(solution.nit),
                    evaluations=int(solution.nfev), initial_log_expected_density=seed.score,
                    optimized_log_expected_density=-float(solution.fun) * objective_scale,
                    optimized_increment=solution.x.copy(), trace=trace,
                    integration=dict(body=candidates.law.descriptor(), goal=goal.metadata,
                        collision_condition="all evaluated joint candidates; sampled continuous laws have no full-support certificate"))
        for prediction in sorted(proposals, key=lambda item: item.score, reverse=True):
            if not safe(prediction):
                continue
            estimate = precision[tuple(prediction.increment)]
            prediction.ik = dict(info, expected_density_diagnostics=estimate,
                expected_density_query=dict(prediction.joint_after.metadata),
                integration_points=prediction.tips, integration_weights=prediction.joint_after.weights,
                goal_density_diagnostics=prediction._expected_density_query[2])
            return prediction, goal, result, debug
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
