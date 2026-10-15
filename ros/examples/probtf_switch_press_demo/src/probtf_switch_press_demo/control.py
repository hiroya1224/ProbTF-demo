"""Expected goal density at finite body predictions, with image-only feedback."""
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from probtf.distributions import DistributionStatus, compose_with_deterministic_right
from probtf.geometry import DeterministicTransform, quat_to_rotmat
from probtf.probability.sampling import sample_bingham_orientation

from probtf_switch_press_demo.body import EquilibriumBody, clearance, motion_samples
from probtf_switch_press_demo.scene import apply, inverse, transform
from probtf_switch_press_demo.vision import ImageObservation


class GoalLikelihood:
    """Rao-Blackwellized orientation integration, keeping residual Gaussians exact."""
    def __init__(self, marker_record, hand_camera, goal_offset, sample_count, rng):
        self.record = compose_with_deterministic_right(
            marker_record, DeterministicTransform(np.asarray(goal_offset), [1, 0, 0, 0]),
            child_frame_id="switch_goal", edge_id="observed_switch_goal")
        normalized = self.record.distribution.normalize_weights()
        if normalized.status is not DistributionStatus.OK or sample_count < 1:
            raise ValueError("invalid goal mixture or sample count")
        self.terms = []
        self.mean = np.zeros(3)
        self.marker_modes = []
        rotation = hand_camera[:3, :3]
        for item in normalized.components:
            component = item.component
            quaternions = sample_bingham_orientation(component.orientation, sample_count, rng)
            means = apply(hand_camera, np.array([component.conditional_translation_mean(q) for q in quaternions]))
            covariance = rotation @ component.translation.residual_covariance @ rotation.T
            cholesky = np.linalg.cholesky(covariance)
            if not np.all(np.isfinite(cholesky)) or np.any(np.diag(cholesky) <= 0):
                raise ValueError("invalid conditional goal covariance; hold and reobserve")
            log_normalizer = 1.5 * np.log(2 * np.pi) + np.log(np.diag(cholesky)).sum()
            self.terms.append((float(item.weight), means, cholesky, log_normalizer))
            self.mean += item.weight * means.mean(axis=0)
        for item in marker_record.distribution.normalized_components():
            component = item.component
            quaternion = component.orientation.reference_quaternion_wxyz
            pose = np.eye(4)
            pose[:3, :3] = quat_to_rotmat(quaternion)
            pose[:3, 3] = component.conditional_translation_mean(quaternion)
            self.marker_modes.append(hand_camera @ pose)

    def value_and_tip_gradient(self, tips, body_weights):
        tips = np.asarray(tips, dtype=float)
        weights = np.asarray(body_weights, dtype=float)
        values, precision_differences = [], []
        for weight, means, cholesky, normalizer in self.terms:
            differences = tips[:, None, :] - means[None, :, :]
            whitened = solve_triangular(cholesky, differences.reshape(-1, 3).T, lower=True).T
            logpdf = (-0.5 * np.sum(whitened ** 2, axis=1) - normalizer).reshape(len(tips), len(means))
            values.append((logpdf + np.log(weights)[:, None] + np.log(weight) - np.log(len(means))).reshape(-1))
            precision_differences.append(solve_triangular(cholesky.T, whitened.T, lower=False).T.reshape(len(tips), len(means), 3))
        score = float(logsumexp(np.concatenate(values)))
        gradient = np.zeros_like(tips)
        for log_terms, differences in zip(values, precision_differences):
            responsibilities = np.exp(log_terms.reshape(differences.shape[:2]) - score)
            gradient -= np.sum(responsibilities[:, :, None] * differences, axis=1)
        return score, gradient

    def log_expected_density(self, tips, body_weights):
        return self.value_and_tip_gradient(tips, body_weights)[0]

    def display_samples(self, rng):
        """Actual mixture samples for display; no scaling of uncertainty."""
        values, weights = [], []
        for weight, means, cholesky, _normalizer in self.terms:
            values.append(means + rng.normal(size=means.shape) @ cholesky.T)
            weights.extend([weight / len(means)] * len(means))
        points = np.vstack(values)
        # Fixed size, weighted display; even negligible IPPE branches keep their true weights.
        chosen = rng.choice(len(points), size=96, p=np.asarray(weights) / np.sum(weights))
        return points[chosen]


@dataclass
class Prediction:
    increment: np.ndarray
    tips: np.ndarray
    rods: np.ndarray
    poses: tuple
    score: float
    minimum_clearance: float = float("inf")
    ik: dict = field(default_factory=dict)


class BodyCandidates:
    def __init__(self, urdf, candidates, rod_length):
        self.ids = tuple(candidate["id"] for candidate in candidates)
        self.weights = np.array([candidate["weight"] for candidate in candidates])
        self.bodies = [EquilibriumBody(urdf, candidate["stiffness"]) for candidate in candidates]
        self.tip = np.array([rod_length, 0.0, 0.0])
        self.warm = [None] * len(self.bodies)
        self.command = None
        self.current_poses = None
        self.current_flanges = None

    def current(self, command):
        command = np.asarray(command, dtype=float)
        if self.command is None or not np.array_equal(command, self.command):
            self.current_poses = tuple(body.equilibrium(command, warm) for body, warm in zip(self.bodies, self.warm))
            self.current_flanges = tuple(body.flange(q) for body, q in zip(self.bodies, self.current_poses))
            self.warm = [q.copy() for q in self.current_poses]
            self.command = command.copy()
        return self.current_poses, self.current_flanges

    def predict(self, command, increment):
        poses, flanges = self.current(command)
        tips, rods, after = [], [], []
        for body, q0, h0 in zip(self.bodies, poses, flanges):
            # The identical latent body appears on both sides of this inverse/composition.
            if np.all(increment == 0):
                q1, relative = q0.copy(), np.eye(4)
            else:
                q1 = body.equilibrium(np.asarray(command) + increment, q0)
                relative = inverse(h0) @ body.flange(q1)
            tips.append(apply(relative, self.tip))
            rods.append(apply(relative, np.array([[0, 0, 0], self.tip])))
            after.append(q1)
        return Prediction(np.array(increment, copy=True), np.array(tips), np.array(rods), tuple(after), float("-inf"))

    def safe_path(self, command, prediction, marker_modes, obstacles, settings):
        safe, smallest = self.safe_motion(command, command + prediction.increment, command,
                                         marker_modes, obstacles, settings)
        prediction.minimum_clearance = smallest
        return safe

    def safe_motion(self, start, end, observed_command, marker_modes, obstacles, settings):
        """Check a splice in the observation's frame, pairing the same body throughout."""
        observed_poses, flanges = self.current(observed_command)
        smallest = float("inf")
        for body, observed_pose, h0 in zip(self.bodies, observed_poses, flanges):
            q0 = (observed_pose if np.array_equal(start, observed_command) else
                  body.equilibrium(start, observed_pose))
            reference = inverse(h0)
            for _command, pose in motion_samples(body, start, end, q0, settings):
                boxes = body.boxes(pose, reference)
                for marker in marker_modes:
                    gap, _pair = clearance(boxes, marker, obstacles)
                    smallest = min(smallest, gap)
                    if gap < settings["clearance"]:
                        return False, smallest
        return True, smallest

    def tip_jacobians(self, command, prediction):
        _poses, flanges = self.current(command)
        return np.array([body.tip_command_jacobian(q1, h0[:3, :3].T)
                         for body, q1, h0 in zip(self.bodies, prediction.poses, flanges)])


class Controller:
    """No access to the plant, world marker pose, true joints, or display TF."""
    def __init__(self, urdf, control, motion, marker, camera, rod_length, obstacles, seed):
        # This argument contains known marker geometry and the image prior only.
        self.goal_offset = np.array(marker["goal_offset"])
        observation_settings = {key: marker[key] for key in
                                ("id", "size", "corner_sigma_px", "translation_prior_variance", "adaptive_covariance")}
        self.observation = ImageObservation(observation_settings, camera)
        self.hand_camera = transform(camera["translation"], camera["rpy"])
        self.candidates = BodyCandidates(urdf, control["bodies"], rod_length)
        self.settings, self.motion, self.obstacles = control, motion, obstacles
        self.rng = np.random.default_rng(seed)
        self.display_rng = np.random.default_rng(seed + 1)

    def propose(self, command, goal):
        settings = self.settings
        eps, limit = settings["jacobian_step"], settings["max_joint_step"]
        nominal = int(np.argmax(self.candidates.weights))
        jacobian = np.column_stack([(self.candidates.predict(command, np.eye(6)[j] * eps).tips[nominal]
                                     - self.candidates.tip) / eps for j in range(6)])
        error = goal - self.candidates.tip
        error *= min(1.0, settings["proposal_distance"] / max(np.linalg.norm(error), 1e-12))
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + settings["damping"] ** 2 * np.eye(3), error)
        step *= min(1.0, limit / max(np.max(np.abs(step)), 1e-12))
        increments = [np.zeros(6)] + [step * fraction for fraction in (1.0, 0.5, 0.25)]
        increments += [axis * sign * limit / 2 for axis in np.eye(6) for sign in (-1, 1)]
        body = self.candidates.bodies[nominal]
        return [value for value in increments if np.all(command + value >= body.lower) and np.all(command + value <= body.upper)]

    def choose(self, image, command, stamp, on_iteration=None):
        result, debug = self.observation.estimate(image, stamp)
        if result is None:
            return None, None, None, debug
        goal = GoalLikelihood(result.record, self.hand_camera, self.goal_offset,
                              self.settings["orientation_samples"], self.rng)
        proposals = []
        for increment in self.propose(np.asarray(command), goal.mean):
            try:
                prediction = self.candidates.predict(command, increment)
                prediction.score = goal.log_expected_density(prediction.tips, self.candidates.weights)
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
        objective_scale = max(1.0, abs(seed.score))
        cache = {tuple(item.increment): item for item in proposals}
        trace = []

        def evaluate(increment):
            key = tuple(np.asarray(increment, dtype=float))
            if key not in cache:
                try:
                    item = self.candidates.predict(command, np.asarray(increment))
                    item.score = goal.log_expected_density(item.tips, self.candidates.weights)
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
            _score, tip_gradient = goal.value_and_tip_gradient(item.tips, self.candidates.weights)
            jacobians = self.candidates.tip_jacobians(command, item)
            return -np.einsum("ij,ijk->k", tip_gradient, jacobians) / objective_scale

        def iteration(increment):
            item = evaluate(increment)
            if item is None:
                return
            trace.append(dict(iteration=len(trace), increment=item.increment.copy(),
                              log_expected_density=item.score, predicted_tips=item.tips.copy()))
            if on_iteration is not None:
                on_iteration(item, goal, result, debug, len(trace) - 1)

        iteration(seed.increment)
        body = self.candidates.bodies[0]
        limit = self.settings["max_joint_step"]
        bounds = list(zip(np.maximum(body.lower - command, -limit),
                          np.minimum(body.upper - command, limit)))
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
                    optimized_increment=solution.x.copy(), trace=trace)
        for prediction in sorted(proposals, key=lambda item: item.score, reverse=True):
            if self.candidates.safe_path(np.asarray(command), prediction, goal.marker_modes, self.obstacles, self.motion):
                prediction.ik = info
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
