import numpy as np

from probik_demo.distribution_metrics import fit_bingham_from_quaternion_samples


class EndEffectorBeliefModel:
    def __init__(
        self,
        robot_model,
        joint_noise_stddev,
        position_covariance_floor=5e-5,
        sample_count=24,
        sample_seed=23,
        bingham_integration_steps=80,
        bingham_fit_max_iterations=40,
        orientation_initial_concentrations=(420.0, 320.0, 220.0),
    ):
        self.robot_model = robot_model
        self.joint_noise_stddev = np.asarray(joint_noise_stddev, dtype=float)
        self.position_covariance_floor = float(position_covariance_floor)
        self.sample_count = max(int(sample_count), self.robot_model.dof + 2)
        self.sample_seed = int(sample_seed)
        self.bingham_integration_steps = int(bingham_integration_steps)
        self.bingham_fit_max_iterations = int(bingham_fit_max_iterations)
        self.orientation_initial_concentrations = np.asarray(orientation_initial_concentrations, dtype=float)

        self._base_offsets = self._make_whitened_offsets()
        self._last_fitted_eigenvalues = None
        self._cache = {}

    def clear_cache(self):
        self._cache.clear()

    def _make_whitened_offsets(self):
        rng = np.random.default_rng(self.sample_seed)
        offsets = rng.normal(size=(self.sample_count, self.robot_model.dof))
        offsets -= offsets.mean(axis=0, keepdims=True)

        covariance = offsets.T @ offsets / max(offsets.shape[0], 1)
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
        whitening = eigenvectors @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, 1e-8))) @ eigenvectors.T
        offsets = offsets @ whitening.T
        offsets -= offsets.mean(axis=0, keepdims=True)
        return offsets

    def _make_joint_samples(self, joint_positions):
        joint_positions = np.asarray(joint_positions, dtype=float)
        noisy_samples = joint_positions[None, :] + self._base_offsets * self.joint_noise_stddev[None, :]
        return np.asarray(
            [self.robot_model.clip_to_limits(sample) for sample in noisy_samples],
            dtype=float,
        )

    def estimate_distribution(self, joint_positions):
        joint_positions = self.robot_model.clip_to_limits(joint_positions)
        cache_key = tuple(np.round(joint_positions, 8).tolist())
        if cache_key in self._cache:
            return self._cache[cache_key]

        position_mode, quaternion_mode, _ = self.robot_model.forward_kinematics(joint_positions)
        sampled_joint_positions = self._make_joint_samples(joint_positions)

        sampled_positions = []
        sampled_quaternions = []
        for sample in sampled_joint_positions:
            sample_position, sample_quaternion, _ = self.robot_model.forward_kinematics(sample)
            sampled_positions.append(sample_position)
            sampled_quaternions.append(sample_quaternion)

        sampled_positions = np.asarray(sampled_positions, dtype=float)
        sampled_quaternions = np.asarray(sampled_quaternions, dtype=float)

        position_mean = sampled_positions.mean(axis=0)
        position_covariance = np.cov(sampled_positions.T, bias=True) + self.position_covariance_floor * np.eye(
            3,
            dtype=float,
        )

        bingham_fit = fit_bingham_from_quaternion_samples(
            sampled_quaternions,
            initial_eigenvalues=self._last_fitted_eigenvalues,
            integration_steps=self.bingham_integration_steps,
            max_iterations=self.bingham_fit_max_iterations,
            fallback_concentrations=self.orientation_initial_concentrations,
            reference_mode=quaternion_mode,
        )
        self._last_fitted_eigenvalues = bingham_fit["Z"]

        estimate = {
            "position_mean": position_mean,
            "position_mode": position_mode,
            "position_covariance": position_covariance,
            "orientation_mode": bingham_fit["mode"],
            "orientation_deterministic_mode": quaternion_mode,
            "orientation_bingham": bingham_fit["A"],
            "orientation_log_normalizer": bingham_fit["log_normalizer"],
        }
        self._cache[cache_key] = estimate
        return estimate
