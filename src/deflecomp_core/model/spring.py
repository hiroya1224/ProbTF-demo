from abc import ABC, abstractmethod

import numpy as np


class SpringModel(ABC):
    @abstractmethod
    def torque(self, theta: np.ndarray, theta_cmd: np.ndarray, kp_vec: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def potential(self, theta: np.ndarray, theta_cmd: np.ndarray, kp_vec: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def stiffness_diag(self, theta: np.ndarray, theta_cmd: np.ndarray, kp_vec: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def log_stiffness_jacobian_diag(
        self,
        theta: np.ndarray,
        theta_cmd: np.ndarray,
        kp_vec: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def theta_cmd_from_theta_ref(
        self,
        tau_gravity: np.ndarray,
        theta_ref: np.ndarray,
        kp_vec: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class LinearSpringModel(SpringModel):
    def torque(self, theta: np.ndarray, theta_cmd: np.ndarray, kp_vec: np.ndarray) -> np.ndarray:
        return kp_vec * (theta - theta_cmd)

    def potential(self, theta: np.ndarray, theta_cmd: np.ndarray, kp_vec: np.ndarray) -> float:
        delta = theta - theta_cmd
        return float(0.5 * np.dot(delta * kp_vec, delta))

    def stiffness_diag(self, theta: np.ndarray, theta_cmd: np.ndarray, kp_vec: np.ndarray) -> np.ndarray:
        del theta, theta_cmd
        return np.asarray(kp_vec, dtype=float)

    def log_stiffness_jacobian_diag(
        self,
        theta: np.ndarray,
        theta_cmd: np.ndarray,
        kp_vec: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(kp_vec, dtype=float) * (np.asarray(theta, dtype=float) - np.asarray(theta_cmd, dtype=float))

    def theta_cmd_from_theta_ref(
        self,
        tau_gravity: np.ndarray,
        theta_ref: np.ndarray,
        kp_vec: np.ndarray,
    ) -> np.ndarray:
        kp_safe = np.maximum(np.asarray(kp_vec, dtype=float), 1e-12)
        return np.asarray(theta_ref, dtype=float) + np.asarray(tau_gravity, dtype=float) / kp_safe


class PeriodicSpringModel(SpringModel):
    def torque(self, theta: np.ndarray, theta_cmd: np.ndarray, kp_vec: np.ndarray) -> np.ndarray:
        delta = np.asarray(theta, dtype=float) - np.asarray(theta_cmd, dtype=float)
        return 2.0 * np.asarray(kp_vec, dtype=float) * np.sin(0.5 * delta)

    def potential(self, theta: np.ndarray, theta_cmd: np.ndarray, kp_vec: np.ndarray) -> float:
        delta = np.asarray(theta, dtype=float) - np.asarray(theta_cmd, dtype=float)
        return float(np.sum(4.0 * np.asarray(kp_vec, dtype=float) * (1.0 - np.cos(0.5 * delta))))

    def stiffness_diag(self, theta: np.ndarray, theta_cmd: np.ndarray, kp_vec: np.ndarray) -> np.ndarray:
        delta = np.asarray(theta, dtype=float) - np.asarray(theta_cmd, dtype=float)
        return np.asarray(kp_vec, dtype=float) * np.cos(0.5 * delta)

    def log_stiffness_jacobian_diag(
        self,
        theta: np.ndarray,
        theta_cmd: np.ndarray,
        kp_vec: np.ndarray,
    ) -> np.ndarray:
        return self.torque(theta=theta, theta_cmd=theta_cmd, kp_vec=kp_vec)

    def theta_cmd_from_theta_ref(
        self,
        tau_gravity: np.ndarray,
        theta_ref: np.ndarray,
        kp_vec: np.ndarray,
    ) -> np.ndarray:
        kp_safe = np.maximum(np.asarray(kp_vec, dtype=float), 1e-12)
        arg = np.asarray(tau_gravity, dtype=float) / (2.0 * kp_safe)
        arg = np.clip(arg, -1.0 + 1e-9, 1.0 - 1e-9)
        return np.asarray(theta_ref, dtype=float) + 2.0 * np.arcsin(arg)


class JointTypeAwareSpringModel(SpringModel):
    def __init__(self, periodic_mask: np.ndarray) -> None:
        self.periodic_mask = np.asarray(periodic_mask, dtype=bool).copy()
        self.linear_model = LinearSpringModel()
        self.periodic_model = PeriodicSpringModel()

    @classmethod
    def from_joint_types(cls, joint_types) -> "JointTypeAwareSpringModel":
        periodic_types = {"revolute", "continuous"}
        return cls([joint_type in periodic_types for joint_type in joint_types])

    def _mask(self, size: int) -> np.ndarray:
        if self.periodic_mask.size != size:
            raise ValueError(
                f"JointTypeAwareSpringModel mask size {self.periodic_mask.size} does not match vector size {size}"
            )
        return self.periodic_mask

    def torque(self, theta: np.ndarray, theta_cmd: np.ndarray, kp_vec: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        theta_cmd = np.asarray(theta_cmd, dtype=float)
        kp_vec = np.asarray(kp_vec, dtype=float)
        mask = self._mask(theta.size)
        torque = self.linear_model.torque(theta, theta_cmd, kp_vec)
        torque[mask] = self.periodic_model.torque(theta[mask], theta_cmd[mask], kp_vec[mask])
        return torque

    def potential(self, theta: np.ndarray, theta_cmd: np.ndarray, kp_vec: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        theta_cmd = np.asarray(theta_cmd, dtype=float)
        kp_vec = np.asarray(kp_vec, dtype=float)
        mask = self._mask(theta.size)
        total = 0.0
        if np.any(mask):
            total += self.periodic_model.potential(theta[mask], theta_cmd[mask], kp_vec[mask])
        if np.any(~mask):
            total += self.linear_model.potential(theta[~mask], theta_cmd[~mask], kp_vec[~mask])
        return float(total)

    def stiffness_diag(self, theta: np.ndarray, theta_cmd: np.ndarray, kp_vec: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        theta_cmd = np.asarray(theta_cmd, dtype=float)
        kp_vec = np.asarray(kp_vec, dtype=float)
        mask = self._mask(theta.size)
        stiffness = self.linear_model.stiffness_diag(theta, theta_cmd, kp_vec)
        stiffness[mask] = self.periodic_model.stiffness_diag(theta[mask], theta_cmd[mask], kp_vec[mask])
        return stiffness

    def log_stiffness_jacobian_diag(
        self,
        theta: np.ndarray,
        theta_cmd: np.ndarray,
        kp_vec: np.ndarray,
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        theta_cmd = np.asarray(theta_cmd, dtype=float)
        kp_vec = np.asarray(kp_vec, dtype=float)
        mask = self._mask(theta.size)
        jac = self.linear_model.log_stiffness_jacobian_diag(theta, theta_cmd, kp_vec)
        jac[mask] = self.periodic_model.log_stiffness_jacobian_diag(theta[mask], theta_cmd[mask], kp_vec[mask])
        return jac

    def theta_cmd_from_theta_ref(
        self,
        tau_gravity: np.ndarray,
        theta_ref: np.ndarray,
        kp_vec: np.ndarray,
    ) -> np.ndarray:
        tau_gravity = np.asarray(tau_gravity, dtype=float)
        theta_ref = np.asarray(theta_ref, dtype=float)
        kp_vec = np.asarray(kp_vec, dtype=float)
        mask = self._mask(theta_ref.size)
        theta_cmd = self.linear_model.theta_cmd_from_theta_ref(tau_gravity, theta_ref, kp_vec)
        theta_cmd[mask] = self.periodic_model.theta_cmd_from_theta_ref(
            tau_gravity[mask],
            theta_ref[mask],
            kp_vec[mask],
        )
        return theta_cmd
