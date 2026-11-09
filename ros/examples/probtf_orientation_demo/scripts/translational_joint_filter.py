"""Finite joint attitude/translation recursion for the isolated IMU experiment.

Observation factors and Gaussian marginalization are analytic. Gyro products
propagate the required quaternion/rotation moments with the same independent
Bingham increment. Finite forward-KL projection and deterministic integration
are the explicitly numerical parts; no truth or sampled particles enter here.
"""
import numpy as np

from probtf.geometry.quaternion import quat_mul, quat_right_matrix
from translational_attitude_law import ExponentialAttitudeLaw, forward_kl_projection
from translational_observation_factors import acceleration_factor, magnetic_factor
from translational_prediction import predict_joint_moments, conditional_parameters


class JointTranslationAttitudeFilter:
    def __init__(self, initial_wxyz, gravity_world, magnetic_world,
                 gyro_covariance, acceleration_covariance, magnetic_covariance,
                 use_magnetic=True, estimate_translation=True, quadrature_order=7,
                 projection_iterations=3, initial_attitude_std_deg=3.,
                 initial_acceleration_std=.3, acceleration_time_constant=.7,
                 acceleration_stationary_std=(1.5, 1.5, 1.5),
                 quadrature_options=None):
        self.g = np.asarray(gravity_world, float)
        self.b = np.asarray(magnetic_world, float)
        self.gyro_covariance = np.asarray(gyro_covariance, float)
        self.acceleration_covariance = np.asarray(acceleration_covariance, float)
        self.magnetic_covariance = np.asarray(magnetic_covariance, float)
        self.use_magnetic = bool(use_magnetic)
        self.estimate_translation = bool(estimate_translation)
        self.order = int(quadrature_order)
        self.projection_iterations = int(projection_iterations)
        # Numerical integration controls only. An empty mapping preserves
        # the standalone experiment's historical defaults and density law.
        self.quadrature_options = dict(quadrature_options or {})
        self.acceleration_time_constant = float(acceleration_time_constant)
        self.stationary_acceleration_covariance = np.diag(np.asarray(acceleration_stationary_std, float)**2)
        self.reference = np.asarray(initial_wxyz, float).copy()
        self.reference /= np.linalg.norm(self.reference)
        variance = np.deg2rad(float(initial_attitude_std_deg))**2
        matrix = 2. / variance * (np.outer(self.reference, self.reference) - np.eye(4))
        self.law = ExponentialAttitudeLaw(matrix, np.zeros((3,3)), np.zeros((3,3)), self.g, self.b)
        self.proposal = variance / 4. * np.eye(3)
        self.m = np.zeros(3)
        self.F = np.zeros((3,3))
        self.S = (float(initial_acceleration_std)**2 * np.eye(3)
                  if self.estimate_translation else np.zeros((3,3)))
        self.moments = self.law.integrate(self.reference, self.proposal, self.order,
                                        **self.quadrature_options)
        self.last_diagnostics = {}

    def _predict(self, gyro, dt):
        old = self.moments
        h = old['Ehg']
        covariance_h = old['Cgg']
        mean_a = self.m + self.F @ h
        covariance_a = self.S + self.F @ covariance_h @ self.F.T
        covariance_ah = self.F @ covariance_h
        beta = np.exp(-dt / self.acceleration_time_constant)
        process = (1. - beta**2) * self.stationary_acceleration_covariance
        predicted = predict_joint_moments(old['Mqq'], old['Ehg'], old['Ngg'],
            old['Ehb'], old['Nbb'], mean_a, covariance_a, covariance_ah,
            gyro, dt, self.gyro_covariance, beta, process, cov_hg=covariance_h)
        increment = predicted['delta']
        omega = quat_right_matrix(increment['mean_quaternion'])
        rotation = increment['nominal_rotation']
        transported = ExponentialAttitudeLaw(omega @ self.law.A @ omega.T,
            rotation.T @ self.law.Wg @ rotation,
            rotation.T @ self.law.Wb @ rotation, self.g, self.b)
        reference = quat_mul(self.reference, increment['mean_quaternion'])
        proposal = transported.proposal_covariance(reference, fallback=self.proposal)
        self.law, diagnostics = forward_kl_projection(
            predicted['Mq'], predicted['Ngg'], predicted['Nbb'], transported,
            reference, proposal, order=self.order,
            max_iterations=self.projection_iterations, **self.quadrature_options)
        self.reference, self.proposal = reference, proposal
        if self.estimate_translation:
            conditional = conditional_parameters(predicted['mean_a'], predicted['cov_a'],
                predicted['Ehg'], predicted['cov_hg'], predicted['cov_ah'])
            self.m, self.F, self.S = conditional['m'], conditional['F'], conditional['S']
        return diagnostics

    def update(self, gyro_rad_s, acceleration_m_s2, dt, magnetic=None):
        if dt < 0.:
            raise ValueError('Time intervals must be nonnegative')
        prediction = self._predict(np.asarray(gyro_rad_s, float), float(dt)) if dt else {}
        acceleration = acceleration_factor(acceleration_m_s2, self.m, self.F, self.S,
                                            self.acceleration_covariance, self.g)
        field = (magnetic_factor(magnetic, self.magnetic_covariance, self.b)
                 if self.use_magnetic and magnetic is not None
                 else {'A':np.zeros((4,4)), 'Wb':np.zeros((3,3))})
        # Exactly one copy of the gyro prior plus each NEW observation factor.
        self.law = ExponentialAttitudeLaw(self.law.A + acceleration['A'] + field['A'],
            self.law.Wg + acceleration['Wg'], self.law.Wb + field['Wb'], self.g, self.b)
        self.m, self.F, self.S = (acceleration['updated_m'],
            acceleration['updated_F'], acceleration['updated_S'])
        self.reference, mode_diagnostics = self.law.find_mode(self.reference)
        self.proposal = self.law.proposal_covariance(self.reference, fallback=self.proposal)
        self.moments = self.law.integrate(self.reference, self.proposal, self.order,
                                        **self.quadrature_options)
        moments = self.moments
        local_mode = self.reference.copy()
        mass_reference = np.linalg.eigh(moments['Mqq'])[1][:,-1]
        if mass_reference @ local_mode < 0.:
            mass_reference = -mass_reference
        # Carry a probability-mass reference, rather than letting a narrow
        # high-density satellite mode steer the next integration chart.
        self.reference = mass_reference
        conditional_means = self.m + moments['hg'] @ self.F.T
        acceleration_body = self.m + self.F @ moments['Ehg']
        world_means = np.einsum('nij,nj->ni', moments['R'], conditional_means)
        acceleration_world = moments['weights'] @ world_means
        centered_world = world_means - acceleration_world
        covariance_world = np.einsum('n,nij,jk,nlk->il', moments['weights'],
            moments['R'], self.S, moments['R']) + np.einsum('n,ni,nj->ij',
            moments['weights'], centered_world, centered_world)
        pitch_nodes = np.arcsin(np.clip(-moments['R'][:,2,0],-1.,1.))
        pitch_mean = float(moments['weights'] @ pitch_nodes)
        pitch_variance = float(moments['weights'] @ (pitch_nodes-pitch_mean)**2)
        self.last_diagnostics = {'projection':prediction, 'mode':mode_diagnostics}
        return {'quaternion_wxyz':local_mode,
            'mean_quaternion_wxyz':mass_reference.copy(),
            'pitch_mean_rad':pitch_mean,'pitch_variance_rad2':pitch_variance,
            'acceleration_body_mean':acceleration_body,
            'acceleration_body_covariance':self.S+self.F@moments['Cgg']@self.F.T,
            'acceleration_world_mean':acceleration_world,
            'acceleration_world_covariance':covariance_world,
            'conditional_m':self.m.copy(), 'conditional_F':self.F.copy(),
            'conditional_S':self.S.copy(), 'A':self.law.A.copy(),
            'Wg':self.law.Wg.copy(), 'Wb':self.law.Wb.copy(),
            'Mq':moments['Mqq'].copy(), 'Ngg':moments['Ngg'].copy(),
            'Nbb':moments['Nbb'].copy(), 'diagnostics':self.last_diagnostics}
