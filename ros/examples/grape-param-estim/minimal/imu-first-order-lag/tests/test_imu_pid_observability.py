"""Actual nonlinear plant/controller checks for the IMU translation certificate.

The wrapper below exercises the existing controller, actuator response and RK4
plant with an estimated controller input. It is not a navigation estimator.
The causal-estimator step follows mathematically from identical input histories.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))
from imu_pid_observability import (imu_observation_for_certificate,
                                  imu_only_stability_certificate)
from imu_pid_inputs import load_imu_pid_inputs
from grape_param_estim.controller import ControllerConfig, GrapeController
from grape_param_estim.dynamics import FullSixDofPlant, advance_actuators
from grape_param_estim.geometry import euler_xyz_to_matrix, matrix_to_quaternion
from grape_param_estim.gimbalrotor_pid_postprocess import load_vehicle_model
from grape_param_estim.system import (ActuatorParameters, ActuatorState,
    ControllerState, GRAVITY, ReferenceState, RigidBodyState)


def physical_setup():
    model = load_vehicle_model(HERE.parent/'grape_vehicle_model.json')
    # Nonzero velocity drag does not remove the true-position symmetry.
    parameters = replace(model.parameters,linear_drag=np.array([.1,.2,.3]),
                         angular_drag=np.array([.02,.03,.04]))
    plant = FullSixDofPlant(parameters,model.body_geometry)
    state = RigidBodyState(position=np.array([1.,-.3,.7]),
        orientation_xyzw=matrix_to_quaternion(euler_xyz_to_matrix([.13,-.21,.32])),
        linear_velocity=np.array([.1,-.2,.05]),angular_velocity=np.array([.07,-.08,.04]))
    actuators = ActuatorState(thrust=parameters.mass*GRAVITY*np.array([.24,.26,.25,.25]),
                             gimbal_angle=np.array([.01,-.02,.03,-.01]))
    return model,plant,state,actuators


def estimated_state_controlled_step(plant,controller,true_state,estimated_state,
        reference,controller_state,actuators,actuator_parameters,sensor_state,taus,
        residual,dt):
    command,next_controller = controller.step(estimated_state,reference,
        controller_state,dt,gimbal_angles=actuators.gimbal_angle)
    midpoint = advance_actuators(actuators,command,actuator_parameters,.5*dt)
    next_true = plant.step(0.,true_state,midpoint,dt,residual)
    next_actuators = advance_actuators(midpoint,command,actuator_parameters,.5*dt)
    observation = imu_observation_for_certificate(plant,next_true,next_actuators,
        lever_body=np.array([.02,-.03,.01]),residual_wrench_body=residual)
    # Exact ZOH discretisation of both specified sensor lag equations.
    filtered = {name: observation[name]+(sensor_state[name]-observation[name])
        *np.exp(-dt/tau) for name,tau in zip(('gyro','accel'),taus)}
    return command,next_controller,next_true,next_actuators,observation,filtered


class TranslationCertificateChecks(unittest.TestCase):
    def test_actual_continuous_and_rk4_jacobians_have_exact_position_columns(self):
        model,plant,state,actuators = physical_setup()
        residual = np.array([.2,-.1,.3,.004,-.005,.006])
        _,continuous,_,_ = plant.derivative_with_jacobian(0.,state.as_vector(),actuators,residual)
        np.testing.assert_array_equal(continuous[:,:3],np.zeros((13,3)))
        step = plant.step_with_jacobian(0.,state,actuators,.005,residual)
        np.testing.assert_array_equal(step.jacobian.state[:,:3],np.eye(12)[:,:3])
        shift = np.array([.3,-.7,.11])
        shifted = replace(state,position=state.position+shift)
        actual = plant.step(0.,shifted,actuators,.005,residual)
        np.testing.assert_allclose(actual.position-step.next_state.position,shift,
                                   rtol=0.,atol=2e-16)
        np.testing.assert_array_equal(actual.as_vector()[3:],step.next_state.as_vector()[3:])

    def test_estimated_state_controller_and_both_sensor_lags_keep_translation(self):
        model,plant,state,actuators = physical_setup()
        controller = GrapeController(ControllerConfig.grape(),model.parameters,model.body_geometry)
        estimated = replace(state,position=np.array([.8,-.2,.6]))
        reference = ReferenceState(np.array([.7,.1,.9]),np.zeros(3),np.zeros(3),
                                   np.zeros(3),np.zeros(3),np.zeros(3))
        controller_state = ControllerState(np.array([.01,.02,.03,-.02,.01,.01]))
        sensor_state = dict(gyro=np.array([.1,-.02,.04]),accel=np.array([.3,-.1,9.6]))
        common = dict(plant=plant,controller=controller,estimated_state=estimated,
            reference=reference,controller_state=controller_state,actuators=actuators,
            actuator_parameters=ActuatorParameters(thrust_time_constant=.16),
            sensor_state=sensor_state,taus=(.03,.08),
            residual=np.array([.2,-.1,.3,.004,-.005,.006]),dt=.005)
        baseline = estimated_state_controlled_step(true_state=state,**common)
        # Three independent true-position perturbations: estimator/reference
        # state deliberately remains fixed, as required by the full Jacobian.
        for shift in (.3*np.eye(3)):
            with self.subTest(shift=shift):
                shifted = estimated_state_controlled_step(
                    true_state=replace(state,position=state.position+shift),**common)
                for name in ('thrust','gimbal_angle','virtual_force','desired_acceleration'):
                    np.testing.assert_array_equal(getattr(shifted[0],name),getattr(baseline[0],name))
                np.testing.assert_array_equal(shifted[1].integral_error,baseline[1].integral_error)
                np.testing.assert_allclose(shifted[2].position-baseline[2].position,shift,
                                           rtol=0.,atol=2e-16)
                np.testing.assert_array_equal(shifted[2].as_vector()[3:],baseline[2].as_vector()[3:])
                np.testing.assert_array_equal(shifted[3].thrust,baseline[3].thrust)
                np.testing.assert_array_equal(shifted[3].gimbal_angle,baseline[3].gimbal_angle)
                for name in ('gyro','accel','magnetic'):
                    np.testing.assert_array_equal(shifted[4][name],baseline[4][name])
                for name in ('gyro','accel'):
                    np.testing.assert_array_equal(shifted[5][name],baseline[5][name])

    def test_true_state_feedback_is_a_different_and_symmetry_breaking_contract(self):
        model,plant,state,actuators = physical_setup()
        controller = GrapeController(ControllerConfig.grape(),model.parameters,model.body_geometry)
        reference = ReferenceState(state.position.copy(),np.zeros(3),np.zeros(3),
                                   np.zeros(3),np.zeros(3),np.zeros(3))
        internal = ControllerState(np.zeros(6))
        original,_ = controller.step(state,reference,internal,.005)
        shifted,_ = controller.step(replace(state,position=state.position+[.2,0.,0.]),
                                    reference,internal,.005)
        self.assertGreater(np.linalg.norm(shifted.desired_acceleration-original.desired_acceleration),.1)

    def test_certificate_preserves_all_nodes_and_does_not_invent_numerical_results(self):
        inputs = load_imu_pid_inputs('failure1',node_count=128)
        certificate = imu_only_stability_certificate(inputs)
        json.dumps(certificate,allow_nan=False)
        self.assertEqual(certificate['denominator'],128)
        self.assertEqual(certificate['known_unit_modes'],3)
        self.assertEqual(certificate['rho_lower_bound'],1.)
        self.assertEqual(certificate['stable_fraction'],0.)
        self.assertTrue(certificate['criterion_decided'])
        self.assertFalse(certificate['numeric_evaluated'])
        self.assertFalse(certificate['jacobian_computed'])
        self.assertFalse(certificate['hover_validity_evaluated'])
        self.assertIsNone(certificate['hover_exists'])
        self.assertIsNone(certificate['hover_trim'])
        self.assertIsNone(certificate['exact_spectral_radius'])
        self.assertFalse(certificate['stable_models_certified'])
        np.testing.assert_array_equal(list(certificate['sensor_time_constants_seconds'].values()),
                                      inputs['nominal_taus'])
        for record in certificate['source'].values():
            source = Path(record['path'])
            self.assertTrue(source.is_file())
            self.assertEqual(record['sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertIn('restricted manifold',certificate['known_initial_frame_caveat'])

    def test_certificate_declines_explicit_symmetry_breaking_models(self):
        inputs = load_imu_pid_inputs('failure1',node_count=1)
        for flag in ('external_position_aiding','position_dependent_external_force',
                     'position_dependent_gravity','position_dependent_magnetic_field'):
            with self.subTest(flag=flag),self.assertRaises(ValueError):
                imu_only_stability_certificate(dict(inputs,**{flag:True}))


if __name__ == '__main__':
    unittest.main()
