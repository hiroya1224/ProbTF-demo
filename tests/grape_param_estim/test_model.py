import unittest

import numpy as np

from grape_param_estim.data import AnalysisData
from grape_param_estim.model import (
    GrapeRigidBodyModel,
    RigidBodyParameters,
    command_to_wrench,
    replay_segments,
    rotation_vector_from_matrix,
)


NOMINAL_MASS = 2.351557590812377
NOMINAL_INERTIA = (0.0649940671, 0.0649466618, 0.1289801290)


def make_analysis(two_segments=False) -> AnalysisData:
    times = np.linspace(0.0, 0.4, 41)
    quaternion = np.zeros((times.size, 4))
    quaternion[:, 3] = 1.0
    thrust = np.tile(
        np.asarray((5.7, 5.9, 5.8, 5.6), dtype=float),
        (times.size, 1),
    )
    thrust += 0.15 * np.sin(7.0 * times)[:, np.newaxis] * np.asarray(
        ((1.0, -0.5, 0.25, -0.75),)
    )
    gimbal = np.tile(
        np.asarray((0.04, -0.03, 0.02, -0.01), dtype=float),
        (times.size, 1),
    )
    if two_segments:
        segment_id = (times >= 0.2).astype(int)
    else:
        segment_id = np.zeros(times.size, dtype=int)
    return AnalysisData(
        bag_path="/tmp/synthetic.bag",
        start_time=0.0,
        end_time=0.4,
        segment_duration=0.2 if two_segments else 0.4,
        times=times,
        position=np.zeros((times.size, 3)),
        orientation_xyzw=quaternion,
        linear_velocity=np.zeros((times.size, 3)),
        angular_velocity=np.zeros((times.size, 3)),
        specific_force=np.zeros((times.size, 3)),
        base_thrust=thrust,
        gimbal_angle=gimbal,
        flight_state=np.full(times.size, 5),
        segment_id=segment_id,
    )


def nominal_parameters(**scales) -> RigidBodyParameters:
    return RigidBodyParameters.from_diagonal(
        NOMINAL_MASS, NOMINAL_INERTIA, **scales
    )


class CommandGeometryTest(unittest.TestCase):
    def test_symmetric_vertical_command_has_expected_force(self):
        wrench = command_to_wrench(np.ones(4), np.zeros(4))

        np.testing.assert_allclose(wrench[:3], (0.0, 0.0, 4.0), atol=1e-12)
        self.assertAlmostEqual(wrench[5], 0.0, places=12)

    def test_rotation_log_recovers_axis_angle(self):
        angle = 0.2
        rotation = np.asarray(
            (
                (np.cos(angle), -np.sin(angle), 0.0),
                (np.sin(angle), np.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            )
        )

        np.testing.assert_allclose(
            rotation_vector_from_matrix(rotation),
            (0.0, 0.0, angle),
            atol=1e-12,
        )


class GrapeRigidBodyModelTest(unittest.TestCase):
    def setUp(self):
        self.model = GrapeRigidBodyModel(maximum_time_step=0.005)
        self.data = make_analysis()
        self.segment = next(self.data.segments())[1]

    def test_mass_and_force_common_scale_preserves_translation(self):
        nominal = self.model.simulate_segment(
            self.data, nominal_parameters(), self.segment
        )
        scaled = self.model.simulate_segment(
            self.data,
            nominal_parameters(mass_scale=1.7, force_scale=1.7),
            self.segment,
        )

        np.testing.assert_allclose(
            scaled.position, nominal.position, atol=1e-11
        )
        np.testing.assert_allclose(
            scaled.linear_velocity, nominal.linear_velocity, atol=1e-11
        )

    def test_inertia_and_torque_common_scale_preserves_rotation(self):
        nominal = self.model.simulate_segment(
            self.data, nominal_parameters(), self.segment
        )
        scaled = self.model.simulate_segment(
            self.data,
            nominal_parameters(inertia_scale=1.6, torque_scale=1.6),
            self.segment,
        )

        np.testing.assert_allclose(
            scaled.orientation_xyzw,
            nominal.orientation_xyzw,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            scaled.angular_velocity,
            nominal.angular_velocity,
            atol=1e-11,
        )

    def test_nominal_synthetic_trajectory_replays_with_zero_residual(self):
        generated = self.model.simulate_segment(
            self.data, nominal_parameters(), self.segment
        )
        observed = AnalysisData(
            bag_path=self.data.bag_path,
            start_time=self.data.start_time,
            end_time=self.data.end_time,
            segment_duration=self.data.segment_duration,
            times=self.data.times,
            position=generated.position,
            orientation_xyzw=generated.orientation_xyzw,
            linear_velocity=generated.linear_velocity,
            angular_velocity=generated.angular_velocity,
            specific_force=self.data.specific_force,
            base_thrust=self.data.base_thrust,
            gimbal_angle=self.data.gimbal_angle,
            flight_state=self.data.flight_state,
            segment_id=self.data.segment_id,
        )

        replay = replay_segments(
            observed, self.model, nominal_parameters()
        )

        np.testing.assert_allclose(
            replay.position, observed.position, atol=1e-12
        )
        np.testing.assert_allclose(replay.residual_se3, 0.0, atol=1e-11)
        np.testing.assert_allclose(
            replay.correction_translation, 0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            replay.correction_rotation_vector, 0.0, atol=1e-11
        )

    def test_each_segment_restarts_from_its_observed_state(self):
        data = make_analysis(two_segments=True)

        replay = replay_segments(data, self.model, nominal_parameters())
        starts = [segment.start for _, segment in data.segments()]

        np.testing.assert_allclose(
            replay.position[starts], data.position[starts], atol=0.0
        )
        np.testing.assert_allclose(
            replay.correction_translation[starts], 0.0, atol=0.0
        )
        np.testing.assert_allclose(
            replay.residual_se3[starts], 0.0, atol=1e-15
        )


if __name__ == "__main__":
    unittest.main()
