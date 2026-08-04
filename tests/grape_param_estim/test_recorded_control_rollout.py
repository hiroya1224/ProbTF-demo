import unittest
from unittest import mock

import numpy as np

from grape_param_estim.batch.recorded_control_rollout import (
    interpolate_observed_pose,
    simulate_recorded_control_rollout,
)
from grape_param_estim.batch_artifact_export import (
    _actuator_intervals_at_delay,
)
from grape_param_estim.dynamics import FullSixDofPlant
from grape_param_estim.geometry import quaternion_to_matrix
from tests.grape_param_estim import test_batch_preparation as preparation_test


class RecordedControlRolloutTests(unittest.TestCase):
    def setUp(self):
        self.helper = preparation_test.BatchPreparationTests()
        self.helper.setUp()
        self.prepared = self.helper._prepare()

    def tearDown(self):
        self.helper.tearDown()

    def _rollout(self):
        return simulate_recorded_control_rollout(
            prepared_bag=self.prepared.bags[0],
            initial_state=self.helper.initialization.state,
            parameter_chart=self.prepared.parameter_chart,
            parameter_coordinates=self.prepared.initial_parameter_coordinates,
            geometry=self.prepared.geometry,
        )

    def test_replays_prepared_commands_from_one_initial_state(self):
        rollout = self._rollout()
        bag = self.prepared.bags[0]
        self.assertTrue(np.all(rollout.valid))
        np.testing.assert_array_equal(
            rollout.time,
            np.asarray(tuple(knot.time for knot in bag.knots)),
        )
        parameters = self.prepared.parameter_chart.decode(
            self.prepared.initial_parameter_coordinates
        )
        body_rotation = quaternion_to_matrix(
            rollout.body_orientation_xyzw[0]
        )
        expected_sensor_position = (
            rollout.cog_position[0]
            + body_rotation
            @ (
                bag.sensor_extrinsics.pose_sensor_position_in_body
                - parameters.cog_offset
            )
        )
        np.testing.assert_allclose(
            rollout.sensor_position[0], expected_sensor_position
        )
        self.assertGreater(
            np.linalg.norm(
                rollout.cog_position[-1] - rollout.cog_position[0]
            ),
            0.0,
        )

    def test_unstable_rollout_is_a_finite_prefix_not_an_export_failure(self):
        with mock.patch.object(
            FullSixDofPlant,
            "step",
            side_effect=ValueError("synthetic divergence"),
        ):
            rollout = self._rollout()
        np.testing.assert_array_equal(
            rollout.valid,
            np.asarray(
                (True,) + (False,) * (rollout.time.size - 1), dtype=bool
            ),
        )
        self.assertTrue(np.all(np.isfinite(rollout.sensor_position)))
        np.testing.assert_allclose(
            rollout.sensor_position[1:],
            np.repeat(
                rollout.sensor_position[0][None, :],
                rollout.time.size - 1,
                axis=0,
            ),
        )

    def test_candidate_cog_rebases_the_same_initial_sensor_pose_and_twist(self):
        anchor_coordinates = (
            self.prepared.initial_parameter_coordinates.copy()
        )
        candidate_coordinates = anchor_coordinates.copy()
        candidate_coordinates[7:10] += np.asarray((0.12, -0.08, 0.04))
        anchor = self._rollout()
        candidate = simulate_recorded_control_rollout(
            prepared_bag=self.prepared.bags[0],
            initial_state=self.helper.initialization.state,
            parameter_chart=self.prepared.parameter_chart,
            parameter_coordinates=candidate_coordinates,
            geometry=self.prepared.geometry,
            initial_state_parameter_coordinates=anchor_coordinates,
        )
        np.testing.assert_allclose(
            candidate.sensor_position[0], anchor.sensor_position[0]
        )
        np.testing.assert_allclose(
            candidate.sensor_orientation_xyzw[0],
            anchor.sensor_orientation_xyzw[0],
        )
        rotation = quaternion_to_matrix(anchor.body_orientation_xyzw[0])
        sensor_position = (
            self.prepared.bags[0]
            .sensor_extrinsics.pose_sensor_position_in_body
        )
        anchor_parameters = self.prepared.parameter_chart.decode(
            anchor_coordinates
        )
        candidate_parameters = self.prepared.parameter_chart.decode(
            candidate_coordinates
        )
        anchor_sensor_velocity = (
            anchor.linear_velocity[0]
            + rotation
            @ np.cross(
                anchor.angular_velocity[0],
                sensor_position - anchor_parameters.cog_offset,
            )
        )
        candidate_sensor_velocity = (
            candidate.linear_velocity[0]
            + rotation
            @ np.cross(
                candidate.angular_velocity[0],
                sensor_position - candidate_parameters.cog_offset,
            )
        )
        np.testing.assert_allclose(
            candidate_sensor_velocity, anchor_sensor_velocity
        )

    def test_observed_pose_interpolation_does_not_extrapolate(self):
        half_angle = 0.25 * np.pi
        orientation = np.asarray(
            (
                (0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, np.sin(half_angle), np.cos(half_angle)),
            )
        )
        position, interpolated, valid = interpolate_observed_pose(
            (1.0, 3.0),
            np.asarray(((0.0, 0.0, 0.0), (2.0, 4.0, 6.0))),
            orientation,
            (0.0, 1.0, 2.0, 3.0, 4.0),
        )
        np.testing.assert_array_equal(
            valid, np.asarray((False, True, True, True, False))
        )
        np.testing.assert_allclose(position[2], (1.0, 2.0, 3.0))
        midpoint_rotation = quaternion_to_matrix(interpolated[2])
        expected = np.asarray(
            ((np.sqrt(0.5), -np.sqrt(0.5), 0.0),
             (np.sqrt(0.5), np.sqrt(0.5), 0.0),
             (0.0, 0.0, 1.0))
        )
        np.testing.assert_allclose(midpoint_rotation, expected, atol=1.0e-12)

    def test_posterior_delay_rebuilds_the_recorded_zoh_segments(self):
        bag = self.prepared.bags[0]
        zero_delay = _actuator_intervals_at_delay(
            bag, self.helper.flight, 0.0
        )
        shifted = _actuator_intervals_at_delay(
            bag, self.helper.flight, 0.04
        )
        zero_signature = tuple(
            (
                segment.duration,
                tuple(segment.command.thrust),
                tuple(segment.command.gimbal_angle),
            )
            for interval in zero_delay
            for segment in interval.delayed_command_segments
        )
        shifted_signature = tuple(
            (
                segment.duration,
                tuple(segment.command.thrust),
                tuple(segment.command.gimbal_angle),
            )
            for interval in shifted
            for segment in interval.delayed_command_segments
        )
        self.assertNotEqual(zero_signature, shifted_signature)
        for intervals in (zero_delay, shifted):
            for index, interval in enumerate(intervals):
                expected = bag.knots[index + 1].time - bag.knots[index].time
                self.assertAlmostEqual(
                    sum(
                        segment.duration
                        for segment in interval.delayed_command_segments
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
