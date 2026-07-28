import unittest

import numpy as np

from grape_param_estim.forward.rollout import CommandSample, RolloutResult
from grape_param_estim.inference import (
    ControllerEventObservations,
    EpisodeLikelihood,
    LikelihoodConfig,
    ObservationDataset,
)
from grape_param_estim.plant.actuator import RealizedWrench
from grape_param_estim.plant.parameters import PlantHypothesis
from grape_param_estim.plant.rigid_body import PlantState
from grape_param_estim.plant.sensor import PredictedObservation


def _command(stamp, events=(), saturated=False):
    return CommandSample(
        stamp=stamp,
        base_thrust=np.ones(4),
        gimbal_angle=np.zeros(4),
        events=events,
        saturated=saturated,
    )


def _rollout(commands):
    timestamps = np.asarray([0.0, 1.0, 2.0])
    states = tuple(
        PlantState(
            stamp=stamp,
            position_world=np.zeros(3),
            velocity_world=np.zeros(3),
            orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
            angular_velocity_body=np.zeros(3),
        )
        for stamp in timestamps
    )
    predictions = tuple(
        PredictedObservation(
            stamp=stamp,
            position_world=np.zeros(3),
            orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
            velocity_world=np.zeros(3),
            angular_velocity_body=np.zeros(3),
            specific_force_body=np.zeros(3),
            model_id="likelihood-event-test/v1",
        )
        for stamp in timestamps
    )
    wrenches = tuple(
        RealizedWrench(
            stamp=stamp,
            force_body=np.zeros(3),
            torque_body=np.zeros(3),
            actuator_state=np.zeros(0),
            saturated=False,
            calibrated_wrench=False,
            model_id="likelihood-event-test/v1",
        )
        for stamp in timestamps[1:]
    )
    hypothesis = PlantHypothesis(
        model_id="likelihood-event-test/v1",
        plant_parameters=np.asarray([1.0]),
        actuator_parameters=np.empty(0),
        disturbance_parameters=np.empty(0),
        actuator_parameter_names=(),
    )
    return RolloutResult(
        mode="closed_loop_plant_identification",
        model_id="likelihood-event-test/v1",
        hypothesis=hypothesis,
        integration_timestamps=timestamps,
        plant_states=states,
        commands=tuple(commands),
        realized_wrenches=wrenches,
        predicted_observations=predictions,
        controller_tick_timestamps=timestamps,
        likelihood_timestamps=timestamps,
        events=(),
        used_recorded_commands=False,
        controller_fidelity="pc_exact",
    )


def _observations(event_observations=None, failure_time=None):
    timestamps = np.asarray([0.0, 1.0, 2.0])
    return ObservationDataset(
        episode_id="failure-events",
        role="inference_failure",
        timestamps=timestamps,
        position_world=np.zeros((3, 3)),
        orientation_xyzw=np.tile(
            np.asarray([0.0, 0.0, 0.0, 1.0]), (3, 1)
        ),
        failure_time=failure_time,
        event_observations=event_observations,
    )


class ControllerEventObservationTest(unittest.TestCase):
    def test_schema_separates_saturation_mode_and_other_event_bits(self):
        evidence = ControllerEventObservations(
            timestamps=np.asarray([0.0, 1.0, 2.0]),
            event_bitmasks=np.asarray([0, 4 | 32, 1], dtype=np.uint32),
        )
        np.testing.assert_array_equal(
            evidence.saturated, [False, True, False]
        )
        np.testing.assert_array_equal(
            evidence.mode_event_bitmasks, [0, 4, 0]
        )
        np.testing.assert_array_equal(
            evidence.other_event_bitmasks, [0, 0, 1]
        )
        with self.assertRaises(ValueError):
            evidence.event_bitmasks[0] = 1
        with self.assertRaisesRegex(
            ValueError, "saturation observations disagree"
        ):
            ControllerEventObservations(
                timestamps=np.asarray([0.0, 1.0]),
                event_bitmasks=np.asarray([0, 32], dtype=np.uint32),
                saturated=np.asarray([False, False]),
            )

    def test_event_frames_contribute_and_report_separate_totals(self):
        evidence = ControllerEventObservations(
            timestamps=np.asarray([0.0, 1.0, 2.0]),
            event_bitmasks=np.asarray([0, 4 | 32, 1], dtype=np.uint32),
        )
        matching = _rollout(
            (
                _command(0.0),
                _command(1.0, events=(4, 32), saturated=True),
                _command(2.0, events=(1,)),
            )
        )
        mismatching = _rollout(
            (
                _command(0.0),
                _command(1.0, events=(8,)),
                _command(2.0),
            )
        )
        likelihood = EpisodeLikelihood(
            LikelihoodConfig(
                saturation_event_error_probability=0.1,
                mode_event_error_probability=0.1,
                other_event_error_probability=0.1,
                require_controller_event_evidence=True,
            )
        )
        observations = _observations(evidence)
        match_components = likelihood.evaluate(matching, observations)
        mismatch_components = likelihood.evaluate(
            mismatching, observations
        )

        self.assertLess(match_components.saturation_mode_event, 0.0)
        self.assertLess(
            mismatch_components.saturation_mode_event,
            match_components.saturation_mode_event,
        )
        self.assertAlmostEqual(
            match_components.trajectory_total,
            sum(
                (
                    match_components.pose,
                    match_components.orientation,
                    match_components.velocity,
                    match_components.imu,
                    match_components.angular_velocity,
                    match_components.command,
                )
            ),
        )
        self.assertAlmostEqual(
            match_components.event_total,
            match_components.failure_event
            + match_components.saturation_mode_event,
        )
        self.assertAlmostEqual(
            match_components.total,
            match_components.trajectory_total
            + match_components.event_total,
        )
        self.assertEqual(match_components.scored_event_sample_count, 3)
        self.assertEqual(
            match_components.controller_event_evidence_status,
            "scored",
        )
        diagnostics = match_components.diagnostics[
            "controller_event_likelihood"
        ]
        self.assertEqual(diagnostics["status"], "scored")
        self.assertEqual(
            diagnostics["saturation"]["mismatch_count"], 0
        )
        self.assertEqual(diagnostics["mode_event"]["mismatch_count"], 0)
        self.assertEqual(diagnostics["other_event"]["mismatch_count"], 0)
        mismatch_diagnostics = mismatch_components.diagnostics[
            "controller_event_likelihood"
        ]
        self.assertGreater(
            mismatch_diagnostics["saturation"]["mismatch_count"], 0
        )
        self.assertGreater(
            mismatch_diagnostics["mode_event"]["mismatch_count"], 0
        )
        self.assertGreater(
            mismatch_diagnostics["other_event"]["mismatch_count"], 0
        )

    def test_absent_evidence_is_explicit_and_can_be_required(self):
        rollout = _rollout(
            (_command(0.0), _command(1.0), _command(2.0))
        )
        legacy = EpisodeLikelihood().evaluate(
            rollout, _observations()
        )
        self.assertEqual(legacy.saturation_mode_event, 0.0)
        self.assertEqual(legacy.scored_event_sample_count, 0)
        self.assertEqual(
            legacy.controller_event_evidence_status,
            "not_scored_no_evidence",
        )
        self.assertEqual(
            legacy.diagnostics["controller_event_likelihood"]["status"],
            "not_scored_no_evidence",
        )
        required = EpisodeLikelihood(
            LikelihoodConfig(require_controller_event_evidence=True)
        )
        with self.assertRaisesRegex(
            ValueError, "requires factual event evidence"
        ):
            required.evaluate(rollout, _observations())

    def test_event_scoring_fails_closed_without_exact_prediction(self):
        off_grid = ControllerEventObservations(
            timestamps=np.asarray([0.0, 0.5]),
            event_bitmasks=np.asarray([0, 0], dtype=np.uint32),
        )
        rollout = _rollout(
            (_command(0.0), _command(1.0), _command(2.0))
        )
        with self.assertRaisesRegex(
            ValueError, "lacks exact predicted tick coverage"
        ):
            EpisodeLikelihood().evaluate(
                rollout, _observations(off_grid)
            )

        inconsistent_prediction = _rollout(
            (
                _command(0.0),
                _command(1.0, saturated=True),
                _command(2.0),
            )
        )
        evidence = ControllerEventObservations(
            timestamps=np.asarray([0.0, 1.0, 2.0]),
            event_bitmasks=np.asarray([0, 32, 0], dtype=np.uint32),
        )
        with self.assertRaisesRegex(
            ValueError, "predicted saturation flag"
        ):
            EpisodeLikelihood().evaluate(
                inconsistent_prediction, _observations(evidence)
            )

    def test_event_frames_follow_failure_censoring(self):
        evidence = ControllerEventObservations(
            timestamps=np.asarray([0.0, 1.0, 2.0]),
            event_bitmasks=np.asarray([0, 0, 0], dtype=np.uint32),
        )
        rollout = _rollout(
            (_command(0.0), _command(1.0), _command(2.0))
        )
        components = EpisodeLikelihood().evaluate(
            rollout, _observations(evidence, failure_time=1.0)
        )
        self.assertEqual(components.scored_event_sample_count, 2)
        self.assertEqual(components.censored_event_sample_count, 1)
        diagnostics = components.diagnostics[
            "controller_event_likelihood"
        ]
        self.assertEqual(diagnostics["scored_frame_count"], 2)
        self.assertEqual(diagnostics["censored_frame_count"], 1)


if __name__ == "__main__":
    unittest.main()
