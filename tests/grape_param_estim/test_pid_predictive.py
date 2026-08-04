from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from grape_param_estim.closed_loop_stepper import ClosedLoopStepper
from grape_param_estim.controller import ControllerConfig, initial_controller_state
from grape_param_estim.controller_config import PidGainConfiguration
from grape_param_estim.pid.particle_search import (
    BODY_WRENCH_MODEL_DISCREPANCY,
    SAMPLE_MODEL_DISCREPANCY,
    ZERO_MODEL_DISCREPANCY,
    ModelDiscrepancyConfiguration,
    SPECIFIC_ACCELERATION_MODEL_DISCREPANCY,
    evaluate_pid_candidates,
)
from grape_param_estim.pid.predictive import (
    ClosedLoopPidForecastEvaluator,
    PidForecastInitialCondition,
    PidForecastScenario,
    run_pid_forecast,
)
from grape_param_estim.pid.proposal import (
    PhysicalPlantPosterior,
    current_pid_candidate,
    derive_pid_proposals,
    sample_pid_candidate,
)
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    GrapeGeometry,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


def _gain_configuration(controller):
    return PidGainConfiguration(
        np.asarray(
            (
                (
                    controller.pid[0].p_gain,
                    controller.pid[0].i_gain,
                    controller.pid[0].d_gain,
                ),
                (
                    controller.pid[2].p_gain,
                    controller.pid[2].i_gain,
                    controller.pid[2].d_gain,
                ),
                (
                    controller.pid[3].p_gain,
                    controller.pid[3].i_gain,
                    controller.pid[3].d_gain,
                ),
                (
                    controller.pid[5].p_gain,
                    controller.pid[5].i_gain,
                    controller.pid[5].d_gain,
                ),
            )
        )
    )


def _reference():
    return ReferenceState(
        position=np.zeros(3),
        linear_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        rpy=np.zeros(3),
        angular_velocity=np.zeros(3),
        angular_acceleration=np.zeros(3),
    )


class PidPredictiveTests(unittest.TestCase):
    def setUp(self):
        self.parameters = VehicleParameters.nominal()
        self.geometry = GrapeGeometry.grape()
        self.controller = ControllerConfig.grape()
        self.current = _gain_configuration(self.controller)
        self.posterior = PhysicalPlantPosterior.from_aligned_values(
            ("chain-a:000000", "chain-b:000000"),
            (self.parameters, self.parameters),
            (0.0, 0.012),
            ("mode-map", "mode-map"),
        )
        initial_state = RigidBodyState(
            position=np.zeros(3),
            orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )
        self.scenario = PidForecastScenario(
            bag_id="failure-bag",
            times=np.asarray((18.0, 18.02, 18.04, 18.06)),
            references=(_reference(),) * 4,
            initial_conditions=tuple(
                PidForecastInitialCondition(
                    sample_id=sample.sample_id,
                    rigid_body_state=initial_state,
                    controller_state=initial_controller_state(self.controller),
                    actuator_state=None,
                    source="shared_map_initial",
                )
                for sample in self.posterior.samples
            ),
            controller_configuration=self.controller,
            controller_nominal_parameters=self.parameters,
            controller_geometry=self.geometry,
            plant_geometry=self.geometry,
            actuator_parameters=ActuatorParameters(),
            provenance=(("interval", "18.0--18.06 s"),),
        )
        self.candidate = current_pid_candidate(self.current)

    def test_zero_discrepancy_nominal_hover_completes_without_state_resets(self):
        realization = ModelDiscrepancyConfiguration(
            ZERO_MODEL_DISCREPANCY,
            np.ones(6),
            base_seed=41,
            residual_quantity=BODY_WRENCH_MODEL_DISCREPANCY,
        ).realization("chain-a:000000", "failure-bag", 0)
        outcome = run_pid_forecast(
            self.candidate,
            self.posterior.samples[0],
            self.scenario,
            realization,
        )
        self.assertTrue(outcome.trace.completed)
        self.assertEqual(outcome.trace.times.size, self.scenario.times.size)
        self.assertEqual(outcome.metrics.forecast_completion, 1.0)
        self.assertEqual(outcome.metrics.numerical_failure_count, 0)
        self.assertLess(outcome.metrics.position_rmse, 1.0e-12)
        self.assertLess(outcome.metrics.orientation_rmse, 1.0e-12)

    def test_sampled_q_is_fresh_repeatable_and_distinct_from_zero(self):
        sampled_configuration = ModelDiscrepancyConfiguration(
            SAMPLE_MODEL_DISCREPANCY,
            np.asarray((0.3, 0.2, 0.1, 0.02, 0.03, 0.01)),
            base_seed=90210,
            residual_quantity=BODY_WRENCH_MODEL_DISCREPANCY,
        )
        sampled = sampled_configuration.realization(
            "chain-a:000000", "failure-bag", 0
        )
        first = run_pid_forecast(
            self.candidate,
            self.posterior.samples[0],
            self.scenario,
            sampled,
        )
        second = run_pid_forecast(
            self.candidate,
            self.posterior.samples[0],
            self.scenario,
            sampled,
        )
        zero = run_pid_forecast(
            self.candidate,
            self.posterior.samples[0],
            self.scenario,
            ModelDiscrepancyConfiguration(
                ZERO_MODEL_DISCREPANCY,
                sampled_configuration.diagonal_q,
                base_seed=90210,
                residual_quantity=BODY_WRENCH_MODEL_DISCREPANCY,
            ).realization("chain-a:000000", "failure-bag", 0),
        )
        np.testing.assert_array_equal(first.trace.position, second.trace.position)
        self.assertGreater(
            np.linalg.norm(first.trace.position - zero.trace.position), 0.0
        )

    def test_specific_acceleration_q_is_converted_per_physical_plant(self):
        parameters = replace(
            self.parameters,
            mass=2.0,
            inertia=np.diag((2.0, 3.0, 4.0)),
        )
        sample = PhysicalPlantPosterior.from_aligned_values(
            ("chain-a:000000",),
            (parameters,),
            (0.0,),
            ("mode-map",),
        ).samples[0]
        specific = ModelDiscrepancyConfiguration(
            SAMPLE_MODEL_DISCREPANCY,
            np.ones(6),
            base_seed=55,
            residual_quantity=SPECIFIC_ACCELERATION_MODEL_DISCREPANCY,
        ).realization("chain-a:000000", "failure-bag", 0)
        equivalent_wrench = ModelDiscrepancyConfiguration(
            SAMPLE_MODEL_DISCREPANCY,
            np.asarray((4.0, 4.0, 4.0, 4.0, 9.0, 16.0)),
            base_seed=55,
            residual_quantity=BODY_WRENCH_MODEL_DISCREPANCY,
        ).realization("chain-a:000000", "failure-bag", 0)
        first = run_pid_forecast(
            self.candidate, sample, self.scenario, specific
        )
        second = run_pid_forecast(
            self.candidate, sample, self.scenario, equivalent_wrench
        )
        np.testing.assert_allclose(
            first.trace.position, second.trace.position, rtol=0.0, atol=1.0e-15
        )
        np.testing.assert_allclose(
            first.trace.orientation_xyzw,
            second.trace.orientation_xyzw,
            rtol=0.0,
            atol=1.0e-15,
        )

    def test_cross_evaluator_covers_every_candidate_and_plant_sample(self):
        proposals = derive_pid_proposals(
            self.posterior, self.parameters, self.geometry, self.current
        )
        derived = sample_pid_candidate(proposals, "chain-a:000000")
        evaluator = ClosedLoopPidForecastEvaluator((self.scenario,))
        result = evaluate_pid_candidates(
            (derived,),
            self.posterior,
            evaluator.bag_ids,
            evaluator,
            self.current,
            ModelDiscrepancyConfiguration(
                ZERO_MODEL_DISCREPANCY,
                np.ones(6),
                base_seed=8,
                residual_quantity=BODY_WRENCH_MODEL_DISCREPANCY,
            ),
        )
        self.assertEqual(len(result.records), 4)
        self.assertEqual(
            {
                (record.candidate_id, record.sample_id)
                for record in result.records
            },
            {
                (candidate_id, sample_id)
                for candidate_id in ("current", derived.candidate_id)
                for sample_id in self.posterior.sample_id.tolist()
            },
        )
        self.assertTrue(
            all(record.metrics.forecast_completion == 1.0 for record in result.records)
        )

    def test_numerical_failure_returns_a_finite_partial_forecast(self):
        original = ClosedLoopStepper.advance_interval
        call_count = 0

        def fail_second_interval(stepper, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise FloatingPointError("synthetic integration failure")
            return original(stepper, *args, **kwargs)

        realization = ModelDiscrepancyConfiguration(
            ZERO_MODEL_DISCREPANCY,
            np.ones(6),
            base_seed=7,
            residual_quantity=BODY_WRENCH_MODEL_DISCREPANCY,
        ).realization("chain-a:000000", "failure-bag", 0)
        with patch.object(
            ClosedLoopStepper,
            "advance_interval",
            new=fail_second_interval,
        ):
            outcome = run_pid_forecast(
                self.candidate,
                self.posterior.samples[0],
                self.scenario,
                realization,
            )
        self.assertFalse(outcome.trace.completed)
        self.assertEqual(outcome.trace.completed_intervals, 1)
        self.assertAlmostEqual(outcome.metrics.forecast_completion, 1.0 / 3.0)
        self.assertEqual(outcome.metrics.numerical_failure_count, 1)
        self.assertIn("synthetic integration failure", outcome.trace.failure_reason)

    def test_actuator_limit_activity_is_reported_separately(self):
        saturated = PidForecastInitialCondition(
            sample_id="chain-a:000000",
            rigid_body_state=self.scenario.initial_conditions[0].rigid_body_state,
            controller_state=self.scenario.initial_conditions[0].controller_state,
            actuator_state=ActuatorState(
                thrust=np.full(4, ActuatorParameters().maximum_thrust),
                gimbal_angle=np.full(
                    4, ActuatorParameters().maximum_gimbal_angle
                ),
            ),
            source="synthetic_saturated_initial",
        )
        scenario = PidForecastScenario(
            bag_id=self.scenario.bag_id,
            times=self.scenario.times,
            references=self.scenario.references,
            initial_conditions=(saturated,),
            controller_configuration=self.controller,
            controller_nominal_parameters=self.parameters,
            controller_geometry=self.geometry,
            plant_geometry=self.geometry,
            actuator_parameters=ActuatorParameters(),
        )
        outcome = run_pid_forecast(
            self.candidate,
            self.posterior.samples[0],
            scenario,
            ModelDiscrepancyConfiguration(
                ZERO_MODEL_DISCREPANCY,
                np.ones(6),
                base_seed=1,
                residual_quantity=BODY_WRENCH_MODEL_DISCREPANCY,
            ).realization("chain-a:000000", "failure-bag", 0),
        )
        self.assertGreater(outcome.metrics.actuator_saturation_duration, 0.0)
        self.assertGreater(outcome.metrics.actuator_saturation_rate, 0.0)

    def test_missing_sample_aligned_initial_condition_is_rejected(self):
        evaluator = ClosedLoopPidForecastEvaluator((self.scenario,))
        missing = PhysicalPlantPosterior.from_aligned_values(
            ("absent-sample",),
            (self.parameters,),
            (0.0,),
            ("mode-map",),
        ).samples[0]
        with self.assertRaisesRegex(KeyError, "initial condition"):
            evaluator(
                self.candidate,
                missing,
                "failure-bag",
                ModelDiscrepancyConfiguration(
                    ZERO_MODEL_DISCREPANCY,
                    np.ones(6),
                    base_seed=1,
                    residual_quantity=BODY_WRENCH_MODEL_DISCREPANCY,
                ).realization("absent-sample", "failure-bag", 0),
            )


if __name__ == "__main__":
    unittest.main()
