import unittest

import numpy as np

from grape_param_estim.controller_replay import (
    ACCELERATION_CONTROL,
    ControllerFeedback,
    ControllerParameters,
    ControllerReference,
    ControllerReplay,
    ParameterChange,
    PidLimits,
    ReplayRequest,
    VectorPidSurrogate,
    replay_metrics,
)


def parameters(p=2.0, i=0.5, d=0.25, output_limit=100.0):
    limits = PidLimits(
        output=output_limit,
        p_term=20.0,
        i_term=10.0,
        d_term=10.0,
        p_error=5.0,
        i_state=4.0,
        d_error=5.0,
    )
    return ControllerParameters(
        p_gain=np.full(6, p),
        i_gain=np.full(6, i),
        d_gain=np.full(6, d),
        limits=limits,
        controller_mass=2.0,
        controller_inertia_diagonal=[0.1, 0.2, 0.3],
        allocation_scale=np.ones(6),
        thrust_scale=0.8,
    )


class ControllerReplayTests(unittest.TestCase):
    def test_surrogate_matches_deployed_pid_clamp_and_reset_semantics(self):
        controller = VectorPidSurrogate(parameters())
        reference = ControllerReference(
            position=[2.0, 0.0, 1.0, 0.0, 0.0, -np.pi + 0.1],
            velocity=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            acceleration=[0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        feedback = ControllerFeedback(
            position=[0.0, 0.0, 2.0, 0.0, 0.0, np.pi - 0.1],
            velocity=np.zeros(6),
        )
        step = controller.step(reference, feedback, delta=0.1)
        # x: P=4, I=0.1, D=.25, FF=.2
        self.assertAlmostEqual(step.acceleration_command[0], 4.55)
        # z integrator is clamped nonnegative even for negative position error.
        self.assertEqual(step.integral_state[2], 0.0)
        # Yaw error follows shortest-angle semantics (+0.2 rad), not -2pi.
        self.assertAlmostEqual(step.p_error[5], 0.2)
        np.testing.assert_allclose(
            step.generalized_wrench_command[:3],
            step.acceleration_command[:3] * 2.0 / 0.8,
        )
        controller.reset()
        reset_step = controller.step(reference, feedback, delta=0.0)
        np.testing.assert_allclose(reset_step.integral_state, 0.0)

    def test_control_modes_and_integration_gate_are_applied_per_axis(self):
        controller = VectorPidSurrogate(parameters())
        reference = ControllerReference(np.ones(6), np.ones(6), np.full(6, 0.3))
        feedback = ControllerFeedback(np.zeros(6), np.zeros(6))
        modes = np.array([ACCELERATION_CONTROL] * 6)
        step = controller.step(
            reference,
            feedback,
            delta=0.2,
            integration_enabled=np.zeros(6, dtype=bool),
            control_mode=modes,
        )
        np.testing.assert_allclose(step.p_error, 0.0)
        np.testing.assert_allclose(step.d_error, 0.0)
        np.testing.assert_allclose(step.integral_state, 0.0)
        np.testing.assert_allclose(step.acceleration_command, 0.3)

    def test_teacher_forced_and_free_run_recompute_candidate_commands(self):
        timestamps = np.arange(0.0, 1.01, 0.1)
        count = timestamps.size
        reference_position = np.zeros((count, 6))
        reference_position[:, 0] = 1.0
        zeros = np.zeros((count, 6))
        request = ReplayRequest(
            timestamps=timestamps,
            reference_position=reference_position,
            reference_velocity=zeros,
            reference_acceleration=zeros,
            actual_position=zeros,
            actual_velocity=zeros,
        )
        replay = ControllerReplay()
        teacher = replay.run(request, parameters(p=1.0, i=0.0, d=0.0), "teacher_forced")
        np.testing.assert_allclose(teacher.acceleration_command[:, 0], 1.0)
        self.assertFalse(teacher.is_exact)
        self.assertIn("surrogate", teacher.backend_id)

        def double_integrator(position, velocity, acceleration, delta, _index):
            return (
                position + velocity * delta + 0.5 * acceleration * delta * delta,
                velocity + acceleration * delta,
            )

        slow = replay.run(
            request,
            parameters(p=1.0, i=0.0, d=0.0),
            "free_run",
            initial_position=np.zeros(6),
            initial_velocity=np.zeros(6),
            plant_step=double_integrator,
        )
        fast = replay.run(
            request,
            parameters(p=2.0, i=0.0, d=0.0),
            "free_run",
            initial_position=np.zeros(6),
            initial_velocity=np.zeros(6),
            plant_step=double_integrator,
        )
        self.assertGreater(fast.feedback_position[-1, 0], slow.feedback_position[-1, 0])
        self.assertFalse(
            np.allclose(fast.acceleration_command, slow.acceleration_command)
        )

    def test_replay_rejects_truthy_non_boolean_backend_exactness(self):
        timestamps = np.array([0.0, 0.1])
        zeros = np.zeros((timestamps.size, 6))
        request = ReplayRequest(
            timestamps=timestamps,
            reference_position=zeros,
            reference_velocity=zeros,
            reference_acceleration=zeros,
            actual_position=zeros,
            actual_velocity=zeros,
        )
        for invalid in ("false", 0, np.bool_(False)):
            with self.subTest(invalid=repr(invalid)):
                class NonBooleanExactSurrogate(VectorPidSurrogate):
                    pass

                NonBooleanExactSurrogate.is_exact = invalid
                with self.assertRaisesRegex(TypeError, "is_exact"):
                    ControllerReplay(
                        backend_factory=NonBooleanExactSurrogate
                    ).run(
                        request,
                        parameters(),
                        "teacher_forced",
                    )

    def test_gain_schedule_keeps_integrator_but_changes_subsequent_output(self):
        timestamps = np.arange(0.0, 0.51, 0.1)
        count = timestamps.size
        reference = np.ones((count, 6))
        zeros = np.zeros((count, 6))
        request = ReplayRequest(
            timestamps=timestamps,
            reference_position=reference,
            reference_velocity=zeros,
            reference_acceleration=zeros,
            actual_position=zeros,
            actual_velocity=zeros,
        )
        initial = parameters(p=1.0, i=1.0, d=0.0)
        changed = parameters(p=3.0, i=1.0, d=0.0)
        result = ControllerReplay().run(
            request,
            initial,
            "teacher_forced",
            parameter_changes=(ParameterChange(0.3, changed),),
        )
        self.assertGreater(result.integral_state[3, 0], result.integral_state[2, 0])
        self.assertGreater(result.acceleration_command[3, 0], result.acceleration_command[2, 0])
        self.assertFalse(np.any(result.reset_applied))

    def test_replay_gate_requires_continuous_and_discrete_agreement(self):
        recorded = np.column_stack((np.linspace(0.0, 1.0, 101), np.linspace(1.0, 2.0, 101)))
        close = recorded + 1.0e-4
        events = np.arange(101) % 3
        passed = replay_metrics(close, recorded, events, events)
        self.assertTrue(passed.passed)
        failed = replay_metrics(recorded + 0.2, recorded, events, np.roll(events, 1))
        self.assertFalse(failed.passed)
        self.assertLess(failed.event_agreement, 1.0)

    def test_delay_compensation_advances_known_reference_and_changes_command(self):
        timestamps = np.arange(0.0, 1.01, 0.1)
        count = timestamps.size
        reference = np.zeros((count, 6))
        reference[:, 0] = timestamps
        zeros = np.zeros((count, 6))
        request = ReplayRequest(
            timestamps=timestamps,
            reference_position=reference,
            reference_velocity=zeros,
            reference_acceleration=zeros,
            actual_position=zeros,
            actual_velocity=zeros,
        )
        no_compensation = parameters(p=1.0, i=0.0, d=0.0)
        compensated = ControllerParameters(
            p_gain=no_compensation.p_gain,
            i_gain=no_compensation.i_gain,
            d_gain=no_compensation.d_gain,
            limits=no_compensation.limits,
            controller_mass=no_compensation.controller_mass,
            controller_inertia_diagonal=no_compensation.controller_inertia_diagonal,
            allocation_scale=no_compensation.allocation_scale,
            thrust_scale=no_compensation.thrust_scale,
            delay_compensation_s=0.2,
        )
        baseline = ControllerReplay().run(
            request, no_compensation, "teacher_forced"
        )
        advanced = ControllerReplay().run(
            request, compensated, "teacher_forced"
        )
        self.assertAlmostEqual(baseline.acceleration_command[3, 0], 0.3)
        self.assertAlmostEqual(advanced.acceleration_command[3, 0], 0.5)
        self.assertGreater(
            advanced.acceleration_command[3, 0],
            baseline.acceleration_command[3, 0],
        )


if __name__ == "__main__":
    unittest.main()
