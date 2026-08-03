import multiprocessing
import unittest

import numpy as np

from grape_param_estim.articulated import GrapeArticulatedModel
from grape_param_estim.closed_loop_stepper import (
    ClosedLoopStepper,
    ClosedLoopStepperState,
)
from grape_param_estim.controller import (
    ControllerConfig,
    GrapeController,
    initial_controller_state,
)
from grape_param_estim.dynamics import FullSixDofPlant
from grape_param_estim.filter_state import GrapeFilterState
from grape_param_estim.geometry import (
    euler_xyz_to_matrix,
    matrix_to_quaternion,
)
from grape_param_estim.parallel_stepper import PersistentParallelSteppers
from grape_param_estim.progress import CancellationToken, ProgressCancelled
from grape_param_estim.synthetic import full_six_dof_reference
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    GrapeGeometry,
    RigidBodyState,
    VehicleParameters,
)


class PersistentParallelSteppersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times = np.asarray((0.0, 0.04, 0.11))
        cls.references = full_six_dof_reference(cls.times)
        cls.parameters = VehicleParameters.nominal()
        cls.geometry = GrapeGeometry.grape()
        cls.configuration = ControllerConfig.grape()
        cls.actuator_parameters = ActuatorParameters(delay=0.065)
        first = cls.references[0]
        rigid = RigidBodyState(
            first.position,
            matrix_to_quaternion(euler_xyz_to_matrix(first.rpy)),
            first.linear_velocity,
            first.angular_velocity,
        )
        controller = initial_controller_state(
            cls.configuration, trim_hover=True
        )
        cls.initial_states = tuple(
            GrapeFilterState(
                rigid=rigid,
                controller=controller,
                actuator=ActuatorState(
                    np.asarray((4.0, 4.1, 4.2, 4.3)) + member * 0.01,
                    np.asarray((0.01, -0.02, 0.03, -0.04)),
                ),
                residual_wrench=np.ones(6) * member * 0.01,
            )
            for member in range(4)
        )

    def _controller(self):
        return GrapeController(
            self.configuration,
            self.parameters,
            self.geometry,
            articulated_model=GrapeArticulatedModel(),
        )

    def _parallel(self):
        return PersistentParallelSteppers(
            controller=self._controller(),
            plant=FullSixDofPlant(self.parameters, self.geometry),
            actuator_parameters=self.actuator_parameters,
            initial_time=self.times[0],
            initial_states=self.initial_states,
            worker_count=2,
        )

    def test_persistent_parallel_intervals_are_bit_exact_and_causal(self):
        local = tuple(
            ClosedLoopStepper(
                self._controller(),
                FullSixDofPlant(self.parameters, self.geometry),
                self.actuator_parameters,
                ClosedLoopStepperState(
                    self.times[0],
                    state.rigid,
                    state.controller,
                    state.actuator,
                ),
            )
            for state in self.initial_states
        )
        analysis = self.initial_states
        parallel = self._parallel()
        pids = parallel.worker_pids
        self.assertEqual(len(pids), 2)
        try:
            for index in range(self.times.size - 1):
                wrench = np.asarray(
                    [
                        state.residual_wrench + (index + 1) * 0.02
                        for state in analysis
                    ]
                )
                expected = []
                for member, stepper in enumerate(local):
                    state = analysis[member]
                    stepper.replace_dynamic_state(
                        rigid_body_state=state.rigid,
                        controller_state=state.controller,
                        actuator_state=state.actuator,
                    )
                    stepper.advance_interval(
                        self.times[index + 1],
                        self.references[index],
                        wrench[member],
                    )
                    expected.append(stepper.state)
                actual = parallel.advance_interval(
                    end_time=self.times[index + 1],
                    reference=self.references[index],
                    analysis_states=analysis,
                    interval_wrench=wrench,
                )
                for left, right in zip(expected, actual):
                    self.assertEqual(left.time, right.time)
                    for field in (
                        "position",
                        "orientation_xyzw",
                        "linear_velocity",
                        "angular_velocity",
                    ):
                        np.testing.assert_array_equal(
                            getattr(left.rigid_body_state, field),
                            getattr(right.rigid_body_state, field),
                        )
                    np.testing.assert_array_equal(
                        left.controller_state.integral_error,
                        right.controller_state.integral_error,
                    )
                    np.testing.assert_array_equal(
                        left.actuator_state.thrust,
                        right.actuator_state.thrust,
                    )
                    np.testing.assert_array_equal(
                        left.actuator_state.gimbal_angle,
                        right.actuator_state.gimbal_angle,
                    )
                analysis = tuple(
                    GrapeFilterState(
                        state.rigid_body_state,
                        state.controller_state,
                        state.actuator_state,
                        wrench[member],
                    )
                    for member, state in enumerate(actual)
                )
            for history in parallel.command_issue_times():
                np.testing.assert_array_equal(history, self.times[:-1])
        finally:
            parallel.close()
        self.assertTrue(parallel.closed)
        active_pids = {
            child.pid for child in multiprocessing.active_children()
        }
        self.assertTrue(set(pids).isdisjoint(active_pids))

    def test_cancellation_aborts_workers_and_validation_is_strict(self):
        parallel = self._parallel()
        cancellation = CancellationToken()
        cancellation.cancel("unit_test")
        with self.assertRaises(ProgressCancelled):
            parallel.advance_interval(
                end_time=self.times[1],
                reference=self.references[0],
                analysis_states=self.initial_states,
                interval_wrench=np.zeros((4, 6)),
                cancellation_token=cancellation,
            )
        self.assertTrue(parallel.closed)

        with self.assertRaisesRegex(ValueError, "at least two"):
            PersistentParallelSteppers(
                controller=self._controller(),
                plant=FullSixDofPlant(self.parameters, self.geometry),
                actuator_parameters=self.actuator_parameters,
                initial_time=0.0,
                initial_states=self.initial_states,
                worker_count=1,
            )


if __name__ == "__main__":
    unittest.main()
