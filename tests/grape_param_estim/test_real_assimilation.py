from types import SimpleNamespace
import unittest

import numpy as np

from grape_param_estim.controller import (
    ControllerConfig,
    initial_controller_state,
)
from grape_param_estim.real_assimilation import (
    assimilate_real_episode,
    build_real_strong_problem,
)
from grape_param_estim.synthetic import run_synthetic_experiment
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    VehicleParameters,
)


class RealAssimilationCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        synthetic = run_synthetic_experiment(
            duration=0.24,
            time_step=0.04,
            truth_parameters=VehicleParameters.nominal(),
            truth_actuators=ActuatorParameters(),
            truth_residual_wrench=lambda _time, _state: np.zeros(6),
            translation_noise=0.001,
            rotation_noise=0.001,
            seed=72,
        )
        configuration = ControllerConfig.grape()
        first = synthetic.nominal
        cls.episode = SimpleNamespace(
            observations=synthetic.observations,
            references=synthetic.references,
            controller_configuration=configuration,
            initial_controller_state=initial_controller_state(
                configuration, trim_hover=True
            ),
            initial_actuator_state=ActuatorState(
                first.actuator_thrust[0], first.actuator_gimbal_angle[0]
            ),
        )

    def test_nominal_problem_uses_pose_only_initial_anchor(self):
        problem, initial, nominal, _actuators, _parameters = (
            build_real_strong_problem(self.episode)
        )
        np.testing.assert_array_equal(
            initial.position, self.episode.observations.position[0]
        )
        np.testing.assert_array_equal(
            nominal.position[0], self.episode.observations.position[0]
        )
        np.testing.assert_array_equal(
            problem.initial_actuator_state.thrust,
            self.episode.initial_actuator_state.thrust,
        )

    def test_sparse_real_core_preserves_raw_member_alignment(self):
        result = assimilate_real_episode(
            self.episode,
            maximum_knots=2,
            maximum_iterations=1,
            seed=72,
        )
        posterior = result.posterior
        members = posterior.control_ensemble.shape[0]
        self.assertEqual(posterior.innovation_ensemble.shape, (members, 12))
        self.assertEqual(
            posterior.residual_wrench_ensemble.shape,
            (members, self.episode.observations.times.size - 1, 6),
        )
        self.assertEqual(
            posterior.correction_translation.shape[:2],
            (members, self.episode.observations.times.size),
        )
        self.assertEqual(result.mode_diagnostic.weights.tolist(), [1.0])
        self.assertEqual(
            result.mode_diagnostic.selected_mode_id,
            "actuator_wiring_nominal",
        )


if __name__ == "__main__":
    unittest.main()
