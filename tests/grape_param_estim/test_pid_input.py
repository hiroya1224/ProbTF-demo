import unittest
from pathlib import Path

import numpy as np

from grape_param_estim.batch_artifact import BatchEstimationRun
from grape_param_estim.controller import ControllerConfig
from grape_param_estim.pid.input import (
    PidBagForecastModel,
    forecast_scenarios_from_batch_run,
    physical_posterior_from_batch_run,
)
from grape_param_estim.system import (
    ActuatorParameters,
    GrapeGeometry,
    VehicleParameters,
)


def _batch_run(
    source_mode_id=("mode-map", "mode-map"), *, include_trajectory=True
):
    nominal = VehicleParameters.nominal()
    mcmc = {
        "sample_id": np.asarray(("chain-a:1", "chain-b:1")),
        "source_mode_id": np.asarray(source_mode_id),
        "mass": np.asarray((nominal.mass, 1.1 * nominal.mass)),
        "inertia": np.asarray((nominal.inertia, nominal.inertia)),
        "cog": np.asarray((nominal.cog_offset, nominal.cog_offset)),
        "force_effectiveness": np.asarray(
            (nominal.force_effectiveness, nominal.force_effectiveness)
        ),
        "torque_effectiveness": np.asarray(
            (nominal.torque_effectiveness, nominal.torque_effectiveness)
        ),
        "delay": np.asarray((0.01, 0.02)),
    }
    zeros3 = np.zeros((3, 3))
    quaternions = np.tile((0.0, 0.0, 0.0, 1.0), (3, 1))
    bag = {
        "knot_time": np.asarray((0.0, 0.1, 0.2)),
        "reference_time": np.asarray((0.0, 0.05, 0.15)),
        "reference_position": zeros3,
        "reference_linear_velocity": zeros3,
        "reference_linear_acceleration": zeros3,
        "reference_rpy": zeros3,
        "reference_angular_velocity": zeros3,
        "reference_angular_acceleration": zeros3,
        "map_position": zeros3,
        "map_orientation_xyzw": quaternions,
        "map_linear_velocity": zeros3,
        "map_angular_velocity": zeros3,
        "map_controller_integral": np.zeros((3, 6)),
        "map_actuator_thrust": np.ones((3, 4)),
        "map_actuator_gimbal": np.zeros((3, 4)),
    }
    conditional_position = np.zeros((1, 3, 3))
    conditional_position[0, 0, 0] = 1.0
    trajectory = {
        "sample_id": np.asarray(("chain-a:1",)),
        "conditional_position": conditional_position,
        "conditional_orientation_xyzw": quaternions[None, :, :],
        "conditional_linear_velocity": np.zeros((1, 3, 3)),
        "conditional_angular_velocity": np.zeros((1, 3, 3)),
        "conditional_controller_integral": np.zeros((1, 3, 6)),
        "conditional_actuator_thrust": np.ones((1, 3, 4)),
        "conditional_actuator_gimbal": np.zeros((1, 3, 4)),
    }
    return BatchEstimationRun(
        root=Path("/unused"),
        manifest={"selected_bag_ids": ["bag-a"], "run_id": "batch-run"},
        map_static={},
        q_em={},
        laplace={},
        diagnostics={},
        bags={"bag-a": bag},
        mcmc_samples=mcmc,
        trajectories={"bag-a": trajectory} if include_trajectory else {},
    )


class PidInputTests(unittest.TestCase):
    def setUp(self):
        self.run = _batch_run()
        nominal = VehicleParameters.nominal()
        self.model = PidBagForecastModel(
            bag_id="bag-a",
            controller_configuration=ControllerConfig.grape(),
            controller_nominal_parameters=nominal,
            controller_geometry=GrapeGeometry.grape(),
            plant_geometry=GrapeGeometry.grape(),
            actuator_parameters=ActuatorParameters(
                thrust_time_constant=0.03,
                gimbal_time_constant=0.02,
                delay=0.0,
            ),
            roll_pitch_integration_active=True,
            maximum_reference_age_seconds=0.06,
        )

    def test_batch_mcmc_samples_become_equal_weight_physical_plants(self):
        posterior = physical_posterior_from_batch_run(
            self.run,
            fixed_linear_drag=(0.1, 0.2, 0.3),
            fixed_angular_drag=(0.01, 0.02, 0.03),
        )
        self.assertEqual(tuple(posterior.sample_id), ("chain-a:1", "chain-b:1"))
        self.assertEqual(posterior.equal_weight, 0.5)
        np.testing.assert_array_equal(
            posterior.samples[1].parameters.linear_drag, (0.1, 0.2, 0.3)
        )
        self.assertEqual(posterior.samples[1].delay, 0.02)

    def test_multiple_modes_require_explicit_selection(self):
        run = _batch_run(("mode-a", "mode-b"))
        with self.assertRaisesRegex(ValueError, "selected_mode_id"):
            physical_posterior_from_batch_run(
                run,
                fixed_linear_drag=np.zeros(3),
                fixed_angular_drag=np.zeros(3),
            )
        selected = physical_posterior_from_batch_run(
            run,
            fixed_linear_drag=np.zeros(3),
            fixed_angular_drag=np.zeros(3),
            selected_mode_id="mode-b",
        )
        self.assertEqual(tuple(selected.sample_id), ("chain-b:1",))

    def test_scenarios_share_selected_mode_map_initial_with_or_without_trajectories(
        self,
    ):
        posterior = physical_posterior_from_batch_run(
            self.run,
            fixed_linear_drag=np.zeros(3),
            fixed_angular_drag=np.zeros(3),
        )
        stored = forecast_scenarios_from_batch_run(
            self.run, posterior, (self.model,)
        )
        absent_run = _batch_run(include_trajectory=False)
        absent = forecast_scenarios_from_batch_run(
            absent_run, posterior, (self.model,)
        )
        self.assertEqual(len(stored), 1)
        self.assertEqual(len(absent), 1)
        shared = stored[0].initial_condition(posterior.sample_id[0])
        for sample_id in posterior.sample_id:
            with_trajectory = stored[0].initial_condition(sample_id)
            without_trajectory = absent[0].initial_condition(sample_id)
            self.assertEqual(
                with_trajectory.source,
                "shared_selected_mode_map_initial",
            )
            self.assertEqual(without_trajectory.source, with_trajectory.source)
            np.testing.assert_array_equal(
                with_trajectory.rigid_body_state.as_vector(),
                without_trajectory.rigid_body_state.as_vector(),
            )
            np.testing.assert_array_equal(
                with_trajectory.rigid_body_state.as_vector(),
                shared.rigid_body_state.as_vector(),
            )
            np.testing.assert_array_equal(
                with_trajectory.controller_state.integral_error,
                without_trajectory.controller_state.integral_error,
            )
            np.testing.assert_array_equal(
                with_trajectory.controller_state.integral_error,
                shared.controller_state.integral_error,
            )
            np.testing.assert_array_equal(
                with_trajectory.actuator_state.thrust,
                without_trajectory.actuator_state.thrust,
            )
            np.testing.assert_array_equal(
                with_trajectory.actuator_state.thrust,
                shared.actuator_state.thrust,
            )
            np.testing.assert_array_equal(
                with_trajectory.actuator_state.gimbal_angle,
                without_trajectory.actuator_state.gimbal_angle,
            )
            np.testing.assert_array_equal(
                with_trajectory.actuator_state.gimbal_angle,
                shared.actuator_state.gimbal_angle,
            )
            self.assertEqual(with_trajectory.rigid_body_state.position[0], 0.0)
        self.assertEqual(
            dict(stored[0].provenance)["initial_condition_policy"],
            "shared_selected_mode_map_initial",
        )
        self.assertEqual(len(stored[0].references), stored[0].times.size)

    def test_reference_age_and_bag_model_alignment_are_strict(self):
        posterior = physical_posterior_from_batch_run(
            self.run,
            fixed_linear_drag=np.zeros(3),
            fixed_angular_drag=np.zeros(3),
        )
        too_short = PidBagForecastModel(
            bag_id="bag-a",
            controller_configuration=self.model.controller_configuration,
            controller_nominal_parameters=self.model.controller_nominal_parameters,
            controller_geometry=self.model.controller_geometry,
            plant_geometry=self.model.plant_geometry,
            actuator_parameters=self.model.actuator_parameters,
            roll_pitch_integration_active=True,
            maximum_reference_age_seconds=0.01,
        )
        with self.assertRaisesRegex(ValueError, "maximum causal age"):
            forecast_scenarios_from_batch_run(
                self.run, posterior, (too_short,)
            )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            forecast_scenarios_from_batch_run(self.run, posterior, tuple())


if __name__ == "__main__":
    unittest.main()
