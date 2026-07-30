import unittest

import numpy as np

from grape_param_estim.app import _fit_rows, _result_bag_view
from grape_param_estim.estimator import relative_transform_from_poses
from grape_param_estim.model import GrapeRigidBodyModel, replay_segments
from test_estimator import generated_analysis
from test_model import nominal_parameters


class PhaseTwoResultViewTest(unittest.TestCase):
    def test_schema_one_result_uses_real_map_trajectory_as_estimated_nominal(
        self,
    ):
        truth = {
            "mass_scale": 1.2,
            "force_scale": 0.9,
            "inertia_scale": 1.4,
            "torque_scale": 0.7,
        }
        analysis = generated_analysis(truth)
        model = GrapeRigidBodyModel(maximum_time_step=0.02)
        baseline = replay_segments(
            analysis, model, nominal_parameters()
        )
        estimated = replay_segments(
            analysis, model, nominal_parameters(**truth)
        )
        posterior_position = np.stack(
            (baseline.position, estimated.position, baseline.position)
        )
        posterior_orientation = np.stack(
            (
                baseline.orientation_xyzw,
                estimated.orientation_xyzw,
                baseline.orientation_xyzw,
            )
        )
        baseline_delta_translation, baseline_delta_rotation = (
            relative_transform_from_poses(
                baseline.position,
                baseline.orientation_xyzw,
                posterior_position,
                posterior_orientation,
            )
        )
        payload = {
            "schema_version": np.asarray((1,)),
            "particles": np.asarray(
                (
                    (1.0, 1.0, 1.0, 1.0),
                    (1.2, 0.9, 1.4, 0.7),
                    (1.1, 1.1, 1.1, 1.1),
                )
            ),
            "weights": np.asarray((0.1, 0.8, 0.1)),
            "bag_0_times": analysis.times,
            "bag_0_segment_id": analysis.segment_id,
            "bag_0_observed_position": analysis.position,
            "bag_0_observed_orientation_xyzw": (
                analysis.orientation_xyzw
            ),
            "bag_0_nominal_position": baseline.position,
            "bag_0_nominal_orientation_xyzw": (
                baseline.orientation_xyzw
            ),
            "bag_0_posterior_position": posterior_position,
            "bag_0_posterior_orientation_xyzw": posterior_orientation,
            # Schema 1 stored baseline-relative transforms under these names.
            "bag_0_delta_translation": baseline_delta_translation,
            "bag_0_delta_rotation_vector": baseline_delta_rotation,
        }

        view = _result_bag_view(payload, 0)
        rows = _fit_rows(view, payload["weights"], 10.0, 1.0)

        self.assertEqual(view["estimated_particle_index"], 1)
        np.testing.assert_allclose(
            view["estimated_position"], estimated.position
        )
        np.testing.assert_allclose(
            view["delta_translation"][1], 0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            view["delta_rotation_vector"][1], 0.0, atol=1.0e-12
        )
        self.assertLess(
            rows[1]["translation RMS [m]"],
            rows[0]["translation RMS [m]"],
        )
        self.assertLess(
            rows[1]["rotation RMS [deg]"],
            rows[0]["rotation RMS [deg]"],
        )


if __name__ == "__main__":
    unittest.main()
