import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.artifact_io import ArtifactValidationError
from grape_param_estim.batch_request import (
    BATCH_ESTIMATION_REQUEST_SCHEMA,
    FIXED_FACTOR_COVARIANCE_BLOCKS,
    INITIAL_STATE_PRIOR_COVARIANCE_BLOCKS,
    OBSERVATION_FACTOR_NAMES,
    OBSERVATION_COVARIANCE_BLOCKS,
    load_batch_estimation_request,
    validate_batch_estimation_request,
)
from grape_param_estim.batch_artifact import file_sha256


class BatchRequestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.bag = root / "flight.bag"
        self.bag.write_bytes(b"test bag")
        self.output = root / "run"

        def diagonal_covariance(contract, source="project_configuration"):
            coordinates, units = contract
            return {
                "source": source,
                "representation": "diagonal",
                "coordinates": list(coordinates),
                "units": list(units),
                "values": [1.0] * len(coordinates),
            }

        factors = {}
        for name in OBSERVATION_FACTOR_NAMES:
            factors[name] = {
                "enabled": True,
                "disabled_reason": None,
                "covariances": {
                    block_name: diagonal_covariance(contract)
                    for block_name, contract in (
                        OBSERVATION_COVARIANCE_BLOCKS[name].items()
                    )
                },
            }
        factors["accelerometer"] = {
            "enabled": False,
            "disabled_reason": "physical sensor origin is not calibrated",
            "covariances": None,
        }
        fixed_covariances = {
            name: diagonal_covariance(contract, "numerical_tolerance")
            for name, contract in FIXED_FACTOR_COVARIANCE_BLOCKS.items()
        }
        initial_state_prior_covariances = {
            name: diagonal_covariance(contract, "project_configuration")
            for name, contract in (
                INITIAL_STATE_PRIOR_COVARIANCE_BLOCKS.items()
            )
        }
        self.request = {
            "schema": BATCH_ESTIMATION_REQUEST_SCHEMA,
            "run_id": "target-18-24",
            "run_mode": "estimate_and_sample",
            "resume": False,
            "output_directory": str(self.output),
            "bags": [
                {
                    "bag_id": "failure-04",
                    "path": str(self.bag),
                    "sha256": file_sha256(self.bag),
                    "interval_seconds": [18.0, 24.0],
                    "observation_factors": factors,
                    "fixed_factor_covariances": fixed_covariances,
                    "initial_state_prior_covariances": (
                        initial_state_prior_covariances
                    ),
                }
            ],
            "q": {
                "residual_quantity": "specific_acceleration",
                "interval_model": "continuous_spectral_density",
                "component_names": ["x", "y", "z", "roll", "pitch", "yaw"],
                "component_units": ["explicit-unit"] * 6,
                "initial_diagonal": [1.0] * 6,
                "floor_diagonal": [1.0e-8] * 6,
            },
            "parameter_prior": {
                "kind": "gaussian",
                "mean_coordinate": [0.0] * 18,
                "covariance": np.eye(18).tolist(),
            },
            "delay": {
                "prior_kind": "uniform",
                "bounds_seconds": [0.0, 0.05],
                "initial_seconds": 0.01,
                "coarse_grid_points": 9,
                "refinement_tolerance_seconds": 1.0e-5,
                "maximum_refinement_evaluations": 32,
            },
            "knot_policy": {
                "period_seconds": 0.01,
                "origin": "interval_start",
                "maximum_measurement_gap_seconds": 0.03,
            },
            "interpolation_policy": {
                "euclidean": "linear",
                "orientation": "so3_geodesic",
                "command": "zoh_record_issue_time",
                "allow_extrapolation": False,
            },
            "controller_snapshot_policy": {
                "source": "bag_startup_parameter_updates",
                "require_constant_within_interval": True,
            },
            "mode_hypotheses": [
                {
                    "mode_id": "recorded-mode",
                    "bag_schedules": {
                        "failure-04": {
                            "flight_state_source": "recorded_causal_schedule",
                            "integration_gate_source": "deterministic_replay",
                        }
                    },
                }
            ],
            "solver_settings": {
                "maximum_iterations": 50,
                "maximum_factorization_retries": 4,
                "maximum_model_evaluation_retries": 4,
                "acceptance_ratio": 1.0e-4,
                "gradient_tolerance": 1.0e-6,
                "scaled_step_tolerance": 1.0e-7,
                "relative_objective_tolerance": 1.0e-8,
                "initial_damping": 1.0e-3,
                "minimum_damping": 1.0e-12,
                "maximum_damping": 1.0e12,
            },
            "em_settings": {
                "maximum_iterations": 12,
                "minimum_iterations": 2,
                "maximum_repeated_q_rejections": 3,
                "maximum_repeated_lag_profile_failures": 3,
                "log_q_tolerance": 1.0e-3,
                "lag_tolerance": 1.0e-5,
                "map_objective_tolerance": 1.0e-5,
                "marginal_objective_tolerance": 1.0e-5,
                "q_acceptance_objective_tolerance": 0.0,
                "q_minimum_alpha": 1.0 / 64.0,
            },
            "mcmc_settings": {
                "enabled": True,
                "chain_count": 4,
                "warmup_steps": 100,
                "retained_draws": 200,
                "thinning": 1,
                "random_seed": 42,
                "local_scale": 0.5,
                "exact_ridge_scale": 0.25,
                "near_ridge_scale": 0.25,
                "identified_scale": 0.1,
                "delay_scale_seconds": 0.002,
                "near_relative_threshold": 1.0e-6,
                "rhat_threshold": 1.01,
                "minimum_effective_sample_size": 100.0,
            },
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_loads_strict_request_and_computes_stable_fingerprint(self):
        path = Path(self.temporary.name) / "request.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(self.request, stream)
        loaded = load_batch_estimation_request(path)
        direct = validate_batch_estimation_request(self.request)
        self.assertEqual(loaded.bag_ids, ("failure-04",))
        self.assertEqual(loaded.output_directory, self.output.resolve())
        self.assertEqual(loaded.fingerprint, direct.fingerprint)
        self.assertTrue(loaded.fingerprint.startswith("sha256:"))
        with self.assertRaises(TypeError):
            loaded.payload["new"] = True

    def test_q_scientific_definition_has_no_default(self):
        request = copy.deepcopy(self.request)
        del request["q"]["residual_quantity"]
        with self.assertRaisesRegex(ArtifactValidationError, "residual_quantity"):
            validate_batch_estimation_request(request)
        request = copy.deepcopy(self.request)
        request["q"]["residual_quantity"] = "unspecified"
        with self.assertRaisesRegex(ArtifactValidationError, "must be one of"):
            validate_batch_estimation_request(request)

    def test_unknown_and_old_stage_request_schemas_are_rejected(self):
        for schema in (
            "grape-param-estim/diagonal-q-request/v1",
            "grape-param-estim/assimilation-request/v1",
            "grape-param-estim/batch-estimation-request/v2",
        ):
            request = copy.deepcopy(self.request)
            request["schema"] = schema
            with self.assertRaises(ArtifactValidationError):
                validate_batch_estimation_request(request)
        request = copy.deepcopy(self.request)
        request["legacy_stage"] = "q"
        with self.assertRaisesRegex(ArtifactValidationError, "unknown keys"):
            validate_batch_estimation_request(request)

    def test_disabled_factor_requires_reason_and_enabled_factor_covariance(self):
        request = copy.deepcopy(self.request)
        request["bags"][0]["observation_factors"]["accelerometer"][
            "disabled_reason"
        ] = ""
        with self.assertRaisesRegex(ArtifactValidationError, "disabled_reason"):
            validate_batch_estimation_request(request)
        request = copy.deepcopy(self.request)
        request["bags"][0]["observation_factors"]["pose"]["covariances"] = None
        with self.assertRaisesRegex(ArtifactValidationError, "must be an object"):
            validate_batch_estimation_request(request)
        request = copy.deepcopy(self.request)
        request["bags"][0]["observation_factors"]["accelerometer"][
            "covariances"
        ] = {}
        with self.assertRaisesRegex(ArtifactValidationError, "must be null"):
            validate_batch_estimation_request(request)

    def test_initial_state_priors_are_explicit_and_not_sensor_covariances(self):
        request = copy.deepcopy(self.request)
        del request["bags"][0]["initial_state_prior_covariances"]
        with self.assertRaisesRegex(
            ArtifactValidationError, "initial_state_prior_covariances"
        ):
            validate_batch_estimation_request(request)

        request = copy.deepcopy(self.request)
        prior = request["bags"][0]["initial_state_prior_covariances"][
            "gyro_bias"
        ]
        prior["source"] = "preflight_static_interval"
        prior["values"] = [1.0e-6, 2.0e-6, 3.0e-6]
        validate_batch_estimation_request(request)

        request = copy.deepcopy(self.request)
        prior = request["bags"][0]["initial_state_prior_covariances"][
            "actuator_thrust"
        ]
        prior["coordinates"][0] = "observation_rotor_1"
        with self.assertRaisesRegex(ArtifactValidationError, "residual order"):
            validate_batch_estimation_request(request)

    def test_covariance_source_coordinates_and_units_are_protocol_fields(self):
        request = copy.deepcopy(self.request)
        covariance = request["bags"][0]["observation_factors"]["pose"][
            "covariances"
        ]["position_observation"]
        covariance["source"] = "guessed_default"
        with self.assertRaisesRegex(ArtifactValidationError, "source.*must be one of"):
            validate_batch_estimation_request(request)

        request = copy.deepcopy(self.request)
        covariance = request["bags"][0]["observation_factors"]["pose"][
            "covariances"
        ]["position_observation"]
        covariance["coordinates"][0] = "north"
        with self.assertRaisesRegex(ArtifactValidationError, "coordinates.*residual order"):
            validate_batch_estimation_request(request)

        request = copy.deepcopy(self.request)
        covariance = request["bags"][0]["observation_factors"]["gyro"][
            "covariances"
        ]["gyro_observation"]
        covariance["units"] = ["degree/s"] * 3
        with self.assertRaisesRegex(ArtifactValidationError, "units.*residual order"):
            validate_batch_estimation_request(request)

    def test_diagonal_covariance_is_finite_positive_and_exact_dimension(self):
        location = self.request["bags"][0]["observation_factors"]["velocity"][
            "covariances"
        ]["velocity_observation"]
        for invalid, message in (
            ([1.0, 1.0], "3 finite numbers"),
            ([1.0, float("nan"), 1.0], "finite numbers"),
            ([1.0, 0.0, 1.0], "positive numbers"),
            ([1.0, -0.1, 1.0], "positive numbers"),
            ([1.0, True, 1.0], "finite numbers"),
        ):
            request = copy.deepcopy(self.request)
            covariance = request["bags"][0]["observation_factors"][
                "velocity"
            ]["covariances"]["velocity_observation"]
            covariance["values"] = invalid
            with self.assertRaisesRegex(ArtifactValidationError, message):
                validate_batch_estimation_request(request)
        self.assertEqual(location["representation"], "diagonal")

    def test_full_covariance_must_be_finite_symmetric_positive_definite(self):
        request = copy.deepcopy(self.request)
        covariance = request["bags"][0]["observation_factors"]["pose"][
            "covariances"
        ]["orientation_observation"]
        covariance["representation"] = "full"
        covariance["values"] = (
            np.asarray(
                [[2.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 1.0]]
            ).tolist()
        )
        validate_batch_estimation_request(request)

        request = copy.deepcopy(request)
        request["bags"][0]["observation_factors"]["pose"]["covariances"][
            "orientation_observation"
        ]["values"][0][1] = 0.4
        with self.assertRaisesRegex(ArtifactValidationError, "symmetric"):
            validate_batch_estimation_request(request)

        request = copy.deepcopy(self.request)
        covariance = request["bags"][0]["observation_factors"]["pose"][
            "covariances"
        ]["orientation_observation"]
        covariance["representation"] = "full"
        covariance["values"] = np.diag([1.0, 0.0, 1.0]).tolist()
        with self.assertRaisesRegex(ArtifactValidationError, "positive definite"):
            validate_batch_estimation_request(request)

        request = copy.deepcopy(self.request)
        covariance = request["bags"][0]["observation_factors"]["pose"][
            "covariances"
        ]["orientation_observation"]
        covariance["representation"] = "full"
        covariance["values"] = np.eye(2).tolist()
        with self.assertRaisesRegex(ArtifactValidationError, "3 by 3"):
            validate_batch_estimation_request(request)

    def test_pose_and_fixed_graph_covariance_blocks_are_complete_and_strict(self):
        request = copy.deepcopy(self.request)
        del request["bags"][0]["observation_factors"]["pose"]["covariances"][
            "orientation_observation"
        ]
        with self.assertRaisesRegex(ArtifactValidationError, "missing keys"):
            validate_batch_estimation_request(request)

        request = copy.deepcopy(self.request)
        del request["bags"][0]["fixed_factor_covariances"][
            "position_kinematic"
        ]
        with self.assertRaisesRegex(ArtifactValidationError, "position_kinematic"):
            validate_batch_estimation_request(request)

        request = copy.deepcopy(self.request)
        request["bags"][0]["fixed_factor_covariances"]["unknown_factor"] = {}
        with self.assertRaisesRegex(ArtifactValidationError, "unknown keys"):
            validate_batch_estimation_request(request)

        request = copy.deepcopy(self.request)
        covariance = request["bags"][0]["fixed_factor_covariances"][
            "actuator_thrust_transition"
        ]
        covariance["implicit_default"] = True
        with self.assertRaisesRegex(ArtifactValidationError, "unknown keys"):
            validate_batch_estimation_request(request)

    def test_old_source_only_factor_contract_is_rejected(self):
        request = copy.deepcopy(self.request)
        request["bags"][0]["observation_factors"]["pose"] = {
            "enabled": True,
            "disabled_reason": None,
            "covariance_source": "message_covariance",
        }
        with self.assertRaisesRegex(ArtifactValidationError, "covariances"):
            validate_batch_estimation_request(request)

    def test_request_rejects_duplicate_bags_and_nonexistent_paths(self):
        request = copy.deepcopy(self.request)
        request["bags"].append(copy.deepcopy(request["bags"][0]))
        with self.assertRaisesRegex(ArtifactValidationError, "duplicate bag"):
            validate_batch_estimation_request(request)
        request = copy.deepcopy(self.request)
        request["bags"][0]["path"] = str(Path(self.temporary.name) / "missing.bag")
        with self.assertRaisesRegex(ArtifactValidationError, "existing file"):
            validate_batch_estimation_request(request)
        request = copy.deepcopy(self.request)
        request["bags"][0]["sha256"] = "sha256:" + "b" * 64
        with self.assertRaisesRegex(ArtifactValidationError, "does not match"):
            validate_batch_estimation_request(request)

    def test_parameter_prior_must_be_proper_and_mcmc_mode_must_agree(self):
        request = copy.deepcopy(self.request)
        request["parameter_prior"]["covariance"][0][0] = 0.0
        with self.assertRaisesRegex(ArtifactValidationError, "positive definite"):
            validate_batch_estimation_request(request)
        request = copy.deepcopy(self.request)
        request["run_mode"] = "estimate_only"
        with self.assertRaisesRegex(ArtifactValidationError, "must be false"):
            validate_batch_estimation_request(request)
        request["mcmc_settings"] = {"enabled": False}
        validate_batch_estimation_request(request)

    def test_interpolation_extrapolation_and_mode_schedule_are_explicit(self):
        request = copy.deepcopy(self.request)
        request["interpolation_policy"]["allow_extrapolation"] = True
        with self.assertRaisesRegex(ArtifactValidationError, "must be false"):
            validate_batch_estimation_request(request)
        request = copy.deepcopy(self.request)
        request["mode_hypotheses"][0]["bag_schedules"]["failure-04"][
            "flight_state_source"
        ] = "guessed"
        with self.assertRaisesRegex(ArtifactValidationError, "must be one of"):
            validate_batch_estimation_request(request)


if __name__ == "__main__":
    unittest.main()
