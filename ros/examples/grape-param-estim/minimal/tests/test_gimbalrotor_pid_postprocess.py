from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from _support import MINIMAL
from grape_param_estim.controller import (
    _source_pseudoinverse,
    acceleration_allocation_matrix,
)
from grape_param_estim.controller_config import PID_GAIN_NAMES, PID_GROUPS
from grape_param_estim.gimbalrotor_pid_postprocess import (
    BagProvenance,
    PostprocessInputError,
    ScaleFreePlant,
    apply_gain_corrections_to_yaml,
    build_controller_snapshot_geometry,
    build_nominal_controller_allocation,
    build_real_scale_free_allocation,
    build_report,
    calculate_gain_corrections,
    characteristic_length,
    compute_static_pid_proposal,
    dimensionless_effectiveness,
    group_scale,
    load_bag_provenance,
    load_controller_yaml,
    load_estimator_result,
    load_vehicle_model,
    source_compatible_pseudoinverse,
)
from grape_param_estim.system import GrapeGeometry
from three_bag_gimbalrotor_pid_postprocess_summary import (
    build_three_bag_summary,
    render_markdown,
)


FAILURE1_RESULT = (
    MINIMAL
    / "outputs/916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1"
    / "prior_ablation"
    / "single_rosbag_1_nominal_pseudo_conditioning_production_20260817"
    / "cases/prior_free/result.json"
)


def _result_payload():
    return {
        "overall_case_status": "completed",
        "optimization_status": "completed",
        "success": True,
        "case_name": "prior_free",
        "source_commit": "a" * 40,
        "prior": {"active": False},
        "parameters": {
            "scale_free": {
                "inertia_over_mass_m2": [
                    [0.03, 0.001, 0.0],
                    [0.001, 0.032, 0.0],
                    [0.0, 0.0, 0.06],
                ],
                "cog_position_body_m": [0.0, 0.0, 0.01],
                "force_effectiveness_over_mass": [0.4, 0.4, 0.4, 0.4],
            },
            "rotor_lag_seconds": 0.12,
            "estimated": {
                "mass_kg": 999.0,
                "inertia_kg_m2": [[999.0, 0.0, 0.0]] * 3,
                "force_effectiveness": [999.0] * 4,
            },
        },
    }


def _controller_document():
    return {
        "aerial_robot_control_name": (
            "aerial_robot_control/gimbalrotor_controller"
        ),
        "controller": {
            "torque_allocation_matrix_inv_pub_interval": 0.05,
            "wrench_allocation_matrix_pub_interval": 0.1,
            "gimbal_calc_in_fc": False,
            "xy": {
                "p_gain": 4.0,
                "i_gain": 0.1,
                "d_gain": 2.0,
                "limit_sum": 4.0,
            },
            "z": {
                "p_gain": 5.0,
                "i_gain": 1.0,
                "d_gain": 2.5,
                "limit_sum": 25.0,
            },
            "roll_pitch": {
                "p_gain": 13.0,
                "i_gain": 1.0,
                "d_gain": 20.0,
                "start_rp_integration_height": 0.01,
            },
            "yaw": {
                "p_gain": 6.0,
                "i_gain": 1.0,
                "d_gain": 2.0,
                "need_d_control": True,
                "limit_err_p": 0.4,
            },
        },
    }


def _strip_gain_leaves(document):
    copied = deepcopy(document)
    for group in PID_GROUPS:
        for gain in PID_GAIN_NAMES:
            del copied["controller"][group][gain]
    return copied


def _direct_nominal_allocation(model):
    geometry = build_controller_snapshot_geometry(model)
    result = np.zeros((6, 8))
    basis = (np.asarray((0.0, 1.0, 0.0)), np.asarray((0.0, 0.0, 1.0)))
    inverse_inertia = np.linalg.inv(model.parameters.inertia)
    for rotor in range(4):
        yaw = geometry.arm_yaws[rotor]
        rotation = np.asarray(
            (
                (np.cos(yaw), -np.sin(yaw), 0.0),
                (np.sin(yaw), np.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        for component, local in enumerate(basis):
            column = 2 * rotor + component
            force = rotation @ local
            torque = np.cross(geometry.rotor_origins[rotor], force) + (
                geometry.rotor_directions[rotor]
                * geometry.moment_force_rate
                * force
            )
            result[:3, column] = force / model.parameters.mass
            result[3:, column] = inverse_inertia @ torque
    return result


class GimbalrotorPidPostprocessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.model = load_vehicle_model(MINIMAL / "grape_vehicle_model.json")
        self.controller_path = self.directory / "GimbalrotorControl.yaml"
        self.controller_path.write_text(
            yaml.safe_dump(_controller_document(), sort_keys=False),
            encoding="utf-8",
        )
        self.controller = load_controller_yaml(self.controller_path)

    def _write_result(self, payload=None, name="result.json"):
        path = self.directory / name
        path.write_text(
            json.dumps(_result_payload() if payload is None else payload),
            encoding="utf-8",
        )
        return path

    def _nominal_plant(self, force_scale=1.0, cog=None, inertia=None):
        nominal = self.model.parameters
        return ScaleFreePlant(
            inertia_over_mass=(
                nominal.inertia / nominal.mass
                if inertia is None
                else inertia
            ),
            cog_position_body=(nominal.cog_offset if cog is None else cog),
            force_effectiveness_over_mass=(
                force_scale * np.ones(4) / nominal.mass
            ),
            rotor_lag_seconds=0.2,
        )

    def _estimator_with_plant(self, plant):
        payload = _result_payload()
        payload["parameters"]["scale_free"] = {
            "inertia_over_mass_m2": plant.inertia_over_mass.tolist(),
            "cog_position_body_m": plant.cog_position_body.tolist(),
            "force_effectiveness_over_mass": (
                plant.force_effectiveness_over_mass.tolist()
            ),
        }
        payload["parameters"]["rotor_lag_seconds"] = plant.rotor_lag_seconds
        return load_estimator_result(self._write_result(payload, "plant.json"))

    def test_current_result_contract_and_failure1_golden_values(self):
        result = load_estimator_result(FAILURE1_RESULT)
        self.assertEqual(result.case_name, "prior_free")
        self.assertAlmostEqual(result.plant.rotor_lag_seconds, 0.19719922542572021)
        proposal = compute_static_pid_proposal(
            result, self.model, self.controller
        )
        expected = {
            "xy": 1.15204476,
            "z": 1.16976482,
            "roll_pitch": 3.52877431,
            "yaw": 3.37886790,
        }
        for group, scale in expected.items():
            self.assertAlmostEqual(
                proposal.corrections[group].scale, scale, places=7
            )
        self.assertAlmostEqual(
            proposal.characteristic_length, 0.1915849943, places=9
        )
        self.assertAlmostEqual(proposal.error_before, 1.30016466, places=7)
        self.assertAlmostEqual(proposal.error_after, 0.44348763, places=7)
        self.assertAlmostEqual(proposal.coupling_ratio, 0.11628596, places=7)
        self.assertEqual(proposal.proposal_status, "review_required")
        self.assertIn("large_static_gain_change", proposal.warnings)
        self.assertAlmostEqual(
            proposal.corrections["roll_pitch"].proposed.p_gain,
            45.874066,
            places=5,
        )

    def test_failed_optimizer_is_rejected(self):
        for key, value in (
            ("optimization_status", "failed"),
            ("success", False),
        ):
            payload = _result_payload()
            payload[key] = value
            with self.subTest(key=key), self.assertRaises(PostprocessInputError):
                load_estimator_result(self._write_result(payload, key + ".json"))

    def test_non_prior_free_case_requires_explicit_override(self):
        payload = _result_payload()
        payload["case_name"] = "cog_all_nominal"
        payload["prior"] = {
            "active": True,
            "name": "cog_all_nominal",
            "role": "pseudo_conditioning_ablation",
            "source_path": "/tmp/prior.json",
        }
        path = self._write_result(payload)
        with self.assertRaises(PostprocessInputError):
            load_estimator_result(path)
        accepted = load_estimator_result(
            path, allow_non_prior_free_result=True
        )
        self.assertIn("non_prior_free_estimate", accepted.warnings)
        self.assertEqual(accepted.prior["name"], "cog_all_nominal")

    def test_point_estimate_only_requires_override_and_warning(self):
        payload = _result_payload()
        payload["overall_case_status"] = "point_estimate_completed"
        path = self._write_result(payload)
        with self.assertRaises(PostprocessInputError):
            load_estimator_result(path)
        accepted = load_estimator_result(
            path, allow_point_estimate_only=True
        )
        self.assertIn("postfit_uncertainty_unavailable", accepted.warnings)

    def test_scale_free_plant_validation(self):
        nominal = self._nominal_plant()
        invalid = (
            dict(
                inertia_over_mass=np.full((3, 3), np.nan),
                cog_position_body=nominal.cog_position_body,
                force_effectiveness_over_mass=nominal.force_effectiveness_over_mass,
                rotor_lag_seconds=0.1,
            ),
            dict(
                inertia_over_mass=np.asarray(
                    ((1.0, 0.1, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                ),
                cog_position_body=nominal.cog_position_body,
                force_effectiveness_over_mass=nominal.force_effectiveness_over_mass,
                rotor_lag_seconds=0.1,
            ),
            dict(
                inertia_over_mass=np.diag((1.0, 1.0, -1.0)),
                cog_position_body=nominal.cog_position_body,
                force_effectiveness_over_mass=nominal.force_effectiveness_over_mass,
                rotor_lag_seconds=0.1,
            ),
            dict(
                inertia_over_mass=nominal.inertia_over_mass,
                cog_position_body=nominal.cog_position_body,
                force_effectiveness_over_mass=(1.0, 1.0, 0.0, 1.0),
                rotor_lag_seconds=0.1,
            ),
        )
        for index, values in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(PostprocessInputError):
                ScaleFreePlant(**values)

    def test_body_origins_are_explicitly_converted_to_controller_cog_origins(self):
        converted = build_controller_snapshot_geometry(self.model)
        expected = (
            self.model.body_geometry.rotor_origins
            - self.model.parameters.cog_offset[None, :]
        )
        self.assertTrue(np.array_equal(converted.rotor_origins, expected))
        self.assertFalse(
            np.array_equal(
                converted.rotor_origins,
                self.model.body_geometry.rotor_origins,
            )
        )

    def test_nominal_allocation_matches_independent_zero_gimbal_formula(self):
        production = build_nominal_controller_allocation(self.model)
        direct = _direct_nominal_allocation(self.model)
        self.assertEqual(production.source_threshold_rank, 6)
        self.assertTrue(
            np.allclose(production.matrix, direct, rtol=0.0, atol=2.0e-14)
        )

    def test_current_hard_coded_controller_geometry_is_parity_reference_only(self):
        adapted = build_controller_snapshot_geometry(self.model)
        hard_coded = GrapeGeometry.grape()
        self.assertTrue(
            np.allclose(
                adapted.rotor_origins,
                hard_coded.rotor_origins,
                rtol=0.0,
                atol=1.0e-15,
            )
        )
        expected = acceleration_allocation_matrix(
            self.model.parameters, hard_coded, np.zeros(4)
        )
        actual = build_nominal_controller_allocation(self.model).matrix
        self.assertTrue(np.allclose(actual, expected, rtol=0.0, atol=1.0e-14))

    def test_source_pseudoinverse_uses_absolute_threshold_and_controller_parity(self):
        matrix = np.zeros((2, 3))
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0e-4
        actual = source_compatible_pseudoinverse(matrix)
        self.assertTrue(np.array_equal(actual, _source_pseudoinverse(matrix)))
        self.assertEqual(actual[1, 1], 0.0)
        matrix[1, 1] = np.nextafter(1.0e-4, np.inf)
        self.assertGreater(source_compatible_pseudoinverse(matrix)[1, 1], 0.0)

    def test_nominal_identity_invariant_and_unit_scales(self):
        result = self._estimator_with_plant(self._nominal_plant())
        proposal = compute_static_pid_proposal(
            result, self.model, self.controller
        )
        self.assertTrue(
            np.allclose(proposal.effectiveness, np.eye(6), atol=8.0e-15)
        )
        self.assertTrue(
            np.allclose(
                proposal.dimensionless_effectiveness,
                np.eye(6),
                atol=8.0e-15,
            )
        )
        for correction in proposal.corrections.values():
            self.assertAlmostEqual(correction.scale, 1.0, places=13)

    def test_absolute_gauge_representatives_cannot_affect_the_proposal(self):
        first = _result_payload()
        second = deepcopy(first)
        first["parameters"]["estimated"] = {
            "mass_kg": 1.0,
            "inertia_kg_m2": np.eye(3).tolist(),
            "force_effectiveness": [1.0] * 4,
        }
        second["parameters"]["estimated"] = {
            "mass_kg": 1.0e9,
            "inertia_kg_m2": (1.0e9 * np.eye(3)).tolist(),
            "force_effectiveness": [1.0e9] * 4,
        }
        one = load_estimator_result(self._write_result(first, "one.json"))
        two = load_estimator_result(self._write_result(second, "two.json"))
        proposal_one = compute_static_pid_proposal(
            one, self.model, self.controller
        )
        proposal_two = compute_static_pid_proposal(
            two, self.model, self.controller
        )
        self.assertTrue(
            np.array_equal(
                proposal_one.effectiveness, proposal_two.effectiveness
            )
        )
        self.assertEqual(
            [value.scale for value in proposal_one.corrections.values()],
            [value.scale for value in proposal_two.corrections.values()],
        )

    def test_uniform_force_over_mass_scaling_has_inverse_gain_scales(self):
        force_scale = 1.7
        result = self._estimator_with_plant(
            self._nominal_plant(force_scale=force_scale)
        )
        proposal = compute_static_pid_proposal(
            result, self.model, self.controller
        )
        self.assertTrue(
            np.allclose(
                proposal.effectiveness,
                force_scale * np.eye(6),
                rtol=2.0e-14,
                atol=2.0e-14,
            )
        )
        for correction in proposal.corrections.values():
            self.assertAlmostEqual(
                correction.scale, 1.0 / force_scale, places=13
            )

    def test_cog_perturbation_produces_cross_axis_torque_coupling(self):
        cog = self.model.parameters.cog_offset + np.asarray((0.03, -0.02, 0.01))
        real = build_real_scale_free_allocation(
            self._nominal_plant(cog=cog), self.model
        ).matrix
        nominal = build_nominal_controller_allocation(self.model).matrix
        self.assertTrue(np.allclose(real[:3], nominal[:3], atol=1.0e-15))
        self.assertGreater(np.linalg.norm(real[3:] - nominal[3:]), 0.1)

    def test_inertia_perturbation_leaves_translational_allocation_unchanged(self):
        nominal_plant = self._nominal_plant()
        changed_inertia = nominal_plant.inertia_over_mass.copy()
        changed_inertia[0, 0] *= 1.6
        changed = build_real_scale_free_allocation(
            self._nominal_plant(inertia=changed_inertia), self.model
        ).matrix
        baseline = build_real_scale_free_allocation(
            nominal_plant, self.model
        ).matrix
        self.assertTrue(np.array_equal(changed[:3], baseline[:3]))
        self.assertGreater(np.linalg.norm(changed[3:] - baseline[3:]), 0.1)

    def test_rotor_specific_effectiveness_couples_axes_and_group_fit_improves(self):
        plant = self._nominal_plant()
        force = plant.force_effectiveness_over_mass.copy()
        force[0] *= 1.5
        result = self._estimator_with_plant(
            ScaleFreePlant(
                plant.inertia_over_mass,
                plant.cog_position_body,
                force,
                plant.rotor_lag_seconds,
            )
        )
        proposal = compute_static_pid_proposal(
            result, self.model, self.controller
        )
        off_diagonal = proposal.dimensionless_effectiveness - np.diag(
            np.diag(proposal.dimensionless_effectiveness)
        )
        self.assertGreater(np.linalg.norm(off_diagonal), 0.05)
        for correction in proposal.corrections.values():
            self.assertLessEqual(
                correction.error_after,
                correction.error_before + 1.0e-14,
            )

    def test_mixed_unit_group_fitting_uses_dimensionless_effectiveness(self):
        raw = np.eye(6)
        raw[0, 3] = 2.0
        length = 0.2
        scaled = dimensionless_effectiveness(raw, length)
        self.assertAlmostEqual(scaled[0, 3], 10.0)
        raw_scale = group_scale(raw, (3, 4))
        scaled_scale = group_scale(scaled, (3, 4))
        self.assertNotAlmostEqual(raw_scale, scaled_scale)
        corrections = calculate_gain_corrections(
            scaled, self.controller.gains
        )
        self.assertAlmostEqual(
            corrections["roll_pitch"].scale, scaled_scale
        )

    def test_yaml_transformation_changes_only_twelve_gain_leaves(self):
        original_text = self.controller_path.read_text(encoding="utf-8")
        result = load_estimator_result(FAILURE1_RESULT)
        proposal = compute_static_pid_proposal(
            result, self.model, self.controller
        )
        full, overlay = apply_gain_corrections_to_yaml(
            self.controller, proposal.corrections
        )
        self.assertEqual(
            _strip_gain_leaves(self.controller.document),
            _strip_gain_leaves(full),
        )
        changed = []
        for group in PID_GROUPS:
            for gain in PID_GAIN_NAMES:
                if (
                    full["controller"][group][gain]
                    != self.controller.document["controller"][group][gain]
                ):
                    changed.append("{}.{}".format(group, gain))
        self.assertEqual(len(changed), 12)
        self.assertEqual(set(overlay["controller"]), set(PID_GROUPS))
        self.assertEqual(
            self.controller_path.read_text(encoding="utf-8"), original_text
        )

    def test_source_default_controller_modes_are_resolved(self):
        mode = self.controller.mode
        self.assertEqual(mode.gimbal_dof, 1)
        self.assertEqual(mode.gimbal_dof_source, "cpp_default")
        self.assertFalse(mode.underactuate)
        self.assertEqual(mode.underactuate_source, "cpp_default")
        self.assertFalse(mode.gimbal_calc_in_fc)
        self.assertEqual(mode.gimbal_calc_in_fc_source, "yaml")
        self.assertFalse(mode.hovering_approximate)
        self.assertEqual(mode.hovering_approximate_source, "cpp_default")

    def test_unsupported_controller_branches_are_rejected(self):
        changes = (
            ("gimbal_dof", 2),
            ("underactuate", True),
            ("gimbal_calc_in_fc", True),
            ("yaw.need_d_control", False),
        )
        for index, (key, value) in enumerate(changes):
            document = _controller_document()
            if key.startswith("yaw."):
                document["controller"]["yaw"][key.split(".", 1)[1]] = value
            else:
                document["controller"][key] = value
            path = self.directory / "unsupported-{}.yaml".format(index)
            path.write_text(
                yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
            )
            with self.subTest(key=key), self.assertRaises(PostprocessInputError):
                load_controller_yaml(path)

    def test_bag_provenance_and_report_contain_fixed_assumptions(self):
        bag = load_bag_provenance(
            MINIMAL / "bag_jsons/single_rosbag_1.json"
        )
        self.assertEqual((bag.start_seconds, bag.end_seconds), (19.0, 25.0))
        result = load_estimator_result(FAILURE1_RESULT)
        proposal = compute_static_pid_proposal(
            result, self.model, self.controller
        )
        report = build_report(
            source_commit="source-test",
            result=result,
            bag=bag,
            model=self.model,
            controller=self.controller,
            proposal=proposal,
        )
        self.assertEqual(
            report["nominal_controller_model"][
                "torque_effectiveness_source"
            ],
            "fixed_nominal_vehicle_model",
        )
        self.assertEqual(report["input"]["bag_interval_seconds"], [19.0, 25.0])
        self.assertNotIn("estimated", report["scale_free_plant"])

    def test_three_bag_summary_reports_spread_without_deployment_yaml(self):
        result = load_estimator_result(FAILURE1_RESULT)
        proposal = compute_static_pid_proposal(
            result, self.model, self.controller
        )
        base = build_report(
            source_commit="source-test",
            result=result,
            bag=BagProvenance(
                Path("bag.json"), "/tmp/test.bag", 1.0, 2.0
            ),
            model=self.model,
            controller=self.controller,
            proposal=proposal,
        )
        reports = {}
        for index, label in enumerate(("failure1", "failure2", "success")):
            report = deepcopy(base)
            for group in PID_GROUPS:
                report["gain_groups"][group]["scale"] += 0.1 * index
            reports[label] = report
        summary = build_three_bag_summary(
            reports, summary_source_commit="source-test"
        )
        self.assertFalse(summary["deployment_yaml_generated"])
        self.assertAlmostEqual(
            summary["scale_statistics"]["xy"]["standard_deviation"],
            np.std((1.152044762905967, 1.252044762905967, 1.352044762905967)),
        )
        markdown = render_markdown(summary)
        self.assertIn("No mean deployment YAML", markdown)
        self.assertIn("failure1", markdown)


if __name__ == "__main__":
    unittest.main()
