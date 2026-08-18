from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
SOURCE = PROJECT / "src"
for path in (HERE, SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gimbalrotor_pid_local_pole_validation as local_poles
import three_bag_gimbalrotor_pid_local_pole_validation as three_bag
from grape_param_estim.controller import ControllerConfig, GrapeController
from grape_param_estim.controller_config import PID_GAIN_NAMES, PID_GROUPS
from grape_param_estim.dynamics import FullSixDofPlant
from grape_param_estim.geometry import quaternion_to_matrix
from grape_param_estim.gimbalrotor_pid_postprocess import (
    PostprocessInputError,
    build_controller_snapshot_geometry,
    load_vehicle_model,
)
from grape_param_estim.system import ActuatorParameters, ActuatorState, VehicleParameters


ESTIMATOR = (
    HERE
    / "outputs"
    / "916b66168ba4cc5493cd9a7b04dae1d63b0b1ba1"
    / "prior_ablation"
    / "single_rosbag_2_nominal_pseudo_conditioning_production_20260817"
    / "cases"
    / "prior_free"
)
STATIC = (
    HERE
    / "outputs"
    / "585db5ba8a236232d85f2097615cf64b7eb76ff0"
    / "gimbalrotor_pid_postprocess"
    / "single_rosbag_2_prior_free_static_pid_production_20260817"
    / "pid_gain_postprocess.json"
)
YAML = Path(
    "/home/leus/catkin_ws/src/jsk_aerial_robot/robots/gimbalrotor/config/grape/"
    "GimbalrotorControl.yaml"
)
MODEL = HERE / "grape_vehicle_model.json"
BAG_JSON = HERE / "bag_jsons" / "single_rosbag_2.json"


@lru_cache(maxsize=2)
def case_inputs(mode: str = "conservative_fusion"):
    return local_poles.load_case_inputs(
        result_path=ESTIMATOR / "result.json",
        arrays_path=ESTIMATOR / "arrays.npz",
        static_postprocess_path=STATIC,
        arguments_path=ESTIMATOR / "arguments.json",
        bag_json_path=BAG_JSON,
        controller_yaml_path=YAML,
        vehicle_model_path=MODEL,
        covariance_mode=mode,
    )


@lru_cache(maxsize=8)
def nominal_bundle(delay_seconds: float = 0.0):
    model = load_vehicle_model(MODEL)
    configuration = ControllerConfig.grape()
    controller = GrapeController(
        configuration, model.parameters, build_controller_snapshot_geometry(model)
    )
    plant = FullSixDofPlant(model.parameters, model.body_geometry)
    actuator = ActuatorParameters()
    delay = local_poles.decompose_thrust_delay(delay_seconds, 0.01)
    trim = local_poles.solve_hover_trim(
        controller=controller,
        plant=plant,
        actuator_parameters=actuator,
        reference=local_poles.hover_reference(),
        controller_dt=0.01,
        delay=delay,
    )
    context = local_poles.PoleContext(
        controller,
        plant,
        actuator,
        local_poles.hover_reference(),
        0.01,
        delay,
        trim.state,
    )
    return model, controller, plant, actuator, trim, context


# Input and provenance contract.
def test_result_arrays_and_arguments_are_one_estimator_case():
    inputs = case_inputs()
    assert inputs.result_path.parent == inputs.arrays_path.parent == inputs.arguments_path.parent


def test_non_sibling_arrays_are_rejected():
    with pytest.raises(PostprocessInputError, match="siblings"):
        local_poles.load_case_inputs(
            result_path=ESTIMATOR / "result.json",
            arrays_path=HERE / "not-the-case" / "arrays.npz",
            static_postprocess_path=STATIC,
            arguments_path=ESTIMATOR / "arguments.json",
            bag_json_path=BAG_JSON,
            controller_yaml_path=YAML,
            vehicle_model_path=MODEL,
            covariance_mode="conservative_fusion",
        )


def test_static_postprocess_must_match_estimator_case():
    wrong = str(STATIC).replace("single_rosbag_2_", "single_rosbag_1_")
    with pytest.raises(PostprocessInputError, match="source commit|result|bag"):
        local_poles.load_case_inputs(
            result_path=ESTIMATOR / "result.json",
            arrays_path=ESTIMATOR / "arrays.npz",
            static_postprocess_path=Path(wrong),
            arguments_path=ESTIMATOR / "arguments.json",
            bag_json_path=BAG_JSON,
            controller_yaml_path=YAML,
            vehicle_model_path=MODEL,
            covariance_mode="conservative_fusion",
        )


def test_recorded_gain_source_is_retained():
    inputs = case_inputs()
    assert inputs.static_baseline.recorded_gain_source == "rosbag_recorded_dynamic_reconfigure"


def test_controller_yaml_sha_is_audited():
    inputs = case_inputs()
    assert inputs.controller_yaml.sha256 == inputs.static_payload["input"]["controller_yaml_sha256"]


def test_recorded_gains_replace_template_exactly():
    inputs = case_inputs()
    expected_groups = (0, 0, 1, 2, 2, 3)
    for axis, group in enumerate(expected_groups):
        pid = inputs.controller_configuration.pid[axis]
        assert np.array_equal(
            np.asarray((pid.p_gain, pid.i_gain, pid.d_gain)),
            inputs.recorded_gains.values[group],
        )


def test_controller_limits_survive_gain_replacement():
    inputs = case_inputs()
    template = ControllerConfig.grape()
    for before, after in zip(template.pid, inputs.controller_configuration.pid):
        assert before.limit_sum == after.limit_sum
        assert before.limit_error_i == after.limit_error_i


# Scale-free plant and physical gauge.
def test_quotient_center_decodes_saved_scale_free_point():
    inputs = case_inputs()
    decoded = inputs.sampling_coordinates.decode(np.zeros(13))
    assert np.allclose(
        local_poles.scale_free_vector(decoded),
        local_poles.scale_free_vector(inputs.result.plant),
        rtol=2e-9,
        atol=2e-11,
    )


def test_nominal_mass_gauge_reproduces_inertia_ratio():
    inputs = case_inputs()
    physical = local_poles.physical_plant_from_scale_free(inputs.result.plant, inputs.vehicle_model)
    assert np.allclose(physical.inertia / physical.mass, inputs.result.plant.inertia_over_mass)


def test_nominal_mass_gauge_reproduces_force_ratio():
    inputs = case_inputs()
    physical = local_poles.physical_plant_from_scale_free(inputs.result.plant, inputs.vehicle_model)
    assert np.allclose(
        physical.force_effectiveness / physical.mass,
        inputs.result.plant.force_effectiveness_over_mass,
    )


def test_common_positive_scale_preserves_hover_acceleration():
    model, _controller, plant, _actuator, trim, _context = nominal_bundle()
    scale = 7.5
    parameters = model.parameters
    scaled = VehicleParameters(
        mass=scale * parameters.mass,
        inertia=scale * parameters.inertia,
        cog_offset=parameters.cog_offset,
        force_effectiveness=scale * parameters.force_effectiveness,
        torque_effectiveness=parameters.torque_effectiveness,
        linear_drag=parameters.linear_drag,
        angular_drag=parameters.angular_drag,
    )
    other = FullSixDofPlant(scaled, model.body_geometry)
    first = plant.derivative(0.0, trim.state.rigid_body.as_vector(), trim.state.actuators)
    second = other.derivative(0.0, trim.state.rigid_body.as_vector(), trim.state.actuators)
    assert np.allclose(first, second, rtol=2e-13, atol=2e-13)


def test_common_positive_scale_preserves_local_poles():
    model, controller, _plant, actuator, _trim, context = nominal_bundle()
    parameters = model.parameters
    scale = 3.0
    scaled = VehicleParameters(
        scale * parameters.mass,
        scale * parameters.inertia,
        parameters.cog_offset,
        scale * parameters.force_effectiveness,
        parameters.torque_effectiveness,
        parameters.linear_drag,
        parameters.angular_drag,
    )
    other_plant = FullSixDofPlant(scaled, model.body_geometry)
    other_trim = local_poles.solve_hover_trim(
        controller=controller,
        plant=other_plant,
        actuator_parameters=actuator,
        reference=context.reference,
        controller_dt=context.controller_dt,
        delay=context.delay,
    )
    other_context = replace(context, plant=other_plant, trim=other_trim.state)
    first = np.sort_complex(np.linalg.eigvals(local_poles.central_difference_jacobian(context)))
    second = np.sort_complex(np.linalg.eigvals(local_poles.central_difference_jacobian(other_context)))
    assert np.allclose(first, second, rtol=5e-7, atol=5e-8)


# Hover trim.
def test_nominal_symmetric_hover_trim_matches_gravity_solution():
    _model, _controller, _plant, _actuator, trim, _context = nominal_bundle()
    assert trim.state.controller.integral_error[2] == pytest.approx(9.80665, abs=2e-7)


def test_nominal_trim_has_machine_scale_rigid_acceleration():
    _model, _controller, _plant, _actuator, trim, _context = nominal_bundle()
    assert np.linalg.norm(trim.residual[:6]) < 1e-7


def test_nominal_trim_gimbal_is_fixed_point():
    _model, _controller, _plant, _actuator, trim, _context = nominal_bundle()
    assert np.linalg.norm(trim.gimbal_fixed_point_defect, ord=np.inf) < trim.equilibrium_tolerance


def test_nominal_trim_integral_is_unchanged():
    _model, _controller, _plant, _actuator, trim, _context = nominal_bundle()
    assert np.array_equal(trim.controller_integral_defect, np.zeros(6))


def test_filled_nonzero_delay_queue_preserves_trim():
    _model, _controller, _plant, _actuator, trim, _context = nominal_bundle(0.015)
    assert trim.equilibrium_valid
    assert trim.one_step_defect_norm <= trim.equilibrium_tolerance


def test_delay_changes_jacobian_not_constant_equilibrium():
    *_, zero_trim, zero_context = nominal_bundle(0.0)
    *_, delayed_trim, delayed_context = nominal_bundle(0.015)
    assert np.allclose(zero_trim.state.controller.integral_error, delayed_trim.state.controller.integral_error)
    zero = local_poles.central_difference_jacobian(zero_context)
    delayed = local_poles.central_difference_jacobian(delayed_context)
    assert zero.shape != delayed.shape


# Exact thrust-only ZOH delay.
def test_zero_delay_uses_current_thrust_for_whole_interval():
    delay = local_poles.decompose_thrust_delay(0.0, 0.1)
    current = np.arange(4.0)
    segments = local_poles.delayed_thrust_segments(delay, np.empty((0, 4)), current)
    assert segments[0][0] == 0.1
    assert np.array_equal(segments[0][1], current)


def test_exact_multiple_uses_c_k_minus_m():
    delay = local_poles.decompose_thrust_delay(0.2, 0.1)
    queue = np.asarray(((1, 2, 3, 4), (5, 6, 7, 8)), dtype=float)
    segments = local_poles.delayed_thrust_segments(delay, queue, np.full(4, 9.0))
    assert len(segments) == 1
    assert np.array_equal(segments[0][1], queue[0])


def test_fractional_delay_uses_two_historical_targets():
    delay = local_poles.decompose_thrust_delay(0.25, 0.1)
    queue = np.arange(12.0).reshape(3, 4)
    segments = local_poles.delayed_thrust_segments(delay, queue, np.full(4, 99.0))
    assert [value[0] for value in segments] == pytest.approx((0.05, 0.05))
    assert np.array_equal(segments[0][1], queue[0])
    assert np.array_equal(segments[1][1], queue[1])


def test_subsample_delay_second_segment_uses_current_thrust():
    delay = local_poles.decompose_thrust_delay(0.025, 0.1)
    queue = np.full((1, 4), 1.0)
    current = np.full(4, 2.0)
    segments = local_poles.delayed_thrust_segments(delay, queue, current)
    assert np.array_equal(segments[0][1], queue[0])
    assert np.array_equal(segments[1][1], current)


def test_delay_queue_contains_four_thrust_values_per_depth():
    delay = local_poles.decompose_thrust_delay(0.21, 0.1)
    assert delay.depth == 3
    assert 4 * delay.depth == 12


def test_delay_selection_has_no_gimbal_input():
    assert "gimbal" not in local_poles.delayed_thrust_segments.__annotations__


def test_queue_shift_matches_hand_constructed_sequence():
    queue = np.asarray(((1, 1, 1, 1), (2, 2, 2, 2)), dtype=float)
    shifted = local_poles.shift_thrust_queue(queue, np.full(4, 3.0))
    assert np.array_equal(shifted, np.asarray(((2, 2, 2, 2), (3, 3, 3, 3))))


def test_near_exact_multiple_is_canonicalized():
    delay = local_poles.decompose_thrust_delay(np.nextafter(0.2, 0.0), 0.1)
    assert delay.whole_steps == 2 and delay.remainder_seconds == 0.0 and delay.depth == 2


# Forward map and local coordinates.
def test_right_tangent_encode_decode_roundtrip():
    *_, context = nominal_bundle(0.015)
    delta = np.linspace(-1e-3, 1e-3, context.local_dimension)
    recovered = local_poles.encode_local_state(local_poles.decode_local_state(delta, context), context)
    assert np.allclose(recovered, delta, rtol=1e-10, atol=1e-12)


def test_right_tangent_is_trim_rotation_times_exp():
    *_, context = nominal_bundle()
    delta = np.zeros(context.local_dimension)
    delta[3:6] = (0.01, -0.02, 0.03)
    decoded = local_poles.decode_local_state(delta, context)
    relative = quaternion_to_matrix(context.trim.rigid_body.orientation_xyzw).T @ quaternion_to_matrix(decoded.rigid_body.orientation_xyzw)
    assert np.allclose(relative, local_poles.so3_exp(delta[3:6]))


def test_local_step_at_zero_equals_stored_trim_defect():
    *_, trim, context = nominal_bundle(0.015)
    assert np.allclose(local_poles.local_closed_loop_step(np.zeros(context.local_dimension), context), trim.one_step_defect)


def test_nominal_augmented_forward_step_is_finite():
    *_, context = nominal_bundle(0.015)
    assert np.all(np.isfinite(local_poles.local_closed_loop_step(np.zeros(context.local_dimension), context)))


def test_zero_and_fitted_delay_issue_identical_instantaneous_command():
    *_, zero_context = nominal_bundle(0.0)
    *_, delayed_context = nominal_bundle(0.015)
    command0, _ = zero_context.controller.step(
        zero_context.trim.rigid_body,
        zero_context.reference,
        zero_context.trim.controller,
        zero_context.controller_dt,
        zero_context.trim.actuators.gimbal_angle,
    )
    command1, _ = delayed_context.controller.step(
        delayed_context.trim.rigid_body,
        delayed_context.reference,
        delayed_context.trim.controller,
        delayed_context.controller_dt,
        delayed_context.trim.actuators.gimbal_angle,
    )
    assert np.allclose(command0.thrust, command1.thrust)
    assert np.allclose(command0.gimbal_angle, command1.gimbal_angle)


# Linearization and exact classification.
def test_active_branch_analytic_jacobian_matches_central_difference():
    *_, context = nominal_bundle()
    analytic, near_kink = local_poles.analytic_closed_loop_jacobian(context)
    finite_difference = local_poles.central_difference_jacobian(context)
    relative_error = np.linalg.norm(analytic - finite_difference) / np.linalg.norm(
        finite_difference
    )
    assert not near_kink
    assert relative_error < 2e-5


def test_h_and_half_h_central_differences_agree():
    *_, context = nominal_bundle()
    full = local_poles.central_difference_jacobian(context)
    half = local_poles.central_difference_jacobian(context, divisor=2.0)
    diagnostic = local_poles.finite_difference_diagnostic(full, half)
    assert diagnostic["relative_frobenius_difference"] < 2e-5


def test_jacobian_matches_independent_fixed_step_difference():
    *_, context = nominal_bundle()
    actual = local_poles.central_difference_jacobian(context)
    dimension = context.local_dimension
    brute = np.empty_like(actual)
    step = 1e-6
    for index in range(dimension):
        delta = np.zeros(dimension)
        delta[index] = step
        brute[:, index] = (
            local_poles.local_closed_loop_step(delta, context)
            - local_poles.local_closed_loop_step(-delta, context)
        ) / (2 * step)
    assert np.allclose(actual, brute, rtol=3e-4, atol=3e-5)


def test_large_condition_number_is_not_a_stability_rejection():
    matrix = np.diag((0.5, 1e100))
    result = local_poles.classify_eigenvalues(np.linalg.eigvals(matrix))
    assert result["spectral_radius"] == 1e100
    assert not result["stable"]


@pytest.mark.parametrize(
    "value,stable,unstable,marginal",
    ((np.nextafter(1.0, 0.0), True, 0, 0), (1.0, False, 0, 1), (np.nextafter(1.0, 2.0), False, 1, 0)),
)
def test_unit_circle_classification_is_exact(value, stable, unstable, marginal):
    result = local_poles.classify_eigenvalues(np.asarray((value,), dtype=complex))
    assert result["stable"] is stable
    assert result["unstable_pole_count"] == unstable
    assert result["marginal_pole_count"] == marginal


def test_unexpected_value_error_is_not_a_sample_failure_type():
    assert ValueError not in local_poles.NUMERICAL_SAMPLE_EXCEPTIONS


def test_fd_diagnostic_does_not_return_accept_reject_flag():
    result = local_poles.finite_difference_diagnostic(np.eye(2), np.eye(2))
    assert "valid" not in result and "reject" not in result


# Simplified equal-scaling regression.
def _scalar_pid_matrix(effectiveness: float, gains: np.ndarray, dt: float = 0.01):
    p_gain, i_gain, d_gain = gains
    return np.asarray(
        (
            (1.0, dt, 0.0),
            (-dt * effectiveness * p_gain, 1.0 - dt * effectiveness * d_gain, dt * effectiveness * i_gain),
            (-dt, 0.0, 1.0),
        )
    )


def test_equal_pid_scaling_preserves_scalar_double_integrator_poles():
    b0, b = 2.0, 7.0
    gains = np.asarray((4.0, 0.5, 1.5))
    old = np.sort_complex(np.linalg.eigvals(_scalar_pid_matrix(b0, gains)))
    new = np.sort_complex(np.linalg.eigvals(_scalar_pid_matrix(b, gains * b0 / b)))
    assert np.allclose(old, new, rtol=0.0, atol=2e-15)


# Monte Carlo mechanics.
def test_zero_covariance_gives_point_mass_draws():
    samples, *_ = local_poles.draw_quotient_samples(np.zeros((13, 13)), 8, 0)
    assert np.array_equal(samples, np.zeros((8, 13)))


def test_fixed_seed_gives_bitwise_reproducible_draws():
    covariance = np.diag(np.arange(1.0, 14.0))
    first, *_ = local_poles.draw_quotient_samples(covariance, 16, 42)
    second, *_ = local_poles.draw_quotient_samples(covariance, 16, 42)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("mode", ("conservative_fusion", "overlap_corrected"))
def test_required_covariance_modes_are_accepted(mode):
    assert case_inputs(mode).sampling_coordinates.mode == "estimator_quotient"


def test_nonproduction_covariance_modes_are_not_exposed():
    assert local_poles.COVARIANCE_MODES == ("conservative_fusion", "overlap_corrected")


def test_fitted_delay_is_fixed_outside_physical_draws():
    inputs = case_inputs()
    delays = [
        local_poles.decompose_thrust_delay(inputs.result.plant.rotor_lag_seconds, 0.01)
        for _ in range(3)
    ]
    assert len({value.delay_seconds for value in delays}) == 1


def test_scale_free_sample_vector_has_no_lag_coordinate():
    assert len(local_poles.SCALE_FREE_LABELS) == 13
    assert all("lag" not in value.lower() for value in local_poles.SCALE_FREE_LABELS)


def test_raw_invalid_samples_remain_in_requested_denominator():
    summary = local_poles.stability_summary(
        np.asarray((0.5, np.nan)),
        np.asarray((0.5, np.nan)),
        np.asarray((True, False)),
        np.asarray((0, -1)),
        np.asarray((True, False)),
        2,
    )
    assert summary["pole_valid_samples"] == 1
    assert summary["pole_valid_fraction_of_requested"] == 0.5


def test_prefixes_share_one_ordered_realization():
    covariance = np.eye(13)
    full, *_ = local_poles.draw_quotient_samples(covariance, 512, 9)
    prefix, *_ = local_poles.draw_quotient_samples(covariance, 128, 9)
    assert np.array_equal(full[:128], prefix)


# Scientific labels and orchestration.
def test_outcome_label_does_not_enter_forward_context():
    assert "flight_outcome" not in local_poles.PoleContext.__dataclass_fields__


def test_case_name_does_not_enter_stability_classifier():
    assert local_poles.classify_eigenvalues((0.9,))["stable"]
    assert local_poles.classify_eigenvalues((0.9,))["stable"]


def test_wrapper_keeps_three_case_definitions_separate():
    assert tuple(three_bag.CASE_DEFINITIONS) == ("failure1", "failure2", "success")
    assert len({str(value["estimator"]) for value in three_bag.CASE_DEFINITIONS.values()}) == 3


def test_wrapper_marks_bags_as_not_averaged():
    summary = three_bag.build_summary([])
    assert summary["cases_are_averaged"] is False


def test_wrapper_outcome_metadata_has_two_crashes_and_one_success():
    outcomes = [value["outcome"] for value in three_bag.CASE_DEFINITIONS.values()]
    assert outcomes.count("crashed") == 2
    assert outcomes.count("successful") == 1
