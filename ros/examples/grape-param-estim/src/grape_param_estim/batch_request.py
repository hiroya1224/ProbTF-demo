"""Strict GUI-to-worker request contract for sparse batch estimation."""

from dataclasses import dataclass
from numbers import Real
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple, Union

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactValidationError,
    read_json,
    request_fingerprint,
)
from grape_param_estim.batch.factors.dynamics_factor import (
    BODY_WRENCH_QUANTITY,
    SPECIFIC_ACCELERATION_QUANTITY,
)
from grape_param_estim.batch.laplace_em import QIntervalModel
from grape_param_estim.batch_artifact import file_sha256


BATCH_ESTIMATION_REQUEST_SCHEMA = (
    "grape-param-estim/batch-estimation-request/v1"
)
RUN_MODES = ("estimate_only", "estimate_and_sample")
OBSERVATION_FACTOR_NAMES = (
    "pose",
    "velocity",
    "gyro",
    "accelerometer",
    "issued_rotor_command",
    "issued_gimbal_command",
    "actual_gimbal_position",
    "controller_integral",
)
COVARIANCE_REPRESENTATIONS = ("diagonal", "full")
COVARIANCE_SOURCES = (
    "preflight_static_interval",
    "message_covariance",
    "sensor_specification",
    "project_configuration",
    "numerical_tolerance",
    "reconstruction_tolerance",
    "discretization_model",
)


# Covariance coordinates are part of the request protocol, not display labels.
# The numeric values use products of the listed residual-coordinate units: a
# diagonal entry is in unit[i]^2 and a full entry (i, j) is in
# unit[i] * unit[j].  Pose deliberately has two independent blocks because the
# batch graph represents position and SO(3)-tangent observations as separate
# factors; accepting an unsupported 6-D cross covariance would silently lose
# information.
OBSERVATION_COVARIANCE_BLOCKS = MappingProxyType({
    "pose": MappingProxyType({
        "position_observation": (
            ("position_x", "position_y", "position_z"),
            ("m", "m", "m"),
        ),
        "orientation_observation": (
            (
                "orientation_tangent_x",
                "orientation_tangent_y",
                "orientation_tangent_z",
            ),
            ("rad", "rad", "rad"),
        ),
    }),
    "velocity": MappingProxyType({
        "velocity_observation": (
            ("velocity_world_x", "velocity_world_y", "velocity_world_z"),
            ("m/s", "m/s", "m/s"),
        ),
    }),
    "gyro": MappingProxyType({
        "gyro_observation": (
            (
                "angular_velocity_sensor_x",
                "angular_velocity_sensor_y",
                "angular_velocity_sensor_z",
            ),
            ("rad/s", "rad/s", "rad/s"),
        ),
    }),
    "accelerometer": MappingProxyType({
        "accelerometer_observation": (
            (
                "specific_force_sensor_x",
                "specific_force_sensor_y",
                "specific_force_sensor_z",
            ),
            ("m/s^2", "m/s^2", "m/s^2"),
        ),
    }),
    "issued_rotor_command": MappingProxyType({
        "issued_thrust_observation": (
            tuple("rotor_{}_thrust".format(index) for index in range(1, 5)),
            ("N", "N", "N", "N"),
        ),
    }),
    "issued_gimbal_command": MappingProxyType({
        "issued_gimbal_observation": (
            tuple("gimbal_{}_angle".format(index) for index in range(1, 5)),
            ("rad", "rad", "rad", "rad"),
        ),
    }),
    "actual_gimbal_position": MappingProxyType({
        "actual_gimbal_observation": (
            tuple("gimbal_{}_angle".format(index) for index in range(1, 5)),
            ("rad", "rad", "rad", "rad"),
        ),
    }),
    "controller_integral": MappingProxyType({
        "controller_integral_observation": (
            (
                "position_integral_x",
                "position_integral_y",
                "position_integral_z",
                "attitude_integral_x",
                "attitude_integral_y",
                "attitude_integral_z",
            ),
            ("m*s", "m*s", "m*s", "rad*s", "rad*s", "rad*s"),
        ),
    }),
})


# These five covariances are not sensor observations and cannot be disabled:
# the fixed graph always contains the corresponding controller, actuator, and
# kinematic consistency factors.  Keeping them per bag permits a different
# discretization tolerance when knot periods differ.
FIXED_FACTOR_COVARIANCE_BLOCKS = MappingProxyType({
    "controller_integral_transition": (
        (
            "position_integral_x",
            "position_integral_y",
            "position_integral_z",
            "attitude_integral_x",
            "attitude_integral_y",
            "attitude_integral_z",
        ),
        ("m*s", "m*s", "m*s", "rad*s", "rad*s", "rad*s"),
    ),
    "actuator_thrust_transition": (
        tuple("rotor_{}_thrust".format(index) for index in range(1, 5)),
        ("N", "N", "N", "N"),
    ),
    "actuator_gimbal_transition": (
        tuple("gimbal_{}_angle".format(index) for index in range(1, 5)),
        ("rad", "rad", "rad", "rad"),
    ),
    "position_kinematic": (
        ("position_defect_x", "position_defect_y", "position_defect_z"),
        ("m", "m", "m"),
    ),
    "orientation_kinematic": (
        (
            "orientation_defect_x",
            "orientation_defect_y",
            "orientation_defect_z",
        ),
        ("rad", "rad", "rad"),
    ),
})


# These Gaussian priors anchor otherwise gauge-like bag-local trajectory
# variables at the first knot.  They are deliberately distinct from sensor
# covariances: reusing an observation covariance here would count the same
# evidence twice and would conflate measurement noise with initialization
# uncertainty.  Prior means come from the audited initialization result (and
# the preflight gyro bias); the request supplies only their independent
# covariance and provenance.
INITIAL_STATE_PRIOR_COVARIANCE_BLOCKS = MappingProxyType({
    "gyro_bias": (
        ("gyro_bias_sensor_x", "gyro_bias_sensor_y", "gyro_bias_sensor_z"),
        ("rad/s", "rad/s", "rad/s"),
    ),
    "position": (
        ("position_x", "position_y", "position_z"),
        ("m", "m", "m"),
    ),
    "orientation": (
        (
            "orientation_tangent_x",
            "orientation_tangent_y",
            "orientation_tangent_z",
        ),
        ("rad", "rad", "rad"),
    ),
    "linear_velocity": (
        ("linear_velocity_x", "linear_velocity_y", "linear_velocity_z"),
        ("m/s", "m/s", "m/s"),
    ),
    "angular_velocity": (
        ("angular_velocity_x", "angular_velocity_y", "angular_velocity_z"),
        ("rad/s", "rad/s", "rad/s"),
    ),
    "controller_integral": (
        (
            "position_integral_x",
            "position_integral_y",
            "position_integral_z",
            "attitude_integral_x",
            "attitude_integral_y",
            "attitude_integral_z",
        ),
        ("m*s", "m*s", "m*s", "rad*s", "rad*s", "rad*s"),
    ),
    "actuator_thrust": (
        tuple("rotor_{}_thrust".format(index) for index in range(1, 5)),
        ("N", "N", "N", "N"),
    ),
    "gimbal_angle": (
        tuple("gimbal_{}_angle".format(index) for index in range(1, 5)),
        ("rad", "rad", "rad", "rad"),
    ),
})


_BAG_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def _error(location: str, message: str) -> None:
    raise ArtifactValidationError("{} {}".format(location, message))


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _error(location, "must be an object")
    return value


def _keys(
    value: Mapping[str, Any],
    expected: Sequence[str],
    location: str,
) -> None:
    supplied = set(value)
    required = set(expected)
    missing = sorted(required - supplied)
    unknown = sorted(supplied - required)
    if missing:
        _error(location, "is missing keys: {}".format(", ".join(missing)))
    if unknown:
        _error(location, "has unknown keys: {}".format(", ".join(unknown)))


def _string(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
    ):
        _error(location, "must be a canonical non-empty string")
    return value


def _choice(value: Any, choices: Sequence[str], location: str) -> str:
    selected = _string(value, location)
    if selected not in choices:
        _error(
            location,
            "must be one of {}".format(", ".join(repr(item) for item in choices)),
        )
    return selected


def _boolean(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        _error(location, "must be boolean")
    return value


def _integer(value: Any, minimum: int, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _error(location, "must be an integer >= {}".format(minimum))
    return value


def _number(
    value: Any,
    location: str,
    *,
    minimum: float = None,
    strictly_greater: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _error(location, "must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        _error(location, "must be a finite number")
    if minimum is not None:
        invalid = result <= minimum if strictly_greater else result < minimum
        if invalid:
            operator = ">" if strictly_greater else ">="
            _error(location, "must be {} {}".format(operator, minimum))
    return result


def _vector(
    value: Any,
    size: int,
    location: str,
    *,
    positive: bool = False,
) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(
            isinstance(item, bool) or not isinstance(item, Real)
            for item in value
        )
    ):
        _error(location, "must contain {} finite numbers".format(size))
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        _error(location, "must contain {} finite numbers".format(size))
    if positive and np.any(result <= 0.0):
        _error(location, "must contain positive numbers")
    return result


def _interval(value: Any, location: str) -> Tuple[float, float]:
    result = _vector(value, 2, location)
    if result[0] < 0.0 or result[1] <= result[0]:
        _error(location, "must be non-negative and strictly increasing")
    return float(result[0]), float(result[1])


def _strings(
    value: Any,
    size: int,
    location: str,
    *,
    unique: bool,
) -> Tuple[str, ...]:
    if not isinstance(value, list) or len(value) != size:
        _error(location, "must contain exactly {} strings".format(size))
    result = tuple(
        _string(item, "{}[{}]".format(location, index))
        for index, item in enumerate(value)
    )
    if unique and len(set(result)) != len(result):
        _error(location, "must not contain duplicates")
    return result


def _absolute_path(value: Any, location: str, must_be_file: bool) -> Path:
    selected = Path(_string(value, location))
    if not selected.is_absolute() or ".." in selected.parts:
        _error(location, "must be an absolute path without '..'")
    resolved = selected.resolve()
    if resolved == Path(resolved.anchor):
        _error(location, "cannot name a filesystem root")
    if must_be_file and not resolved.is_file():
        _error(location, "must name an existing file")
    if not must_be_file and resolved.exists() and not resolved.is_dir():
        _error(location, "must name a directory or a new path")
    return resolved


def _validate_covariance(
    value: Any,
    coordinates: Tuple[str, ...],
    units: Tuple[str, ...],
    location: str,
) -> None:
    covariance = _mapping(value, location)
    _keys(
        covariance,
        ("source", "representation", "coordinates", "units", "values"),
        location,
    )
    _choice(covariance["source"], COVARIANCE_SOURCES, location + ".source")
    representation = _choice(
        covariance["representation"],
        COVARIANCE_REPRESENTATIONS,
        location + ".representation",
    )
    supplied_coordinates = _strings(
        covariance["coordinates"],
        len(coordinates),
        location + ".coordinates",
        unique=True,
    )
    if supplied_coordinates != coordinates:
        _error(
            location + ".coordinates",
            "must equal {} in residual order".format(list(coordinates)),
        )
    supplied_units = _strings(
        covariance["units"],
        len(units),
        location + ".units",
        unique=False,
    )
    if supplied_units != units:
        _error(
            location + ".units",
            "must equal {} in residual order".format(list(units)),
        )

    dimension = len(coordinates)
    if representation == "diagonal":
        _vector(
            covariance["values"],
            dimension,
            location + ".values",
            positive=True,
        )
        return

    rows = covariance["values"]
    if (
        not isinstance(rows, list)
        or len(rows) != dimension
        or any(
            not isinstance(row, list)
            or len(row) != dimension
            or any(
                isinstance(item, bool) or not isinstance(item, Real)
                for item in row
            )
            for row in rows
        )
    ):
        _error(
            location + ".values",
            "must be a finite {} by {} matrix".format(dimension, dimension),
        )
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (dimension, dimension) or not np.all(
        np.isfinite(matrix)
    ):
        _error(
            location + ".values",
            "must be a finite {} by {} matrix".format(dimension, dimension),
        )
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        _error(location + ".values", "must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        _error(location + ".values", "must be positive definite")


def _validate_covariance_blocks(
    value: Any,
    expected: Mapping[str, Tuple[Tuple[str, ...], Tuple[str, ...]]],
    location: str,
) -> None:
    blocks = _mapping(value, location)
    _keys(blocks, tuple(expected), location)
    for name, (coordinates, units) in expected.items():
        _validate_covariance(
            blocks[name], coordinates, units, location + "." + name
        )


def _validate_factor(value: Any, name: str, location: str) -> None:
    factor = _mapping(value, location)
    _keys(
        factor,
        ("enabled", "disabled_reason", "covariances"),
        location,
    )
    enabled = _boolean(factor["enabled"], location + ".enabled")
    reason = factor["disabled_reason"]
    if enabled:
        if reason is not None:
            _error(location + ".disabled_reason", "must be null when enabled")
        _validate_covariance_blocks(
            factor["covariances"],
            OBSERVATION_COVARIANCE_BLOCKS[name],
            location + ".covariances",
        )
        return
    _string(reason, location + ".disabled_reason")
    if factor["covariances"] is not None:
        _error(
            location + ".covariances",
            "must be null when the factor is disabled",
        )


def _validate_bags(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        _error("request.bags", "must be a non-empty list")
    bag_ids = []
    paths = []
    for index, item in enumerate(value):
        location = "request.bags[{}]".format(index)
        bag = _mapping(item, location)
        _keys(
            bag,
            (
                "bag_id",
                "path",
                "sha256",
                "interval_seconds",
                "observation_factors",
                "fixed_factor_covariances",
                "initial_state_prior_covariances",
            ),
            location,
        )
        bag_id = _string(bag["bag_id"], location + ".bag_id")
        if _BAG_ID.fullmatch(bag_id) is None:
            _error(location + ".bag_id", "is not safe for artifact paths")
        path = _absolute_path(bag["path"], location + ".path", True)
        digest = _string(bag["sha256"], location + ".sha256")
        if _SHA256.fullmatch(digest) is None:
            _error(location + ".sha256", "must be sha256:<64 lowercase hex>")
        actual_digest = file_sha256(path)
        if digest != actual_digest:
            _error(
                location + ".sha256",
                "does not match the selected bag (actual {})".format(
                    actual_digest
                ),
            )
        _interval(bag["interval_seconds"], location + ".interval_seconds")
        factors = _mapping(
            bag["observation_factors"], location + ".observation_factors"
        )
        _keys(factors, OBSERVATION_FACTOR_NAMES, location + ".observation_factors")
        for name in OBSERVATION_FACTOR_NAMES:
            _validate_factor(
                factors[name],
                name,
                location + ".observation_factors." + name,
            )
        _validate_covariance_blocks(
            bag["fixed_factor_covariances"],
            FIXED_FACTOR_COVARIANCE_BLOCKS,
            location + ".fixed_factor_covariances",
        )
        _validate_covariance_blocks(
            bag["initial_state_prior_covariances"],
            INITIAL_STATE_PRIOR_COVARIANCE_BLOCKS,
            location + ".initial_state_prior_covariances",
        )
        bag_ids.append(bag_id)
        paths.append(path)
    if len(set(bag_ids)) != len(bag_ids):
        _error("request.bags", "contains duplicate bag IDs")
    if len(set(paths)) != len(paths):
        _error("request.bags", "contains duplicate bag paths")
    return tuple(bag_ids)


def _validate_q(value: Any) -> None:
    q = _mapping(value, "request.q")
    _keys(
        q,
        (
            "residual_quantity",
            "interval_model",
            "component_names",
            "component_units",
            "initial_diagonal",
            "floor_diagonal",
        ),
        "request.q",
    )
    _choice(
        q["residual_quantity"],
        (BODY_WRENCH_QUANTITY, SPECIFIC_ACCELERATION_QUANTITY),
        "request.q.residual_quantity",
    )
    _choice(
        q["interval_model"],
        tuple(item.value for item in QIntervalModel),
        "request.q.interval_model",
    )
    _strings(q["component_names"], 6, "request.q.component_names", unique=True)
    _strings(q["component_units"], 6, "request.q.component_units", unique=False)
    initial = _vector(
        q["initial_diagonal"],
        6,
        "request.q.initial_diagonal",
        positive=True,
    )
    floor = _vector(
        q["floor_diagonal"],
        6,
        "request.q.floor_diagonal",
        positive=True,
    )
    if np.any(initial < floor):
        _error("request.q.initial_diagonal", "cannot be below the Q floor")


def _validate_parameter_prior(value: Any) -> None:
    prior = _mapping(value, "request.parameter_prior")
    _keys(prior, ("kind", "mean_coordinate", "covariance"), "request.parameter_prior")
    _choice(prior["kind"], ("gaussian",), "request.parameter_prior.kind")
    _vector(prior["mean_coordinate"], 18, "request.parameter_prior.mean_coordinate")
    covariance_rows = prior["covariance"]
    if (
        not isinstance(covariance_rows, list)
        or len(covariance_rows) != 18
        or any(
            not isinstance(row, list)
            or len(row) != 18
            or any(
                isinstance(item, bool) or not isinstance(item, Real)
                for item in row
            )
            for row in covariance_rows
        )
    ):
        _error("request.parameter_prior.covariance", "must be a finite 18 by 18 matrix")
    covariance = np.asarray(covariance_rows, dtype=float)
    if covariance.shape != (18, 18) or not np.all(np.isfinite(covariance)):
        _error("request.parameter_prior.covariance", "must be a finite 18 by 18 matrix")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-12):
        _error("request.parameter_prior.covariance", "must be symmetric")
    if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
        _error("request.parameter_prior.covariance", "must be positive definite")


def _validate_delay(value: Any) -> None:
    delay = _mapping(value, "request.delay")
    _keys(
        delay,
        (
            "prior_kind",
            "bounds_seconds",
            "initial_seconds",
            "coarse_grid_points",
            "refinement_tolerance_seconds",
            "maximum_refinement_evaluations",
        ),
        "request.delay",
    )
    _choice(delay["prior_kind"], ("uniform",), "request.delay.prior_kind")
    lower, upper = _interval(delay["bounds_seconds"], "request.delay.bounds_seconds")
    initial = _number(delay["initial_seconds"], "request.delay.initial_seconds", minimum=0.0)
    if initial < lower or initial > upper:
        _error("request.delay.initial_seconds", "must be inside delay bounds")
    _integer(delay["coarse_grid_points"], 3, "request.delay.coarse_grid_points")
    tolerance = _number(
        delay["refinement_tolerance_seconds"],
        "request.delay.refinement_tolerance_seconds",
        minimum=0.0,
        strictly_greater=True,
    )
    if tolerance >= upper - lower:
        _error("request.delay.refinement_tolerance_seconds", "must be smaller than bounds")
    _integer(
        delay["maximum_refinement_evaluations"],
        2,
        "request.delay.maximum_refinement_evaluations",
    )


def _validate_knot_and_interpolation(knot_value: Any, interpolation_value: Any) -> None:
    knot = _mapping(knot_value, "request.knot_policy")
    _keys(
        knot,
        (
            "period_seconds",
            "origin",
            "maximum_measurement_gap_seconds",
        ),
        "request.knot_policy",
    )
    _number(
        knot["period_seconds"],
        "request.knot_policy.period_seconds",
        minimum=0.0,
        strictly_greater=True,
    )
    _choice(knot["origin"], ("interval_start",), "request.knot_policy.origin")
    _number(
        knot["maximum_measurement_gap_seconds"],
        "request.knot_policy.maximum_measurement_gap_seconds",
        minimum=0.0,
        strictly_greater=True,
    )
    interpolation = _mapping(
        interpolation_value, "request.interpolation_policy"
    )
    _keys(
        interpolation,
        ("euclidean", "orientation", "command", "allow_extrapolation"),
        "request.interpolation_policy",
    )
    _choice(interpolation["euclidean"], ("linear",), "request.interpolation_policy.euclidean")
    _choice(
        interpolation["orientation"],
        ("so3_geodesic",),
        "request.interpolation_policy.orientation",
    )
    _choice(
        interpolation["command"],
        ("zoh_record_issue_time",),
        "request.interpolation_policy.command",
    )
    if _boolean(
        interpolation["allow_extrapolation"],
        "request.interpolation_policy.allow_extrapolation",
    ):
        _error("request.interpolation_policy.allow_extrapolation", "must be false")


def _validate_numeric_settings(
    value: Any,
    location: str,
    integer_minima: Mapping[str, int],
    nonnegative: Sequence[str],
    positive: Sequence[str],
) -> None:
    settings = _mapping(value, location)
    expected = tuple(integer_minima) + tuple(nonnegative) + tuple(positive)
    _keys(settings, expected, location)
    for name, minimum in integer_minima.items():
        _integer(settings[name], minimum, location + "." + name)
    for name in nonnegative:
        _number(settings[name], location + "." + name, minimum=0.0)
    for name in positive:
        _number(
            settings[name],
            location + "." + name,
            minimum=0.0,
            strictly_greater=True,
        )


def _validate_solver(value: Any) -> None:
    _validate_numeric_settings(
        value,
        "request.solver_settings",
        {
            "maximum_iterations": 1,
            "maximum_factorization_retries": 1,
            "maximum_model_evaluation_retries": 1,
        },
        (
            "acceptance_ratio",
            "gradient_tolerance",
            "scaled_step_tolerance",
            "relative_objective_tolerance",
        ),
        ("initial_damping", "minimum_damping", "maximum_damping"),
    )
    settings = value
    if not (
        settings["minimum_damping"]
        <= settings["initial_damping"]
        <= settings["maximum_damping"]
    ):
        _error("request.solver_settings", "has inconsistent damping bounds")
    if settings["acceptance_ratio"] >= 1.0:
        _error("request.solver_settings.acceptance_ratio", "must be below one")


def _validate_em(value: Any) -> None:
    _validate_numeric_settings(
        value,
        "request.em_settings",
        {
            "maximum_iterations": 1,
            "minimum_iterations": 1,
            "maximum_repeated_q_rejections": 1,
            "maximum_repeated_lag_profile_failures": 1,
        },
        (
            "log_q_tolerance",
            "lag_tolerance",
            "map_objective_tolerance",
            "marginal_objective_tolerance",
            "q_acceptance_objective_tolerance",
        ),
        ("q_minimum_alpha",),
    )
    settings = value
    if settings["minimum_iterations"] > settings["maximum_iterations"]:
        _error("request.em_settings", "minimum iterations exceed maximum")
    if settings["q_minimum_alpha"] > 1.0:
        _error("request.em_settings.q_minimum_alpha", "must be at most one")


def _validate_mcmc(value: Any, run_mode: str) -> None:
    settings = _mapping(value, "request.mcmc_settings")
    enabled = settings.get("enabled")
    _boolean(enabled, "request.mcmc_settings.enabled")
    if not enabled:
        _keys(settings, ("enabled",), "request.mcmc_settings")
        if run_mode != "estimate_only":
            _error("request.mcmc_settings.enabled", "must be true for estimate_and_sample")
        return
    expected = (
        "enabled",
        "chain_count",
        "warmup_steps",
        "retained_draws",
        "thinning",
        "random_seed",
        "local_scale",
        "exact_ridge_scale",
        "near_ridge_scale",
        "identified_scale",
        "delay_scale_seconds",
        "near_relative_threshold",
        "rhat_threshold",
        "minimum_effective_sample_size",
    )
    _keys(settings, expected, "request.mcmc_settings")
    for name, minimum in (
        ("chain_count", 2),
        ("warmup_steps", 0),
        ("retained_draws", 4),
        ("thinning", 1),
        ("random_seed", 0),
    ):
        _integer(settings[name], minimum, "request.mcmc_settings." + name)
    for name in expected[6:]:
        _number(
            settings[name],
            "request.mcmc_settings." + name,
            minimum=0.0,
            strictly_greater=True,
        )
    if settings["rhat_threshold"] <= 1.0:
        _error("request.mcmc_settings.rhat_threshold", "must exceed one")
    if run_mode != "estimate_and_sample":
        _error("request.mcmc_settings.enabled", "must be false for estimate_only")


def _validate_modes(value: Any, bag_ids: Tuple[str, ...]) -> None:
    if not isinstance(value, list) or not value:
        _error("request.mode_hypotheses", "must be a non-empty list")
    mode_ids = []
    for index, item in enumerate(value):
        location = "request.mode_hypotheses[{}]".format(index)
        mode = _mapping(item, location)
        _keys(mode, ("mode_id", "bag_schedules"), location)
        mode_ids.append(_string(mode["mode_id"], location + ".mode_id"))
        schedules = _mapping(mode["bag_schedules"], location + ".bag_schedules")
        if set(schedules) != set(bag_ids):
            _error(location + ".bag_schedules", "must exactly match bag IDs")
        for bag_id in bag_ids:
            schedule_location = location + ".bag_schedules." + bag_id
            schedule = _mapping(schedules[bag_id], schedule_location)
            _keys(
                schedule,
                ("flight_state_source", "integration_gate_source"),
                schedule_location,
            )
            _choice(
                schedule["flight_state_source"],
                ("recorded_causal_schedule",),
                schedule_location + ".flight_state_source",
            )
            _choice(
                schedule["integration_gate_source"],
                ("deterministic_replay", "recorded_pid_debug"),
                schedule_location + ".integration_gate_source",
            )
    if len(set(mode_ids)) != len(mode_ids):
        _error("request.mode_hypotheses", "contains duplicate mode IDs")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class BatchEstimationRequest:
    source_path: Path
    payload: Mapping[str, Any]
    fingerprint: str
    bag_ids: Tuple[str, ...]
    output_directory: Path


def validate_batch_estimation_request(
    payload: Mapping[str, Any],
    source_path: Union[str, Path] = "<memory>",
) -> BatchEstimationRequest:
    """Validate one parsed request without supplying scientific defaults."""

    request = _mapping(payload, "request")
    expected = (
        "schema",
        "run_id",
        "run_mode",
        "resume",
        "output_directory",
        "bags",
        "q",
        "parameter_prior",
        "delay",
        "knot_policy",
        "interpolation_policy",
        "controller_snapshot_policy",
        "mode_hypotheses",
        "solver_settings",
        "em_settings",
        "mcmc_settings",
    )
    _keys(request, expected, "request")
    _choice(
        request["schema"],
        (BATCH_ESTIMATION_REQUEST_SCHEMA,),
        "request.schema",
    )
    _string(request["run_id"], "request.run_id")
    run_mode = _choice(request["run_mode"], RUN_MODES, "request.run_mode")
    _boolean(request["resume"], "request.resume")
    output_directory = _absolute_path(
        request["output_directory"], "request.output_directory", False
    )
    bag_ids = _validate_bags(request["bags"])
    _validate_q(request["q"])
    _validate_parameter_prior(request["parameter_prior"])
    _validate_delay(request["delay"])
    _validate_knot_and_interpolation(
        request["knot_policy"], request["interpolation_policy"]
    )
    controller = _mapping(
        request["controller_snapshot_policy"],
        "request.controller_snapshot_policy",
    )
    _keys(
        controller,
        ("source", "require_constant_within_interval"),
        "request.controller_snapshot_policy",
    )
    _choice(
        controller["source"],
        ("bag_startup_parameter_updates",),
        "request.controller_snapshot_policy.source",
    )
    if not _boolean(
        controller["require_constant_within_interval"],
        "request.controller_snapshot_policy.require_constant_within_interval",
    ):
        _error(
            "request.controller_snapshot_policy.require_constant_within_interval",
            "must be true",
        )
    _validate_modes(request["mode_hypotheses"], bag_ids)
    _validate_solver(request["solver_settings"])
    _validate_em(request["em_settings"])
    _validate_mcmc(request["mcmc_settings"], run_mode)
    frozen = _freeze(request)
    return BatchEstimationRequest(
        source_path=Path(source_path),
        payload=frozen,
        fingerprint=request_fingerprint(request),
        bag_ids=bag_ids,
        output_directory=output_directory,
    )


def load_batch_estimation_request(
    path: Union[str, Path],
) -> BatchEstimationRequest:
    """Read finite duplicate-free JSON and validate the sole request schema."""

    source = Path(path).expanduser().resolve()
    return validate_batch_estimation_request(read_json(source), source)


__all__ = [
    "BATCH_ESTIMATION_REQUEST_SCHEMA",
    "BatchEstimationRequest",
    "COVARIANCE_REPRESENTATIONS",
    "COVARIANCE_SOURCES",
    "FIXED_FACTOR_COVARIANCE_BLOCKS",
    "INITIAL_STATE_PRIOR_COVARIANCE_BLOCKS",
    "OBSERVATION_FACTOR_NAMES",
    "OBSERVATION_COVARIANCE_BLOCKS",
    "RUN_MODES",
    "load_batch_estimation_request",
    "validate_batch_estimation_request",
]
