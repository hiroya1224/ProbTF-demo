"""Strict adapter from a completed batch run to PID forecast inputs.

The estimation artifact intentionally does not contain enough information to
invent a controller or actuator model.  Callers must supply the exact recorded
controller configuration and fixed actuator/geometry assumptions for every
bag.  This module only aligns those audited inputs with MCMC samples, the
shared selected-mode MAP initial state, and the recorded reference path.
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.batch_artifact import BatchEstimationRun
from grape_param_estim.controller import ControllerConfig
from grape_param_estim.pid.predictive import (
    PidForecastInitialCondition,
    PidForecastScenario,
)
from grape_param_estim.pid.proposal import PhysicalPlantPosterior
from grape_param_estim.system import (
    ActuatorParameters,
    ActuatorState,
    ControllerState,
    GrapeGeometry,
    ReferenceState,
    RigidBodyState,
    VehicleParameters,
)


def _canonical(value: object, name: str) -> str:
    selected = str(value)
    if not selected or selected.strip() != selected or "\x00" in selected:
        raise ValueError("{} must be a canonical non-empty string".format(name))
    return selected


def _fixed_drag(value: Sequence[float], name: str) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.shape != (3,) or np.any(~np.isfinite(selected)) or np.any(selected < 0.0):
        raise ValueError("{} must contain three finite non-negative values".format(name))
    return selected.copy()


def _strings(value: np.ndarray) -> Tuple[str, ...]:
    selected = np.asarray(value)
    if selected.ndim != 1 or selected.dtype.kind not in "USiu":
        raise ValueError("artifact identifiers must be a one-dimensional array")
    return tuple(str(item) for item in selected.tolist())


def physical_posterior_from_batch_run(
    run: BatchEstimationRun,
    *,
    fixed_linear_drag: Sequence[float],
    fixed_angular_drag: Sequence[float],
    selected_mode_id: Optional[str] = None,
) -> PhysicalPlantPosterior:
    """Load equal-weight MCMC plants without fabricating fixed drag values."""

    if not isinstance(run, BatchEstimationRun):
        raise TypeError("run must be a validated BatchEstimationRun")
    if run.mcmc_samples is None:
        raise ValueError("PID evaluation requires retained MCMC samples")
    arrays = run.mcmc_samples
    sample_ids = _strings(arrays["sample_id"])
    modes = _strings(arrays["source_mode_id"])
    available_modes = tuple(dict.fromkeys(modes))
    if selected_mode_id is None:
        if len(available_modes) != 1:
            raise ValueError(
                "MCMC samples contain multiple modes; selected_mode_id is required"
            )
        selected_mode = available_modes[0]
    else:
        selected_mode = _canonical(selected_mode_id, "selected_mode_id")
        if selected_mode not in available_modes:
            raise ValueError("selected_mode_id is absent from MCMC samples")
    selected_rows = tuple(
        index for index, mode in enumerate(modes) if mode == selected_mode
    )
    if not selected_rows:
        raise ValueError("selected MCMC mode has no retained samples")
    linear_drag = _fixed_drag(fixed_linear_drag, "fixed_linear_drag")
    angular_drag = _fixed_drag(fixed_angular_drag, "fixed_angular_drag")
    parameters = tuple(
        VehicleParameters(
            mass=arrays["mass"][row],
            inertia=arrays["inertia"][row],
            cog_offset=arrays["cog"][row],
            force_effectiveness=arrays["force_effectiveness"][row],
            torque_effectiveness=arrays["torque_effectiveness"][row],
            linear_drag=linear_drag,
            angular_drag=angular_drag,
        )
        for row in selected_rows
    )
    return PhysicalPlantPosterior.from_aligned_values(
        tuple(sample_ids[row] for row in selected_rows),
        parameters,
        np.asarray(tuple(arrays["delay"][row] for row in selected_rows)),
        (selected_mode,) * len(selected_rows),
    )


@dataclass(frozen=True)
class PidBagForecastModel:
    """Exact fixed model assumptions supplied for one recorded bag."""

    bag_id: str
    controller_configuration: ControllerConfig
    controller_nominal_parameters: VehicleParameters
    controller_geometry: GrapeGeometry
    plant_geometry: GrapeGeometry
    actuator_parameters: ActuatorParameters
    roll_pitch_integration_active: bool
    maximum_reference_age_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "bag_id", _canonical(self.bag_id, "bag_id"))
        for value, expected, name in (
            (self.controller_configuration, ControllerConfig, "controller_configuration"),
            (
                self.controller_nominal_parameters,
                VehicleParameters,
                "controller_nominal_parameters",
            ),
            (self.controller_geometry, GrapeGeometry, "controller_geometry"),
            (self.plant_geometry, GrapeGeometry, "plant_geometry"),
            (self.actuator_parameters, ActuatorParameters, "actuator_parameters"),
        ):
            if not isinstance(value, expected):
                raise TypeError("{} has the wrong type".format(name))
        if not isinstance(self.roll_pitch_integration_active, (bool, np.bool_)):
            raise TypeError("roll_pitch_integration_active must be boolean")
        maximum_age = float(self.maximum_reference_age_seconds)
        if not np.isfinite(maximum_age) or maximum_age <= 0.0:
            raise ValueError("maximum_reference_age_seconds must be positive")
        object.__setattr__(
            self,
            "roll_pitch_integration_active",
            bool(self.roll_pitch_integration_active),
        )
        object.__setattr__(self, "maximum_reference_age_seconds", maximum_age)


def _references_at_knots(
    arrays: Mapping[str, np.ndarray],
    maximum_age_seconds: float,
) -> Tuple[ReferenceState, ...]:
    knots = arrays["knot_time"]
    times = arrays["reference_time"]
    result = []
    for knot in knots:
        index = int(np.searchsorted(times, knot, side="right") - 1)
        if index < 0:
            raise ValueError("recorded reference has no causal value at first knot")
        age = float(knot - times[index])
        if age < -1.0e-12 or age > maximum_age_seconds + 1.0e-12:
            raise ValueError("recorded reference exceeds its maximum causal age")
        result.append(
            ReferenceState(
                position=arrays["reference_position"][index],
                linear_velocity=arrays["reference_linear_velocity"][index],
                linear_acceleration=arrays["reference_linear_acceleration"][index],
                rpy=arrays["reference_rpy"][index],
                angular_velocity=arrays["reference_angular_velocity"][index],
                angular_acceleration=arrays["reference_angular_acceleration"][index],
            )
        )
    return tuple(result)


def _initial_condition(
    bag_arrays: Mapping[str, np.ndarray],
    sample_id: str,
    roll_pitch_integration_active: bool,
) -> PidForecastInitialCondition:
    return PidForecastInitialCondition(
        sample_id=sample_id,
        rigid_body_state=RigidBodyState(
            bag_arrays["map_position"][0],
            bag_arrays["map_orientation_xyzw"][0],
            bag_arrays["map_linear_velocity"][0],
            bag_arrays["map_angular_velocity"][0],
        ),
        controller_state=ControllerState(
            bag_arrays["map_controller_integral"][0],
            roll_pitch_integration_active=roll_pitch_integration_active,
        ),
        actuator_state=ActuatorState(
            bag_arrays["map_actuator_thrust"][0],
            bag_arrays["map_actuator_gimbal"][0],
        ),
        source="shared_selected_mode_map_initial",
    )


def forecast_scenarios_from_batch_run(
    run: BatchEstimationRun,
    posterior: PhysicalPlantPosterior,
    bag_models: Sequence[PidBagForecastModel],
) -> Tuple[PidForecastScenario, ...]:
    """Build all bag scenarios from one explicit selected-mode MAP state."""

    if not isinstance(run, BatchEstimationRun):
        raise TypeError("run must be a validated BatchEstimationRun")
    if not isinstance(posterior, PhysicalPlantPosterior):
        raise TypeError("posterior must be PhysicalPlantPosterior")
    models = tuple(bag_models)
    if any(not isinstance(value, PidBagForecastModel) for value in models):
        raise TypeError("bag_models must contain PidBagForecastModel values")
    model_by_bag = {value.bag_id: value for value in models}
    bag_ids = tuple(str(value) for value in run.manifest["selected_bag_ids"])
    if len(model_by_bag) != len(models) or set(model_by_bag) != set(bag_ids):
        raise ValueError("bag model IDs must exactly match the estimation run")
    scenarios = []
    for bag_id in bag_ids:
        arrays = run.bags[bag_id]
        model = model_by_bag[bag_id]
        scenarios.append(
            PidForecastScenario(
                bag_id=bag_id,
                times=arrays["knot_time"],
                references=_references_at_knots(
                    arrays, model.maximum_reference_age_seconds
                ),
                initial_conditions=tuple(
                    _initial_condition(
                        arrays,
                        sample.sample_id,
                        model.roll_pitch_integration_active,
                    )
                    for sample in posterior.samples
                ),
                controller_configuration=model.controller_configuration,
                controller_nominal_parameters=model.controller_nominal_parameters,
                controller_geometry=model.controller_geometry,
                plant_geometry=model.plant_geometry,
                actuator_parameters=model.actuator_parameters,
                provenance=(
                    ("estimation_run_id", str(run.manifest["run_id"])),
                    ("reference_policy", "causal_zero_order_hold"),
                    (
                        "initial_condition_policy",
                        "shared_selected_mode_map_initial",
                    ),
                ),
            )
        )
    return tuple(scenarios)


__all__ = [
    "PidBagForecastModel",
    "forecast_scenarios_from_batch_run",
    "physical_posterior_from_batch_run",
]
