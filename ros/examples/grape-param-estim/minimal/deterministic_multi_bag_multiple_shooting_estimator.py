#!/usr/bin/env python3
"""Joint smooth-lag multiple shooting for several recorded flights."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares

import deterministic_estimator as baseline
import deterministic_multiple_shooting_estimator as strict
import deterministic_smooth_lag_multiple_shooting_estimator as smooth
from grape_param_estim.real_rosbag import load_flight_data


SCHEMA = "grape-param-estim/minimal-deterministic-multi-bag-shooting/v1"
OUTPUT_SUBDIRECTORY = "deterministic_multiple_shooting_multi"
GLOBAL_DIMENSION = smooth.GLOBAL_DIMENSION
DELAY_INDEX = smooth.DELAY_INDEX
_SAFE_BAG_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class BagSpecification:
    bag_id: str
    path: Path
    start: float
    end: float
    weight: float


@dataclass(frozen=True)
class MultiBagConfig:
    bags: tuple[BagSpecification, ...]
    initial_delay_seconds: float


def load_multi_bag_config(path: Path) -> MultiBagConfig:
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise ValueError("multi-bag config does not exist: {}".format(config_path))
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("could not read multi-bag JSON config") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("bags"), list):
        raise ValueError("multi-bag config must contain a bags array")
    raw_bags = raw["bags"]
    if not raw_bags:
        raise ValueError("multi-bag config must contain at least one bag")
    specifications: list[BagSpecification] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_bags):
        if not isinstance(item, dict):
            raise ValueError("bag entry {} must be an object".format(index))
        try:
            bag_id = str(item["id"])
            raw_path = Path(str(item["path"])).expanduser()
            start = float(item["start"])
            end = float(item["end"])
            weight = float(item.get("weight", 1.0))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("bag entry {} is incomplete".format(index)) from error
        if not _SAFE_BAG_ID.fullmatch(bag_id) or bag_id in seen_ids:
            raise ValueError("bag IDs must be unique safe directory names")
        if (
            not np.isfinite(start)
            or not np.isfinite(end)
            or start >= end
            or not np.isfinite(weight)
            or weight <= 0.0
        ):
            raise ValueError("bag interval and weight must be finite and valid")
        resolved_path = (
            raw_path.resolve()
            if raw_path.is_absolute()
            else (config_path.parent / raw_path).resolve()
        )
        if not resolved_path.is_file():
            raise ValueError("bag does not exist: {}".format(resolved_path))
        specifications.append(
            BagSpecification(
                bag_id=bag_id,
                path=resolved_path,
                start=start,
                end=end,
                weight=weight,
            )
        )
        seen_ids.add(bag_id)
    try:
        initial_delay = float(raw.get("initial_delay_seconds", 0.01))
    except (TypeError, ValueError) as error:
        raise ValueError("initial_delay_seconds must be numeric") from error
    if not np.isfinite(initial_delay) or initial_delay < 0.0:
        raise ValueError("initial_delay_seconds must be finite and nonnegative")
    return MultiBagConfig(tuple(specifications), initial_delay)


@dataclass(frozen=True)
class BagShootingBlock:
    specification: BagSpecification
    normalized_weight: float
    problem: strict.MultipleShootingProblem

    def __post_init__(self) -> None:
        weight = float(self.normalized_weight)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("normalized bag weight must be positive")
        object.__setattr__(self, "normalized_weight", weight)

    @property
    def node_variable_dimension(self) -> int:
        return self.problem.variable_dimension - self.problem.global_dimension


@dataclass(frozen=True)
class BagRollout:
    sensor_position: np.ndarray
    sensor_orientation_xyzw: np.ndarray
    residual: np.ndarray

    @property
    def loss(self) -> float:
        return 0.5 * float(self.residual @ self.residual)


@dataclass(frozen=True)
class JointProblemEvaluation:
    data_residual: np.ndarray
    data_jacobian: np.ndarray
    continuity_residual: np.ndarray
    continuity_jacobian: np.ndarray
    bag_evaluations: tuple[strict.ProblemEvaluation, ...]
    decoded: Any


@dataclass(frozen=True)
class JointSolution:
    delay: float
    coordinate: np.ndarray
    optimizer_history: tuple[dict[str, Any], ...]
    evaluation: JointProblemEvaluation
    bag_rollouts: tuple[BagRollout, ...]
    elapsed_seconds: float


class JointMultipleShootingProblem:
    """Shared global parameters with disjoint bag-local shooting nodes."""

    def __init__(self, blocks: Sequence[BagShootingBlock]) -> None:
        self.blocks = tuple(blocks)
        if not self.blocks:
            raise ValueError("joint problem requires at least one bag block")
        first = self.blocks[0].problem
        self.global_dimension = int(first.global_dimension)
        self.prior_weight = float(first.prior_weight)
        self.prior_scales = np.asarray(first.prior_scales, dtype=float).copy()
        if self.prior_scales.shape != (strict.PHYSICAL_DIMENSION,):
            raise ValueError("bag block prior has the wrong dimension")
        weights = np.asarray(
            [block.normalized_weight for block in self.blocks], dtype=float
        )
        if not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("normalized bag weights must sum to one")
        self.node_slices: list[slice] = []
        self.continuity_slices: list[slice] = []
        variable_offset = self.global_dimension
        continuity_offset = 0
        for block in self.blocks:
            problem = block.problem
            if (
                problem.global_dimension != self.global_dimension
                or not np.isclose(problem.prior_weight, self.prior_weight)
                or not np.array_equal(problem.prior_scales, self.prior_scales)
            ):
                raise ValueError("bag blocks use incompatible global coordinates")
            node_end = variable_offset + block.node_variable_dimension
            self.node_slices.append(slice(variable_offset, node_end))
            variable_offset = node_end
            continuity_end = continuity_offset + problem.continuity_dimension
            self.continuity_slices.append(
                slice(continuity_offset, continuity_end)
            )
            continuity_offset = continuity_end
        self.node_slices = tuple(self.node_slices)
        self.continuity_slices = tuple(self.continuity_slices)
        self.variable_dimension = variable_offset
        self.pose_residual_dimension = sum(
            block.problem.pose_residual_dimension for block in self.blocks
        )
        self.data_residual_dimension = (
            self.pose_residual_dimension + strict.PHYSICAL_DIMENSION
        )
        self.continuity_dimension = continuity_offset

    def set_command_width_fraction(self, width_fraction: float) -> None:
        for block in self.blocks:
            setter = getattr(block.problem, "set_command_width_fraction", None)
            if setter is None:
                raise TypeError("strict-ZOH block has no smooth width")
            setter(width_fraction)

    def split_coordinate(
        self,
        coordinate: Sequence[float],
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        value = np.asarray(coordinate, dtype=float)
        if (
            value.shape != (self.variable_dimension,)
            or np.any(~np.isfinite(value))
        ):
            raise ValueError("joint coordinate has the wrong shape")
        nodes = tuple(value[node_slice] for node_slice in self.node_slices)
        return value[: self.global_dimension], nodes

    def local_coordinate(
        self,
        coordinate: Sequence[float],
        bag_index: int,
    ) -> np.ndarray:
        global_coordinate, nodes = self.split_coordinate(coordinate)
        return np.concatenate((global_coordinate, nodes[bag_index]))

    def initial_coordinate(self) -> np.ndarray:
        first = self.blocks[0].problem.initial_coordinate()
        result = np.empty(self.variable_dimension, dtype=float)
        result[: self.global_dimension] = first[: self.global_dimension]
        for index, block in enumerate(self.blocks):
            local = block.problem.initial_coordinate()
            if not np.array_equal(
                local[: self.global_dimension],
                result[: self.global_dimension],
            ):
                raise ValueError("bag blocks have different global initial values")
            result[self.node_slices[index]] = local[self.global_dimension :]
        return result

    def continuous_coordinate(
        self,
        global_coordinate: Sequence[float],
    ) -> np.ndarray:
        """Construct bag-local nodes by one sequential rollout per bag.

        With the global parameters fixed, the multiple-shooting continuity
        equations are block triangular: the end state of segment ``k`` is
        exactly the initial node of segment ``k + 1``.  Exploiting that
        structure is both exact and substantially cheaper than asking a
        generic least-squares solver to rediscover it.
        """

        global_value = np.asarray(global_coordinate, dtype=float)
        if (
            global_value.shape != (self.global_dimension,)
            or np.any(~np.isfinite(global_value))
        ):
            raise ValueError("joint global coordinate has the wrong shape")
        result = np.empty(self.variable_dimension, dtype=float)
        result[: self.global_dimension] = global_value
        for bag_index, block in enumerate(self.blocks):
            local_problem = block.problem
            local = local_problem.initial_coordinate()
            local[: self.global_dimension] = global_value
            decoded, parameter_jacobian = (
                local_problem._decode_global_coordinate(global_value)
            )
            if hasattr(local_problem, "width_fraction"):
                local_problem._active_delay = local_problem.coordinate_delay(
                    local
                )
            previous_node = None
            for segment_index in range(local_problem.node_count):
                segment = local_problem._evaluate_segment(
                    segment_index,
                    decoded,
                    parameter_jacobian,
                    previous_node,
                )
                previous_node = strict._encode_node(
                    local_problem.node_references[segment_index],
                    segment.end_rigid,
                    segment.end_actuator,
                )
                start = (
                    self.global_dimension
                    + segment_index * strict.NODE_DIMENSION
                )
                local[start : start + strict.NODE_DIMENSION] = previous_node
            result[self.node_slices[bag_index]] = local[
                self.global_dimension :
            ]
        return result

    def bounds(
        self,
        global_lower: np.ndarray,
        global_upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = np.empty(self.variable_dimension, dtype=float)
        upper = np.empty(self.variable_dimension, dtype=float)
        lower[: self.global_dimension] = global_lower
        upper[: self.global_dimension] = global_upper
        for index, block in enumerate(self.blocks):
            local_lower, local_upper = block.problem.bounds(
                global_lower,
                global_upper,
            )
            lower[self.node_slices[index]] = local_lower[self.global_dimension :]
            upper[self.node_slices[index]] = local_upper[self.global_dimension :]
        return lower, upper

    def coordinate_delay(self, coordinate: Sequence[float]) -> float:
        global_coordinate, _nodes = self.split_coordinate(coordinate)
        if self.global_dimension == GLOBAL_DIMENSION:
            return float(global_coordinate[DELAY_INDEX])
        delays = np.asarray(
            [block.problem.delay for block in self.blocks], dtype=float
        )
        if not np.allclose(delays, delays[0], rtol=0.0, atol=1.0e-12):
            raise ValueError("strict bag blocks do not share one fixed delay")
        return float(delays[0])

    def evaluate(self, coordinate: Sequence[float]) -> JointProblemEvaluation:
        global_coordinate, nodes = self.split_coordinate(coordinate)
        data_residual = np.empty(self.data_residual_dimension, dtype=float)
        data_jacobian = np.zeros(
            (self.data_residual_dimension, self.variable_dimension),
            dtype=float,
        )
        continuity_residual = np.empty(
            self.continuity_dimension, dtype=float
        )
        continuity_jacobian = np.zeros(
            (self.continuity_dimension, self.variable_dimension),
            dtype=float,
        )
        bag_evaluations: list[strict.ProblemEvaluation] = []
        pose_offset = 0
        for index, block in enumerate(self.blocks):
            problem = block.problem
            local_coordinate = np.concatenate((global_coordinate, nodes[index]))
            evaluation = problem.evaluate(local_coordinate)
            bag_evaluations.append(evaluation)
            pose_end = pose_offset + problem.pose_residual_dimension
            root_weight = math.sqrt(block.normalized_weight)
            local_pose = evaluation.data_residual[
                : problem.pose_residual_dimension
            ]
            local_pose_jacobian = evaluation.data_jacobian[
                : problem.pose_residual_dimension
            ]
            data_residual[pose_offset:pose_end] = root_weight * local_pose
            data_jacobian[
                pose_offset:pose_end, : self.global_dimension
            ] = root_weight * local_pose_jacobian[:, : self.global_dimension]
            data_jacobian[
                pose_offset:pose_end, self.node_slices[index]
            ] = root_weight * local_pose_jacobian[:, self.global_dimension :]
            pose_offset = pose_end

            continuity_slice = self.continuity_slices[index]
            continuity_residual[continuity_slice] = (
                evaluation.continuity_residual
            )
            continuity_jacobian[
                continuity_slice, : self.global_dimension
            ] = evaluation.continuity_jacobian[:, : self.global_dimension]
            continuity_jacobian[
                continuity_slice, self.node_slices[index]
            ] = evaluation.continuity_jacobian[:, self.global_dimension :]

        prior_residual = (
            math.sqrt(self.prior_weight)
            * global_coordinate[: strict.PHYSICAL_DIMENSION]
            / self.prior_scales
        )
        data_residual[pose_offset:] = prior_residual
        data_jacobian[
            pose_offset:,
            : strict.PHYSICAL_DIMENSION,
        ] = np.diag(math.sqrt(self.prior_weight) / self.prior_scales)
        if (
            np.any(~np.isfinite(data_residual))
            or np.any(~np.isfinite(data_jacobian))
            or np.any(~np.isfinite(continuity_residual))
            or np.any(~np.isfinite(continuity_jacobian))
        ):
            raise FloatingPointError("joint multiple-shooting evaluation is non-finite")
        return JointProblemEvaluation(
            data_residual=data_residual,
            data_jacobian=data_jacobian,
            continuity_residual=continuity_residual,
            continuity_jacobian=continuity_jacobian,
            bag_evaluations=tuple(bag_evaluations),
            decoded=bag_evaluations[0].decoded,
        )

    def full_rollouts(
        self,
        global_coordinate: Sequence[float],
    ) -> tuple[BagRollout, ...]:
        value = np.asarray(global_coordinate, dtype=float)
        if value.shape != (self.global_dimension,):
            raise ValueError("joint global coordinate has the wrong shape")
        rollouts = []
        for block in self.blocks:
            position, orientation, residual = block.problem.full_rollout(value)
            rollouts.append(BagRollout(position, orientation, residual))
        return tuple(rollouts)

    def weighted_full_rollout_loss(
        self,
        global_coordinate: Sequence[float],
    ) -> float:
        return float(
            sum(
                block.normalized_weight * rollout.loss
                for block, rollout in zip(
                    self.blocks,
                    self.full_rollouts(global_coordinate),
                )
            )
        )


def _continuity_max(solution: JointSolution) -> float:
    residual = solution.evaluation.continuity_residual
    return 0.0 if residual.size == 0 else float(np.max(np.abs(residual)))


def _joint_full_loss(
    problem: JointMultipleShootingProblem,
    solution: JointSolution,
) -> float:
    return float(
        sum(
            block.normalized_weight * rollout.loss
            for block, rollout in zip(problem.blocks, solution.bag_rollouts)
        )
    )


def _restore_joint_continuity(
    problem: JointMultipleShootingProblem,
    coordinate: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    arguments: argparse.Namespace,
) -> tuple[np.ndarray, JointProblemEvaluation, dict[str, Any]]:
    """Project bag-local nodes onto continuity with shared globals fixed."""

    global_coordinate, _nodes = problem.split_coordinate(coordinate)
    exact = problem.continuous_coordinate(global_coordinate)
    bound_tolerance = 1.0e-10
    if np.all(exact >= bounds[0] - bound_tolerance) and np.all(
        exact <= bounds[1] + bound_tolerance
    ):
        exact = np.clip(exact, bounds[0], bounds[1])
        evaluation = problem.evaluate(exact)
        continuity = evaluation.continuity_residual
        continuity_max = (
            0.0
            if continuity.size == 0
            else float(np.max(np.abs(continuity)))
        )
        per_bag = {
            block.specification.bag_id: (
                0.0
                if block.problem.continuity_dimension == 0
                else float(
                    np.max(
                        np.abs(
                            continuity[problem.continuity_slices[index]]
                        )
                    )
                )
            )
            for index, block in enumerate(problem.blocks)
        }
        diagnostic = {
            "phase": "continuity_restoration",
            "method": "sequential_rollout",
            "success": bool(
                continuity_max <= arguments.continuity_tolerance
            ),
            "status": 1,
            "message": "bag-local nodes reconstructed by sequential rollout",
            "cost": 0.5 * float(continuity @ continuity),
            "optimality": 0.0,
            "nfev": 1,
            "njev": 0,
            "continuity_l2_normalized": float(np.linalg.norm(continuity)),
            "continuity_max_normalized": continuity_max,
            "per_bag_continuity_max_normalized": per_bag,
            "delay_seconds": problem.coordinate_delay(exact),
        }
        print(
            "  sequential restoration continuity L2={:.3e}, max={:.3e}".format(
                diagnostic["continuity_l2_normalized"],
                continuity_max,
            ),
            flush=True,
        )
        return exact, evaluation, diagnostic

    print(
        "  exact continuity projection exceeds configured node bounds; "
        "falling back to bounded least squares",
        flush=True,
    )
    node_initial = np.asarray(
        coordinate[problem.global_dimension :], dtype=float
    ).copy()
    if node_initial.size == 0:
        evaluation = problem.evaluate(coordinate)
        return coordinate.copy(), evaluation, {
            "phase": "continuity_restoration",
            "nfev": 0,
            "success": True,
            "continuity_max_normalized": 0.0,
        }
    cached_nodes: Optional[np.ndarray] = None
    cached_evaluation: Optional[JointProblemEvaluation] = None

    def evaluate_nodes(nodes: Sequence[float]) -> JointProblemEvaluation:
        nonlocal cached_nodes, cached_evaluation
        value = np.asarray(nodes, dtype=float)
        if cached_nodes is None or not np.array_equal(value, cached_nodes):
            full_coordinate = np.concatenate((global_coordinate, value))
            cached_evaluation = problem.evaluate(full_coordinate)
            cached_nodes = value.copy()
        if cached_evaluation is None:
            raise RuntimeError("continuity restoration cache is empty")
        return cached_evaluation

    result = least_squares(
        lambda nodes: evaluate_nodes(nodes).continuity_residual,
        node_initial,
        jac=lambda nodes: evaluate_nodes(nodes).continuity_jacobian[
            :, problem.global_dimension :
        ],
        bounds=(
            bounds[0][problem.global_dimension :],
            bounds[1][problem.global_dimension :],
        ),
        method="trf",
        x_scale="jac",
        loss="linear",
        ftol=arguments.ftol,
        xtol=arguments.xtol,
        gtol=arguments.gtol,
        max_nfev=arguments.continuity_restoration_max_nfev,
        verbose=1,
    )
    restored = np.concatenate((global_coordinate, result.x))
    evaluation = problem.evaluate(restored)
    continuity = evaluation.continuity_residual
    continuity_max = (
        0.0 if continuity.size == 0 else float(np.max(np.abs(continuity)))
    )
    per_bag = {
        block.specification.bag_id: (
            0.0
            if block.problem.continuity_dimension == 0
            else float(
                np.max(
                    np.abs(continuity[problem.continuity_slices[index]])
                )
            )
        )
        for index, block in enumerate(problem.blocks)
    }
    diagnostic = {
        "phase": "continuity_restoration",
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "njev": None if result.njev is None else int(result.njev),
        "continuity_l2_normalized": float(np.linalg.norm(continuity)),
        "continuity_max_normalized": continuity_max,
        "per_bag_continuity_max_normalized": per_bag,
        "delay_seconds": problem.coordinate_delay(restored),
    }
    print(
        "  restoration continuity L2={:.3e}, max={:.3e}".format(
            diagnostic["continuity_l2_normalized"],
            continuity_max,
        ),
        flush=True,
    )
    return restored, evaluation, diagnostic


def _solve_joint(
    problem: JointMultipleShootingProblem,
    initial_coordinate: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    arguments: argparse.Namespace,
) -> JointSolution:
    started = time.perf_counter()
    coordinate = np.clip(initial_coordinate, bounds[0], bounds[1])
    multipliers = np.zeros(problem.continuity_dimension, dtype=float)
    penalty = float(arguments.continuity_penalty_initial)
    previous_norm = float("inf")
    history: list[dict[str, Any]] = []
    final_evaluation: Optional[JointProblemEvaluation] = None
    for outer_iteration in range(arguments.augmented_lagrangian_iterations):
        print(
            "shared delay {:.6f}s, augmented iteration {}/{}, penalty={:.3g}".format(
                problem.coordinate_delay(coordinate),
                outer_iteration + 1,
                arguments.augmented_lagrangian_iterations,
                penalty,
            ),
            flush=True,
        )
        objective = strict._CachedAugmentedObjective(
            problem,
            multipliers,
            penalty,
        )
        result = least_squares(
            objective.residual,
            coordinate,
            jac=objective.jacobian,
            bounds=bounds,
            method="trf",
            x_scale="jac",
            loss="linear",
            ftol=arguments.ftol,
            xtol=arguments.xtol,
            gtol=arguments.gtol,
            max_nfev=arguments.max_nfev,
            verbose=1,
        )
        coordinate = result.x.copy()
        final_evaluation = problem.evaluate(coordinate)
        continuity = final_evaluation.continuity_residual
        continuity_norm = float(np.linalg.norm(continuity))
        continuity_max = (
            0.0 if continuity.size == 0 else float(np.max(np.abs(continuity)))
        )
        data_cost = 0.5 * float(
            final_evaluation.data_residual @ final_evaluation.data_residual
        )
        per_bag_continuity = {
            block.specification.bag_id: (
                0.0
                if block.problem.continuity_dimension == 0
                else float(
                    np.max(
                        np.abs(
                            continuity[problem.continuity_slices[index]]
                        )
                    )
                )
            )
            for index, block in enumerate(problem.blocks)
        }
        history.append(
            {
                "outer_iteration": outer_iteration + 1,
                "penalty": penalty,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "cost": float(result.cost),
                "data_cost": data_cost,
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "njev": None if result.njev is None else int(result.njev),
                "continuity_l2_normalized": continuity_norm,
                "continuity_max_normalized": continuity_max,
                "per_bag_continuity_max_normalized": per_bag_continuity,
                "delay_seconds": problem.coordinate_delay(coordinate),
            }
        )
        print(
            "  joint data={:.9g}, continuity L2={:.3e}, max={:.3e}".format(
                data_cost,
                continuity_norm,
                continuity_max,
            ),
            flush=True,
        )
        if continuity_max <= arguments.continuity_tolerance:
            break
        multipliers = multipliers + penalty * continuity
        if continuity_norm > arguments.penalty_reduction_target * previous_norm:
            penalty = min(
                arguments.continuity_penalty_max,
                penalty * arguments.continuity_penalty_growth,
            )
        previous_norm = continuity_norm
    if final_evaluation is None:
        raise RuntimeError("joint solver did not evaluate a solution")
    final_continuity_max = (
        0.0
        if final_evaluation.continuity_residual.size == 0
        else float(
            np.max(np.abs(final_evaluation.continuity_residual))
        )
    )
    if final_continuity_max > arguments.continuity_tolerance:
        print("restoring bag-local continuity with shared globals fixed", flush=True)
        coordinate, final_evaluation, restoration = _restore_joint_continuity(
            problem,
            coordinate,
            bounds,
            arguments,
        )
        history.append(restoration)
    global_coordinate, _nodes = problem.split_coordinate(coordinate)
    return JointSolution(
        delay=problem.coordinate_delay(coordinate),
        coordinate=coordinate,
        optimizer_history=tuple(history),
        evaluation=final_evaluation,
        bag_rollouts=problem.full_rollouts(global_coordinate),
        elapsed_seconds=time.perf_counter() - started,
    )


def _normalized_weights(
    specifications: Sequence[BagSpecification],
) -> np.ndarray:
    raw = np.asarray([item.weight for item in specifications], dtype=float)
    return raw / float(np.sum(raw))


def _make_blocks(
    specifications: Sequence[BagSpecification],
    problems: Sequence[strict.MultipleShootingProblem],
) -> tuple[BagShootingBlock, ...]:
    if len(specifications) != len(problems):
        raise ValueError("bag specifications and problems differ in length")
    weights = _normalized_weights(specifications)
    return tuple(
        BagShootingBlock(specification, float(weight), problem)
        for specification, weight, problem in zip(
            specifications,
            weights,
            problems,
        )
    )


def _smooth_joint_problem(
    specifications: Sequence[BagSpecification],
    flights: Sequence[Any],
    initial_delay: float,
    width_fraction: float,
    arguments: argparse.Namespace,
) -> JointMultipleShootingProblem:
    problems = [
        smooth.SmoothLagMultipleShootingProblem(
            flight=flight,
            sample_step=arguments.sample_step,
            integration_step=arguments.integration_step,
            initial_delay=initial_delay,
            width_fraction=width_fraction,
            segment_duration=arguments.segment_duration,
            body_displacement_scale=arguments.body_displacement_scale,
            prior_weight=arguments.prior_weight,
            node_position_bound=arguments.node_position_bound,
            node_orientation_bound=arguments.node_orientation_bound,
            node_velocity_bound=arguments.node_velocity_bound,
            node_angular_velocity_bound=(
                arguments.node_angular_velocity_bound
            ),
        )
        for flight in flights
    ]
    return JointMultipleShootingProblem(_make_blocks(specifications, problems))


def _strict_joint_problem(
    specifications: Sequence[BagSpecification],
    flights: Sequence[Any],
    delay: float,
    arguments: argparse.Namespace,
) -> JointMultipleShootingProblem:
    problems = [
        smooth._strict_problem(flight, delay, arguments) for flight in flights
    ]
    return JointMultipleShootingProblem(_make_blocks(specifications, problems))


def _strict_initial_from_smooth_joint(
    target: JointMultipleShootingProblem,
    source: JointMultipleShootingProblem,
    source_coordinate: Sequence[float],
) -> np.ndarray:
    if len(target.blocks) != len(source.blocks):
        raise ValueError("smooth and strict joint problems differ in bag count")
    source_global, _source_nodes = source.split_coordinate(source_coordinate)
    result = target.initial_coordinate()
    result[: strict.PHYSICAL_DIMENSION] = source_global[
        : strict.PHYSICAL_DIMENSION
    ]
    for index, (target_block, source_block) in enumerate(
        zip(target.blocks, source.blocks)
    ):
        if (
            target_block.specification.bag_id
            != source_block.specification.bag_id
        ):
            raise ValueError("smooth and strict bag ordering differs")
        source_local = source.local_coordinate(source_coordinate, index)
        target_local = smooth._strict_initial_from_smooth(
            target_block.problem,
            source_block.problem,
            source_local,
        )
        result[target.node_slices[index]] = target_local[
            strict.PHYSICAL_DIMENSION :
        ]
    return result


def _bag_continuity_max(
    problem: JointMultipleShootingProblem,
    solution: JointSolution,
    bag_index: int,
) -> float:
    residual = solution.evaluation.continuity_residual[
        problem.continuity_slices[bag_index]
    ]
    return 0.0 if residual.size == 0 else float(np.max(np.abs(residual)))


def _bag_stitched_loss(
    problem: JointMultipleShootingProblem,
    solution: JointSolution,
    bag_index: int,
) -> float:
    bag_problem = problem.blocks[bag_index].problem
    residual = solution.evaluation.bag_evaluations[
        bag_index
    ].data_residual[: bag_problem.pose_residual_dimension]
    return 0.5 * float(residual @ residual)


def _bag_solution_view(
    problem: JointMultipleShootingProblem,
    solution: JointSolution,
    bag_index: int,
) -> strict.FixedDelaySolution:
    rollout = solution.bag_rollouts[bag_index]
    return strict.FixedDelaySolution(
        delay=solution.delay,
        coordinate=problem.local_coordinate(solution.coordinate, bag_index),
        optimizer_history=solution.optimizer_history,
        evaluation=solution.evaluation.bag_evaluations[bag_index],
        full_rollout_position=rollout.sensor_position,
        full_rollout_orientation_xyzw=rollout.sensor_orientation_xyzw,
        full_rollout_residual=rollout.residual,
        elapsed_seconds=solution.elapsed_seconds,
    )


def _soft_prior_cost(
    problem: JointMultipleShootingProblem,
    coordinate: Sequence[float],
) -> float:
    global_coordinate, _nodes = problem.split_coordinate(coordinate)
    normalized = (
        global_coordinate[: strict.PHYSICAL_DIMENSION] / problem.prior_scales
    )
    return 0.5 * problem.prior_weight * float(normalized @ normalized)


def _bag_diagnostic_payload(
    problem: JointMultipleShootingProblem,
    solution: JointSolution,
    bag_index: int,
    continuity_tolerance: float,
) -> dict[str, Any]:
    block = problem.blocks[bag_index]
    bag_solution = _bag_solution_view(problem, solution, bag_index)
    payload = strict._solution_payload(
        block.problem,
        bag_solution,
        continuity_tolerance,
    )
    stitched_loss = _bag_stitched_loss(problem, solution, bag_index)
    full_loss = solution.bag_rollouts[bag_index].loss
    payload.update(
        {
            "id": block.specification.bag_id,
            "raw_weight": block.specification.weight,
            "normalized_weight": block.normalized_weight,
            "weighted_stitched_loss_contribution_m2": (
                block.normalized_weight * stitched_loss
            ),
            "weighted_full_rollout_loss_contribution_m2": (
                block.normalized_weight * full_loss
            ),
            "stitched_full_rollout_loss_difference_m2": (
                stitched_loss - full_loss
            ),
        }
    )
    return payload


def _joint_solution_payload(
    problem: JointMultipleShootingProblem,
    solution: JointSolution,
    continuity_tolerance: float,
) -> dict[str, Any]:
    global_coordinate, _nodes = problem.split_coordinate(solution.coordinate)
    bag_payloads = [
        _bag_diagnostic_payload(
            problem,
            solution,
            index,
            continuity_tolerance,
        )
        for index in range(len(problem.blocks))
    ]
    joint_stitched = float(
        sum(
            block.normalized_weight
            * _bag_stitched_loss(problem, solution, index)
            for index, block in enumerate(problem.blocks)
        )
    )
    return {
        "delay_seconds": solution.delay,
        "physical_coordinate": global_coordinate[: strict.PHYSICAL_DIMENSION],
        "parameters": strict._physical_payload(solution.evaluation.decoded),
        "joint_stitched_inertia_radius_loss_m2": joint_stitched,
        "joint_full_rollout_inertia_radius_loss_m2": _joint_full_loss(
            problem,
            solution,
        ),
        "joint_stitched_full_rollout_loss_difference_m2": (
            joint_stitched - _joint_full_loss(problem, solution)
        ),
        "soft_prior_cost": _soft_prior_cost(problem, solution.coordinate),
        "continuity_max_normalized": _continuity_max(solution),
        "continuity_l2_normalized": float(
            np.linalg.norm(solution.evaluation.continuity_residual)
        ),
        "continuity_converged": bool(
            _continuity_max(solution) <= continuity_tolerance
        ),
        "all_bags_continuity_converged": all(
            payload["continuity_converged"] for payload in bag_payloads
        ),
        "optimizer_history": list(solution.optimizer_history),
        "elapsed_seconds": solution.elapsed_seconds,
        "bags": bag_payloads,
    }


def _smooth_stage_payload(
    problem: JointMultipleShootingProblem,
    solution: JointSolution,
    width_fraction: float,
) -> dict[str, Any]:
    global_coordinate, _nodes = problem.split_coordinate(solution.coordinate)
    bag_results = []
    for index, block in enumerate(problem.blocks):
        stitched_loss = _bag_stitched_loss(problem, solution, index)
        full_loss = solution.bag_rollouts[index].loss
        bag_results.append(
            {
                "id": block.specification.bag_id,
                "normalized_weight": block.normalized_weight,
                "stitched_inertia_radius_loss_m2": stitched_loss,
                "full_rollout_inertia_radius_loss_m2": full_loss,
                "weighted_stitched_loss_contribution_m2": (
                    block.normalized_weight * stitched_loss
                ),
                "weighted_full_rollout_loss_contribution_m2": (
                    block.normalized_weight * full_loss
                ),
                "continuity_max_normalized": _bag_continuity_max(
                    problem,
                    solution,
                    index,
                ),
            }
        )
    pose_rows = problem.pose_residual_dimension
    delay_gradient = float(
        solution.evaluation.data_jacobian[:pose_rows, DELAY_INDEX]
        @ solution.evaluation.data_residual[:pose_rows]
    )
    return {
        "width_fraction": width_fraction,
        "delay_seconds": float(global_coordinate[DELAY_INDEX]),
        "joint_stitched_inertia_radius_loss_m2": float(
            sum(item["weighted_stitched_loss_contribution_m2"] for item in bag_results)
        ),
        "joint_full_rollout_inertia_radius_loss_m2": _joint_full_loss(
            problem,
            solution,
        ),
        "soft_prior_cost": _soft_prior_cost(problem, solution.coordinate),
        "continuity_max_normalized": _continuity_max(solution),
        "lag_data_gradient": delay_gradient,
        "parameters": strict._physical_payload(solution.evaluation.decoded),
        "bags": bag_results,
        "optimizer_history": list(solution.optimizer_history),
        "elapsed_seconds": solution.elapsed_seconds,
    }


def _joint_parameter_summary_lines(
    problem: JointMultipleShootingProblem,
    solution: JointSolution,
    continuity_tolerance: float,
) -> list[str]:
    parameters = solution.evaluation.decoded.parameters
    inertia = parameters.inertia
    principal = np.linalg.eigvalsh(inertia)
    global_coordinate, _nodes = problem.split_coordinate(solution.coordinate)
    payload = _joint_solution_payload(
        problem,
        solution,
        continuity_tolerance,
    )
    lines = [
        "Selected joint multiple-bag estimate",
        "",
        "Shared decoded physical parameters",
        "  mass [kg]                  {:.12g}".format(parameters.mass),
        "  inertia [kg m^2]",
    ]
    lines.extend(
        "    [{: .12g}  {: .12g}  {: .12g}]".format(*row) for row in inertia
    )
    lines.extend(
        (
            "  principal inertia [kg m^2] [{:.12g}, {:.12g}, {:.12g}]".format(
                *principal
            ),
            "  CoG offset [m]             [{:.12g}, {:.12g}, {:.12g}]".format(
                *parameters.cog_offset
            ),
            "  rotor force effectiveness  [{:.12g}, {:.12g}, {:.12g}, {:.12g}]".format(
                *parameters.force_effectiveness
            ),
            "  shared command delay [s]    {:.12g}".format(solution.delay),
            "",
            "Thirteen shared optimized smooth coordinates",
        )
    )
    lines.extend(
        "  {:<43s} {: .12g}".format(name, value)
        for name, value in zip(
            strict.PHYSICAL_PARAMETER_NAMES,
            global_coordinate[: strict.PHYSICAL_DIMENSION],
        )
    )
    lines.extend(
        (
            "",
            "Joint fit diagnostics",
            "  joint stitched loss [m^2]     {:.12g}".format(
                payload["joint_stitched_inertia_radius_loss_m2"]
            ),
            "  joint full-rollout loss [m^2] {:.12g}".format(
                payload["joint_full_rollout_inertia_radius_loss_m2"]
            ),
            "  broad soft-prior cost          {:.12g}".format(
                payload["soft_prior_cost"]
            ),
            "  continuity max (normalized)    {:.12g}".format(
                payload["continuity_max_normalized"]
            ),
            "  continuity tolerance           {:.12g}".format(
                continuity_tolerance
            ),
            "",
            "Per-bag diagnostics",
        )
    )
    for bag in payload["bags"]:
        lines.extend(
            (
                "  {}".format(bag["id"]),
                "    normalized weight        {:.12g}".format(
                    bag["normalized_weight"]
                ),
                "    stitched loss [m^2]       {:.12g}".format(
                    bag["stitched_inertia_radius_loss_m2"]
                ),
                "    full-rollout loss [m^2]   {:.12g}".format(
                    bag["full_rollout_inertia_radius_loss_m2"]
                ),
                "    continuity max            {:.12g}".format(
                    bag["continuity_max_normalized"]
                ),
            )
        )
    return lines


def _write_delay_profile_pdf(
    path: Path,
    smooth_delay: float,
    candidate_delays: np.ndarray,
    unrefined_losses: np.ndarray,
    refined: Sequence[tuple[JointMultipleShootingProblem, JointSolution]],
    selected_delay: float,
) -> None:
    figure, axis = plt.subplots(figsize=(11.7, 8.3), constrained_layout=True)
    axis.plot(
        candidate_delays * 1000.0,
        unrefined_losses,
        marker="o",
        label="strict ZOH, smooth shared physical parameters",
    )
    axis.scatter(
        [solution.delay * 1000.0 for _problem, solution in refined],
        [_joint_full_loss(problem, solution) for problem, solution in refined],
        marker="s",
        s=70,
        label="strict ZOH, joint refined",
    )
    axis.axvline(
        smooth_delay * 1000.0,
        color="#9467bd",
        linestyle="--",
        label="smoothstep joint estimate",
    )
    selected_problem, selected_solution = min(
        refined,
        key=lambda item: abs(item[1].delay - selected_delay),
    )
    axis.scatter(
        [selected_delay * 1000.0],
        [_joint_full_loss(selected_problem, selected_solution)],
        marker="*",
        s=220,
        color="#1e965f",
        label="selected strict ZOH",
        zorder=5,
    )
    axis.set_xlabel("shared recorded-command delay [ms]")
    axis.set_ylabel("weighted joint full-rollout loss [m²]")
    axis.set_title("Joint smooth lag search and strict-ZOH local polish")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    with PdfPages(path) as pdf:
        pdf.savefig(figure)
    plt.close(figure)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly estimate shared vehicle parameters and command lag from "
            "multiple recorded bags with bag-local shooting nodes."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--integration-step", type=float, default=0.025)
    parser.add_argument("--segment-duration", type=float, default=0.5)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--smooth-max-nfev", type=int, default=60)
    parser.add_argument(
        "--continuity-restoration-max-nfev",
        type=int,
        default=120,
    )
    parser.add_argument("--augmented-lagrangian-iterations", type=int, default=10)
    parser.add_argument("--continuity-penalty-initial", type=float, default=1.0)
    parser.add_argument("--continuity-penalty-growth", type=float, default=10.0)
    parser.add_argument("--continuity-penalty-max", type=float, default=1.0e6)
    parser.add_argument("--penalty-reduction-target", type=float, default=0.50)
    parser.add_argument("--continuity-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--ftol", type=float, default=1.0e-6)
    parser.add_argument("--xtol", type=float, default=1.0e-6)
    parser.add_argument("--gtol", type=float, default=1.0e-6)
    parser.add_argument("--delay-bounds", type=float, nargs=2, default=(0.0, 0.20))
    parser.add_argument("--initial-delay", type=float, default=None)
    parser.add_argument(
        "--smoothstep-width-fractions",
        type=float,
        nargs="+",
        default=(0.50, 0.20, 0.05),
    )
    parser.add_argument("--zoh-polish-radius", type=float, default=0.004)
    parser.add_argument("--zoh-polish-step", type=float, default=0.001)
    parser.add_argument("--zoh-polish-top-k", type=int, default=3)
    parser.add_argument("--body-displacement-scale", type=float, default=1.0)
    parser.add_argument("--node-position-bound", type=float, default=np.inf)
    parser.add_argument("--node-orientation-bound", type=float, default=np.inf)
    parser.add_argument("--node-velocity-bound", type=float, default=np.inf)
    parser.add_argument(
        "--node-angular-velocity-bound",
        type=float,
        default=np.inf,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    return parser


def _validate_arguments(
    arguments: argparse.Namespace,
    config: MultiBagConfig,
) -> float:
    positive = (
        arguments.sample_step,
        arguments.integration_step,
        arguments.segment_duration,
        arguments.max_nfev,
        arguments.smooth_max_nfev,
        arguments.continuity_restoration_max_nfev,
        arguments.augmented_lagrangian_iterations,
        arguments.continuity_penalty_initial,
        arguments.continuity_penalty_growth,
        arguments.continuity_penalty_max,
        arguments.continuity_tolerance,
        arguments.ftol,
        arguments.xtol,
        arguments.gtol,
        arguments.zoh_polish_radius,
        arguments.zoh_polish_step,
        arguments.zoh_polish_top_k,
        arguments.body_displacement_scale,
    )
    bounds = np.asarray(arguments.delay_bounds, dtype=float)
    widths = np.asarray(arguments.smoothstep_width_fractions, dtype=float)
    initial_delay = (
        config.initial_delay_seconds
        if arguments.initial_delay is None
        else float(arguments.initial_delay)
    )
    node_bounds = (
        arguments.node_position_bound,
        arguments.node_orientation_bound,
        arguments.node_velocity_bound,
        arguments.node_angular_velocity_bound,
    )
    if (
        any(not np.isfinite(value) or value <= 0.0 for value in positive)
        or any(np.isnan(value) or value <= 0.0 for value in node_bounds)
        or not np.isfinite(arguments.prior_weight)
        or arguments.prior_weight < 0.0
        or arguments.continuity_penalty_growth <= 1.0
        or not 0.0 < arguments.penalty_reduction_target < 1.0
        or bounds.shape != (2,)
        or np.any(~np.isfinite(bounds))
        or bounds[0] < 0.0
        or bounds[1] <= bounds[0]
        or not np.isfinite(initial_delay)
        or not bounds[0] <= initial_delay <= bounds[1]
        or widths.ndim != 1
        or widths.size < 1
        or np.any(~np.isfinite(widths))
        or np.any(widths <= 0.0)
    ):
        raise SystemExit("multi-bag multiple-shooting settings are invalid")
    return initial_delay


def run(arguments: argparse.Namespace) -> int:
    try:
        config = load_multi_bag_config(arguments.config)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    initial_delay = _validate_arguments(arguments, config)
    started = time.perf_counter()
    flights = []
    for index, specification in enumerate(config.bags):
        print(
            "loading bag {}/{} {}: {} [{:.3f}, {:.3f}] s".format(
                index + 1,
                len(config.bags),
                specification.bag_id,
                specification.path,
                specification.start,
                specification.end,
            ),
            flush=True,
        )
        flights.append(
            load_flight_data(
                str(specification.path),
                start_local=specification.start,
                end_local=specification.end,
                include_fc_specific_force=True,
                compute_sha256=False,
            )
        )
    widths = tuple(float(value) for value in arguments.smoothstep_width_fractions)
    smooth_problem = _smooth_joint_problem(
        config.bags,
        flights,
        initial_delay,
        widths[0],
        arguments,
    )
    global_lower, global_upper = smooth._global_bounds(arguments.delay_bounds)
    bounds = smooth_problem.bounds(global_lower, global_upper)
    coordinate = smooth_problem.initial_coordinate()
    smooth_arguments = argparse.Namespace(**vars(arguments))
    smooth_arguments.max_nfev = arguments.smooth_max_nfev
    stage_solutions: list[JointSolution] = []
    stage_payloads: list[dict[str, Any]] = []
    for stage_index, width in enumerate(widths):
        smooth_problem.set_command_width_fraction(width)
        print(
            "joint smoothstep stage {}/{}: width_fraction={:.6g}".format(
                stage_index + 1,
                len(widths),
                width,
            ),
            flush=True,
        )
        solution = _solve_joint(
            smooth_problem,
            coordinate,
            bounds,
            smooth_arguments,
        )
        coordinate = solution.coordinate.copy()
        stage_solutions.append(solution)
        stage_payloads.append(
            _smooth_stage_payload(smooth_problem, solution, width)
        )

    final_smooth = stage_solutions[-1]
    smooth_global, _smooth_nodes = smooth_problem.split_coordinate(
        final_smooth.coordinate
    )
    smooth_delay = float(smooth_global[DELAY_INDEX])
    candidate_delays = smooth.zoh_polish_delays(
        smooth_delay,
        arguments.zoh_polish_radius,
        arguments.zoh_polish_step,
        arguments.delay_bounds,
    )
    strict_problems: dict[float, JointMultipleShootingProblem] = {}
    unrefined_losses = np.empty(candidate_delays.size, dtype=float)
    for index, delay in enumerate(candidate_delays):
        problem = _strict_joint_problem(
            config.bags,
            flights,
            float(delay),
            arguments,
        )
        strict_problems[round(float(delay), 12)] = problem
        unrefined_losses[index] = problem.weighted_full_rollout_loss(
            smooth_global[: strict.PHYSICAL_DIMENSION]
        )
    top_count = min(arguments.zoh_polish_top_k, candidate_delays.size)
    top_indices = np.argsort(unrefined_losses, kind="stable")[:top_count]
    refined: list[tuple[JointMultipleShootingProblem, JointSolution]] = []
    strict_global_lower = np.full(strict.PHYSICAL_DIMENSION, -np.inf)
    strict_global_upper = np.full(strict.PHYSICAL_DIMENSION, np.inf)
    for rank, candidate_index in enumerate(top_indices):
        delay = float(candidate_delays[candidate_index])
        problem = strict_problems[round(delay, 12)]
        initial = _strict_initial_from_smooth_joint(
            problem,
            smooth_problem,
            final_smooth.coordinate,
        )
        print(
            "joint strict ZOH polish {}/{}: delay={:.6f}s, "
            "screening_loss={:.9g}".format(
                rank + 1,
                top_count,
                delay,
                unrefined_losses[candidate_index],
            ),
            flush=True,
        )
        solution = _solve_joint(
            problem,
            initial,
            problem.bounds(strict_global_lower, strict_global_upper),
            arguments,
        )
        refined.append((problem, solution))
    converged = [
        item
        for item in refined
        if _continuity_max(item[1]) <= arguments.continuity_tolerance
    ]
    selected_problem, selected_solution = min(
        converged if converged else refined,
        key=lambda item: _joint_full_loss(item[0], item[1]),
    )

    output_directory = (
        arguments.output_dir.expanduser().resolve() / OUTPUT_SUBDIRECTORY
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    bags_directory = output_directory / "bags"
    bags_directory.mkdir(parents=True, exist_ok=True)
    result_path = output_directory / "result.json"
    parameters_path = output_directory / "parameters.txt"
    delay_profile_path = output_directory / "delay_profile.pdf"
    selected_payload = _joint_solution_payload(
        selected_problem,
        selected_solution,
        arguments.continuity_tolerance,
    )
    refined_payloads = []
    for problem, solution in refined:
        payload = _joint_solution_payload(
            problem,
            solution,
            arguments.continuity_tolerance,
        )
        payload["screening_loss_m2"] = float(
            unrefined_losses[
                int(np.argmin(np.abs(candidate_delays - solution.delay)))
            ]
        )
        refined_payloads.append(payload)

    bag_sources = []
    bag_outputs = {}
    for index, block in enumerate(selected_problem.blocks):
        specification = block.specification
        bag_directory = bags_directory / specification.bag_id
        bag_directory.mkdir(parents=True, exist_ok=True)
        bag_result_path = bag_directory / "result.json"
        trajectory_path = bag_directory / "trajectory.pdf"
        bag_solution = _bag_solution_view(
            selected_problem,
            selected_solution,
            index,
        )
        bag_payload = selected_payload["bags"][index]
        stitched_metrics = bag_payload["stitched_recorded_control_metrics"]
        parameter_lines = strict._parameter_summary_lines(
            block.problem,
            bag_solution,
            stitched_metrics,
            arguments.continuity_tolerance,
        )
        strict._write_pdf(
            trajectory_path,
            block.problem,
            bag_solution,
            stitched_metrics,
            parameter_lines,
            arguments.continuity_tolerance,
        )
        bag_result = {
            "schema": SCHEMA + "/bag-result",
            "id": specification.bag_id,
            "source": {
                "path": str(specification.path),
                "sha256": baseline._sha256(specification.path),
                "requested_interval_seconds": [
                    specification.start,
                    specification.end,
                ],
                "raw_weight": specification.weight,
                "normalized_weight": block.normalized_weight,
            },
            "shared_parameters": selected_payload["parameters"],
            "shared_physical_coordinate": selected_payload[
                "physical_coordinate"
            ],
            "shared_delay_seconds": selected_solution.delay,
            "diagnostics": bag_payload,
            "outputs": {"trajectory_pdf": "trajectory.pdf"},
        }
        baseline._write_json(bag_result_path, bag_result)
        relative_directory = "bags/{}/".format(specification.bag_id)
        bag_outputs[specification.bag_id] = {
            "result_json": relative_directory + "result.json",
            "trajectory_pdf": relative_directory + "trajectory.pdf",
        }
        source_payload = dict(bag_result["source"])
        source_payload["id"] = specification.bag_id
        bag_sources.append(source_payload)

    parameter_lines = _joint_parameter_summary_lines(
        selected_problem,
        selected_solution,
        arguments.continuity_tolerance,
    )
    strict._write_text(parameters_path, parameter_lines)
    _write_delay_profile_pdf(
        delay_profile_path,
        smooth_delay,
        candidate_delays,
        unrefined_losses,
        refined,
        selected_solution.delay,
    )
    result = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(arguments.config.expanduser().resolve()),
        "bags": bag_sources,
        "method": {
            "name": "joint_smooth_lag_then_strict_zoh_multiple_shooting",
            "bag_count": len(config.bags),
            "shared_global_dimension_smooth": GLOBAL_DIMENSION,
            "shared_physical_dimension": strict.PHYSICAL_DIMENSION,
            "shared_delay_index": DELAY_INDEX,
            "bag_local_node_dimension": strict.NODE_DIMENSION,
            "weight_normalization": "alpha_b = raw_weight_b / sum(raw_weight)",
            "pose_residual_normalization": "divide each bag by sqrt(sample count)",
            "pose_metric": "||rho||^2 + phi^T (J0 / m0) phi",
            "soft_prior_application_count": 1,
            "final_model": "strict causal ZOH",
            "physical_and_node_jacobian": "analytic forward sensitivity",
            "lag_jacobian": "analytic smooth-command forward sensitivity",
        },
        "smoothstep_search": {
            "initial_delay_seconds": initial_delay,
            "delay_bounds_seconds": arguments.delay_bounds,
            "width_fractions": widths,
            "stage_results": stage_payloads,
            "estimated_delay_seconds": smooth_delay,
        },
        "exact_zoh_polish": {
            "candidate_delays_seconds": candidate_delays,
            "unrefined_joint_full_rollout_losses_m2": unrefined_losses,
            "refined_candidates": refined_payloads,
            "selected_delay_seconds": selected_solution.delay,
            "smooth_to_selected_delay_difference_seconds": (
                selected_solution.delay - smooth_delay
            ),
        },
        "selection": selected_payload,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "result_json": "result.json",
            "parameters_text": "parameters.txt",
            "delay_profile_pdf": "delay_profile.pdf",
            "bags": bag_outputs,
        },
    }
    baseline._write_json(result_path, result)
    print(
        "smooth shared delay {:.6f}s -> selected strict ZOH delay {:.6f}s, "
        "joint loss {:.9g}".format(
            smooth_delay,
            selected_solution.delay,
            _joint_full_loss(selected_problem, selected_solution),
        ),
        flush=True,
    )
    for path in (result_path, parameters_path, delay_profile_path):
        print("wrote {}".format(path), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(create_argument_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
