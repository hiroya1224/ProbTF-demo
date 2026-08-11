#!/usr/bin/env python3
# Install the rank-aware optimizer after the data-only prior-removal patch.

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import py_compile
import re
import subprocess
import tempfile

RANK_SOLVER_CODE = '\n@dataclass(frozen=True)\nclass _RankAwareLeastSquaresResult:\n    x: np.ndarray\n    cost: float\n    optimality: float\n    nfev: int\n    njev: int\n    status: int\n    success: bool\n    message: str\n    singular_values: np.ndarray\n    svd_threshold: float\n    numerical_rank: int\n    truncated_dimension: int\n    truncated_right_singular_vectors: np.ndarray\n    accepted_steps: int\n    backtrack_evaluations: int\n    final_trust_radius: float\n    maximum_gauge_projection_scaled_norm: float\n\n\ndef _rank_aware_decomposition(\n    jacobian: np.ndarray,\n    svd_rcond: float,\n):\n    """Column-scale the Jacobian and return its truncated SVD."""\n\n    value = np.asarray(jacobian, dtype=float)\n    if value.ndim != 2 or np.any(~np.isfinite(value)):\n        raise ValueError("rank-aware Jacobian must be a finite matrix")\n    if not np.isfinite(svd_rcond) or not 0.0 < svd_rcond < 1.0:\n        raise ValueError(\n            "optimizer SVD rcond must lie strictly between 0 and 1"\n        )\n\n    column_norm = np.linalg.norm(value, axis=0)\n    maximum_column_norm = max(\n        1.0,\n        float(np.max(column_norm)) if column_norm.size else 1.0,\n    )\n    column_floor = (\n        math.sqrt(np.finfo(float).eps) * maximum_column_norm\n    )\n    coordinate_scale = np.ones(value.shape[1], dtype=float)\n    informative_column = column_norm > column_floor\n    coordinate_scale[informative_column] = (\n        1.0 / column_norm[informative_column]\n    )\n\n    scaled_jacobian = value * coordinate_scale[None, :]\n    u, singular, vt = np.linalg.svd(\n        scaled_jacobian,\n        full_matrices=False,\n    )\n    if singular.size == 0:\n        threshold = 0.0\n        retained = np.zeros(0, dtype=bool)\n    else:\n        machine_threshold = (\n            max(scaled_jacobian.shape)\n            * np.finfo(float).eps\n            * float(singular[0])\n        )\n        threshold = max(\n            machine_threshold,\n            float(svd_rcond) * float(singular[0]),\n        )\n        retained = singular > threshold\n\n    return (\n        coordinate_scale,\n        u,\n        singular,\n        vt,\n        float(threshold),\n        retained,\n    )\n\n\ndef _rank_aware_result(\n    *,\n    x,\n    residual,\n    jacobian,\n    svd_rcond,\n    nfev,\n    njev,\n    status,\n    success,\n    message,\n    accepted_steps,\n    backtrack_evaluations,\n    trust_radius,\n    maximum_gauge_projection_scaled_norm,\n    lower,\n    upper,\n):\n    (\n        coordinate_scale,\n        _u,\n        singular,\n        vt,\n        threshold,\n        retained,\n    ) = _rank_aware_decomposition(\n        jacobian,\n        svd_rcond,\n    )\n\n    scaled_gradient = (\n        (jacobian * coordinate_scale[None, :]).T @ residual\n    )\n    bound_tolerance = (\n        128.0\n        * np.finfo(float).eps\n        * np.maximum(1.0, np.abs(x))\n    )\n    feasible_gradient = scaled_gradient.copy()\n    at_lower = (\n        np.isfinite(lower)\n        & (x <= lower + bound_tolerance)\n    )\n    at_upper = (\n        np.isfinite(upper)\n        & (x >= upper - bound_tolerance)\n    )\n    feasible_gradient[\n        at_lower & (scaled_gradient > 0.0)\n    ] = 0.0\n    feasible_gradient[\n        at_upper & (scaled_gradient < 0.0)\n    ] = 0.0\n\n    if np.any(retained):\n        retained_v = vt[retained].T\n        projected_gradient = (\n            retained_v.T @ feasible_gradient\n        )\n        optimality = float(\n            np.max(np.abs(projected_gradient))\n        )\n    else:\n        optimality = 0.0\n\n    truncated_vectors = (\n        vt[~retained].copy()\n        if vt.size\n        else np.empty((0, x.size), dtype=float)\n    )\n\n    return _RankAwareLeastSquaresResult(\n        x=np.asarray(x, dtype=float).copy(),\n        cost=0.5 * float(residual @ residual),\n        optimality=optimality,\n        nfev=int(nfev),\n        njev=int(njev),\n        status=int(status),\n        success=bool(success),\n        message=str(message),\n        singular_values=np.asarray(\n            singular,\n            dtype=float,\n        ).copy(),\n        svd_threshold=float(threshold),\n        numerical_rank=int(np.count_nonzero(retained)),\n        truncated_dimension=int(\n            x.size - np.count_nonzero(retained)\n        ),\n        truncated_right_singular_vectors=truncated_vectors,\n        accepted_steps=int(accepted_steps),\n        backtrack_evaluations=int(backtrack_evaluations),\n        final_trust_radius=float(trust_radius),\n        maximum_gauge_projection_scaled_norm=float(\n            maximum_gauge_projection_scaled_norm\n        ),\n    )\n\n\ndef _rank_aware_least_squares(\n    evaluator,\n    initial,\n    lower,\n    upper,\n    gauge_reference,\n    *,\n    gtol,\n    max_nfev,\n    svd_rcond,\n):\n    """Truncated-SVD Gauss--Newton with local gauge projection."""\n\n    x = np.asarray(initial, dtype=float).copy()\n    lower_value = np.asarray(lower, dtype=float)\n    upper_value = np.asarray(upper, dtype=float)\n    anchor = np.asarray(gauge_reference, dtype=float)\n\n    if (\n        x.ndim != 1\n        or lower_value.shape != x.shape\n        or upper_value.shape != x.shape\n        or anchor.shape != x.shape\n        or np.any(~np.isfinite(x))\n        or np.any(lower_value > upper_value)\n        or np.any(x < lower_value)\n        or np.any(x > upper_value)\n        or np.any(anchor < lower_value)\n        or np.any(anchor > upper_value)\n        or not np.isfinite(gtol)\n        or gtol <= 0.0\n        or int(max_nfev) < 1\n    ):\n        raise ValueError(\n            "rank-aware least-squares inputs are invalid"\n        )\n\n    trust_radius = 1.0\n    minimum_trust_radius = 1.0e-8\n    maximum_trust_radius = 8.0\n    maximum_backtracks = 14\n\n    nfev = 0\n    njev = 0\n    accepted_steps = 0\n    backtrack_evaluations = 0\n    maximum_projection_norm = 0.0\n\n    def evaluate(value):\n        nonlocal nfev, njev\n        evaluation = evaluator(\n            np.asarray(value, dtype=float)\n        )\n        nfev += 1\n        njev += 1\n        residual = np.asarray(\n            evaluation.residual,\n            dtype=float,\n        )\n        jacobian = np.asarray(\n            evaluation.jacobian,\n            dtype=float,\n        )\n        if (\n            residual.ndim != 1\n            or jacobian.shape != (residual.size, x.size)\n            or np.any(~np.isfinite(residual))\n            or np.any(~np.isfinite(jacobian))\n        ):\n            raise FloatingPointError(\n                "rank-aware objective produced non-finite "\n                "residual/Jacobian"\n            )\n        return evaluation, residual, jacobian\n\n    evaluation, residual, jacobian = evaluate(x)\n\n    while True:\n        (\n            coordinate_scale,\n            u,\n            singular,\n            vt,\n            _threshold,\n            retained,\n        ) = _rank_aware_decomposition(\n            jacobian,\n            svd_rcond,\n        )\n        rank = int(np.count_nonzero(retained))\n\n        if rank == 0:\n            return (\n                _rank_aware_result(\n                    x=x,\n                    residual=residual,\n                    jacobian=jacobian,\n                    svd_rcond=svd_rcond,\n                    nfev=nfev,\n                    njev=njev,\n                    status=2,\n                    success=True,\n                    message=(\n                        "no identifiable Jacobian directions "\n                        "remain after SVD truncation"\n                    ),\n                    accepted_steps=accepted_steps,\n                    backtrack_evaluations=(\n                        backtrack_evaluations\n                    ),\n                    trust_radius=trust_radius,\n                    maximum_gauge_projection_scaled_norm=(\n                        maximum_projection_norm\n                    ),\n                    lower=lower_value,\n                    upper=upper_value,\n                ),\n                evaluation,\n            )\n\n        retained_u = u[:, retained]\n        retained_s = singular[retained]\n        retained_v = vt[retained].T\n\n        scaled_jacobian = (\n            jacobian * coordinate_scale[None, :]\n        )\n        scaled_gradient = (\n            scaled_jacobian.T @ residual\n        )\n\n        bound_tolerance = (\n            128.0\n            * np.finfo(float).eps\n            * np.maximum(1.0, np.abs(x))\n        )\n        feasible_gradient = scaled_gradient.copy()\n        at_lower = (\n            np.isfinite(lower_value)\n            & (x <= lower_value + bound_tolerance)\n        )\n        at_upper = (\n            np.isfinite(upper_value)\n            & (x >= upper_value - bound_tolerance)\n        )\n        feasible_gradient[\n            at_lower & (scaled_gradient > 0.0)\n        ] = 0.0\n        feasible_gradient[\n            at_upper & (scaled_gradient < 0.0)\n        ] = 0.0\n\n        projected_gradient = (\n            retained_v.T @ feasible_gradient\n        )\n        optimality = float(\n            np.max(np.abs(projected_gradient))\n        )\n        if optimality <= float(gtol):\n            return (\n                _rank_aware_result(\n                    x=x,\n                    residual=residual,\n                    jacobian=jacobian,\n                    svd_rcond=svd_rcond,\n                    nfev=nfev,\n                    njev=njev,\n                    status=1,\n                    success=True,\n                    message=(\n                        "projected gradient tolerance is satisfied"\n                    ),\n                    accepted_steps=accepted_steps,\n                    backtrack_evaluations=(\n                        backtrack_evaluations\n                    ),\n                    trust_radius=trust_radius,\n                    maximum_gauge_projection_scaled_norm=(\n                        maximum_projection_norm\n                    ),\n                    lower=lower_value,\n                    upper=upper_value,\n                ),\n                evaluation,\n            )\n\n        step_scaled = -retained_v @ (\n            (retained_u.T @ residual) / retained_s\n        )\n        step_scaled_norm = float(\n            np.linalg.norm(step_scaled)\n        )\n        if step_scaled_norm > trust_radius:\n            step_scaled *= (\n                trust_radius / step_scaled_norm\n            )\n        step = coordinate_scale * step_scaled\n\n        blocked_lower = at_lower & (step < 0.0)\n        blocked_upper = at_upper & (step > 0.0)\n        step[\n            blocked_lower | blocked_upper\n        ] = 0.0\n        step_scaled = step / coordinate_scale\n        step_scaled_norm = float(\n            np.linalg.norm(step_scaled)\n        )\n\n        numerical_step_floor = (\n            256.0\n            * np.finfo(float).eps\n            * max(\n                1.0,\n                float(\n                    np.linalg.norm(\n                        (x - anchor)\n                        / coordinate_scale\n                    )\n                ),\n            )\n        )\n        if step_scaled_norm <= numerical_step_floor:\n            return (\n                _rank_aware_result(\n                    x=x,\n                    residual=residual,\n                    jacobian=jacobian,\n                    svd_rcond=svd_rcond,\n                    nfev=nfev,\n                    njev=njev,\n                    status=3,\n                    success=True,\n                    message=(\n                        "rank-aware step is below numerical "\n                        "resolution after active-bound projection"\n                    ),\n                    accepted_steps=accepted_steps,\n                    backtrack_evaluations=(\n                        backtrack_evaluations\n                    ),\n                    trust_radius=trust_radius,\n                    maximum_gauge_projection_scaled_norm=(\n                        maximum_projection_norm\n                    ),\n                    lower=lower_value,\n                    upper=upper_value,\n                ),\n                evaluation,\n            )\n\n        current_cost = 0.5 * float(\n            residual @ residual\n        )\n\n        alpha_bound = 1.0\n        positive_step = step > 0.0\n        negative_step = step < 0.0\n        upper_mask = (\n            positive_step & np.isfinite(upper_value)\n        )\n        lower_mask = (\n            negative_step & np.isfinite(lower_value)\n        )\n        if np.any(upper_mask):\n            alpha_bound = min(\n                alpha_bound,\n                float(\n                    np.min(\n                        (\n                            upper_value[upper_mask]\n                            - x[upper_mask]\n                        )\n                        / step[upper_mask]\n                    )\n                ),\n            )\n        if np.any(lower_mask):\n            alpha_bound = min(\n                alpha_bound,\n                float(\n                    np.min(\n                        (\n                            lower_value[lower_mask]\n                            - x[lower_mask]\n                        )\n                        / step[lower_mask]\n                    )\n                ),\n            )\n        alpha = min(\n            1.0,\n            max(0.0, alpha_bound),\n        )\n        if alpha < 1.0:\n            alpha *= 0.995\n\n        accepted = False\n        for backtrack in range(\n            maximum_backtracks + 1\n        ):\n            if nfev >= int(max_nfev):\n                break\n\n            trial = x + alpha * step\n\n            total_scaled_displacement = (\n                (trial - anchor)\n                / coordinate_scale\n            )\n            identifiable_displacement = (\n                retained_v\n                @ (\n                    retained_v.T\n                    @ total_scaled_displacement\n                )\n            )\n            projected_trial = (\n                anchor\n                + coordinate_scale\n                * identifiable_displacement\n            )\n            projected_trial = np.maximum(\n                projected_trial,\n                lower_value,\n            )\n            projected_trial = np.minimum(\n                projected_trial,\n                upper_value,\n            )\n\n            projection_norm = float(\n                np.linalg.norm(\n                    (projected_trial - trial)\n                    / coordinate_scale\n                )\n            )\n            maximum_projection_norm = max(\n                maximum_projection_norm,\n                projection_norm,\n            )\n\n            trial_displacement_norm = float(\n                np.linalg.norm(\n                    (projected_trial - x)\n                    / coordinate_scale\n                )\n            )\n            if (\n                trial_displacement_norm\n                <= numerical_step_floor\n            ):\n                alpha *= 0.5\n                backtrack_evaluations += 1\n                continue\n\n            (\n                trial_evaluation,\n                trial_residual,\n                trial_jacobian,\n            ) = evaluate(projected_trial)\n            trial_cost = 0.5 * float(\n                trial_residual @ trial_residual\n            )\n\n            if trial_cost < current_cost:\n                x = projected_trial\n                evaluation = trial_evaluation\n                residual = trial_residual\n                jacobian = trial_jacobian\n                accepted = True\n                accepted_steps += 1\n                if backtrack == 0:\n                    trust_radius = min(\n                        maximum_trust_radius,\n                        2.0 * trust_radius,\n                    )\n                elif backtrack >= 2:\n                    trust_radius = max(\n                        minimum_trust_radius,\n                        0.5 * trust_radius,\n                    )\n                break\n\n            alpha *= 0.5\n            backtrack_evaluations += 1\n\n        if not accepted:\n            if nfev >= int(max_nfev):\n                message = (\n                    "maximum number of rank-aware objective "\n                    "evaluations reached"\n                )\n            else:\n                message = (\n                    "rank-aware monotone backtracking found no "\n                    "acceptable descent step"\n                )\n            return (\n                _rank_aware_result(\n                    x=x,\n                    residual=residual,\n                    jacobian=jacobian,\n                    svd_rcond=svd_rcond,\n                    nfev=nfev,\n                    njev=njev,\n                    status=0,\n                    success=False,\n                    message=message,\n                    accepted_steps=accepted_steps,\n                    backtrack_evaluations=(\n                        backtrack_evaluations\n                    ),\n                    trust_radius=trust_radius,\n                    maximum_gauge_projection_scaled_norm=(\n                        maximum_projection_norm\n                    ),\n                    lower=lower_value,\n                    upper=upper_value,\n                ),\n                evaluation,\n            )\n\n\ndef _rank_aware_payload(\n    result,\n    coordinate_names,\n    svd_rcond,\n):\n    singular = np.asarray(\n        result.singular_values,\n        dtype=float,\n    )\n    relative = (\n        singular / singular[0]\n        if singular.size and singular[0] > 0.0\n        else np.zeros_like(singular)\n    )\n    names = tuple(\n        str(name) for name in coordinate_names\n    )\n    weak_directions = []\n    for vector in np.asarray(\n        result.truncated_right_singular_vectors,\n        dtype=float,\n    ):\n        order = np.argsort(\n            np.abs(vector)\n        )[::-1]\n        weak_directions.append(\n            {\n                "scaled_coordinate_direction": vector,\n                "dominant_components": [\n                    {\n                        "name": names[int(index)],\n                        "coefficient": float(\n                            vector[index]\n                        ),\n                    }\n                    for index in order[\n                        : min(6, len(names))\n                    ]\n                ],\n            }\n        )\n\n    return {\n        "method": (\n            "truncated-SVD Gauss-Newton with "\n            "local gauge projection"\n        ),\n        "svd_rcond": float(svd_rcond),\n        "svd_threshold": float(\n            result.svd_threshold\n        ),\n        "singular_values_scaled_jacobian": singular,\n        "relative_singular_values": relative,\n        "numerical_rank": int(\n            result.numerical_rank\n        ),\n        "truncated_dimension": int(\n            result.truncated_dimension\n        ),\n        "truncated_weak_directions": weak_directions,\n        "gauge_reference_policy": (\n            "vehicle-model zero coordinate for physical "\n            "parameters; initial lag coordinates during "\n            "smooth continuation"\n        ),\n        "accepted_steps": int(\n            result.accepted_steps\n        ),\n        "backtrack_evaluations": int(\n            result.backtrack_evaluations\n        ),\n        "final_trust_radius_scaled": float(\n            result.final_trust_radius\n        ),\n        "maximum_gauge_projection_scaled_norm": float(\n            result.maximum_gauge_projection_scaled_norm\n        ),\n    }\n\n\ndef _solve_smooth_pair(\n    problem,\n    initial,\n    width,\n    lower,\n    upper,\n    arguments,\n):\n    evaluator = lambda value: problem.evaluate_smooth(\n        value,\n        width,\n    )\n    gauge_reference = np.concatenate(\n        (\n            np.zeros(\n                PHYSICAL_DIMENSION,\n                dtype=float,\n            ),\n            np.asarray(\n                (\n                    float(arguments.initial_delay),\n                    float(\n                        arguments.initial_gimbal_delay\n                    ),\n                ),\n                dtype=float,\n            ),\n        )\n    )\n    result, evaluation = _rank_aware_least_squares(\n        evaluator,\n        np.asarray(initial, dtype=float),\n        np.asarray(lower, dtype=float),\n        np.asarray(upper, dtype=float),\n        gauge_reference,\n        gtol=float(arguments.gtol),\n        max_nfev=int(\n            arguments.smooth_max_nfev\n        ),\n        svd_rcond=float(\n            arguments.optimizer_svd_rcond\n        ),\n    )\n    payload = _optimizer_payload(result)\n    payload["rank_aware_svd"] = (\n        _rank_aware_payload(\n            result,\n            PHYSICAL_PARAMETER_NAMES\n            + (\n                "rotor_delay_seconds",\n                "gimbal_delay_seconds",\n            ),\n            float(\n                arguments.optimizer_svd_rcond\n            ),\n        )\n    )\n    payload["diagnostics"] = (\n        _optimizer_diagnostics(\n            evaluator,\n            np.asarray(initial, dtype=float),\n            result,\n            evaluation,\n            np.asarray(lower, dtype=float),\n            np.asarray(upper, dtype=float),\n            arguments,\n        )\n    )\n    return result.x, evaluation, payload\n\n\ndef _solve_strict_pair(\n    problem,\n    initial,\n    rotor_delay,\n    gimbal_delay,\n    lower,\n    upper,\n    arguments,\n):\n    evaluator = lambda value: problem.evaluate_strict(\n        value,\n        rotor_delay,\n        gimbal_delay,\n    )\n    gauge_reference = np.zeros(\n        PHYSICAL_DIMENSION,\n        dtype=float,\n    )\n    result, evaluation = _rank_aware_least_squares(\n        evaluator,\n        np.asarray(initial, dtype=float),\n        np.asarray(lower, dtype=float),\n        np.asarray(upper, dtype=float),\n        gauge_reference,\n        gtol=float(arguments.gtol),\n        max_nfev=int(\n            arguments.strict_max_nfev\n        ),\n        svd_rcond=float(\n            arguments.optimizer_svd_rcond\n        ),\n    )\n    payload = _optimizer_payload(result)\n    payload["rank_aware_svd"] = (\n        _rank_aware_payload(\n            result,\n            PHYSICAL_PARAMETER_NAMES,\n            float(\n                arguments.optimizer_svd_rcond\n            ),\n        )\n    )\n    payload["diagnostics"] = (\n        _optimizer_diagnostics(\n            evaluator,\n            np.asarray(initial, dtype=float),\n            result,\n            evaluation,\n            np.asarray(lower, dtype=float),\n            np.asarray(upper, dtype=float),\n            arguments,\n        )\n    )\n    return DynamicsSolution(\n        physical_coordinate=np.asarray(\n            result.x,\n            dtype=float,\n        ).copy(),\n        delay_seconds=float(rotor_delay),\n        gimbal_delay_seconds=float(\n            gimbal_delay\n        ),\n        evaluation=evaluation,\n        optimizer=payload,\n    )\n\n'


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            "{}: expected exactly one occurrence, found {}".format(
                label,
                count,
            )
        )
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.S | re.M,
    )
    if count != 1:
        raise RuntimeError(
            "{}: expected exactly one match, found {}".format(
                label,
                count,
            )
        )
    return updated


def patch_estimator(source: str) -> str:
    required_before = (
        'SCHEMA = "grape-param-estim/minimal-deterministic-savgol-dynamics/v5"',
        "parameter-estimation prior: none (data-only objective)",
        "joint SG dynamics data objective is non-finite",
        "downstream_posterior_prior_used_in_deterministic_estimate",
    )
    missing = [token for token in required_before if token not in source]
    if missing:
        raise RuntimeError(
            "this patch must follow the data-only prior-removal patch; "
            "missing markers: {}".format(", ".join(missing))
        )

    source = replace_once(
        source,
        'SCHEMA = "grape-param-estim/minimal-deterministic-savgol-dynamics/v5"',
        'SCHEMA = "grape-param-estim/minimal-deterministic-savgol-dynamics/v6"',
        "SG schema",
    )

    source = regex_once(
        source,
        r"^def _solve_smooth_pair\(.*?(?=^def _strict_pair_screen\()",
        RANK_SOLVER_CODE + "\n",
        "active split-pair solvers",
    )

    late_anchor = source.index("def _resolve_lag_defaults")
    if 'tr_solver="exact"' in source[late_anchor:]:
        raise RuntimeError(
            "an active late SciPy exact-TRF pair solver remains"
        )

    gtol_argument = '''    parser.add_argument("--gtol", type=float, default=1.0e-6)
'''
    gtol_replacement = '''    parser.add_argument("--gtol", type=float, default=1.0e-6)
    parser.add_argument(
        "--optimizer-svd-rcond",
        type=float,
        default=1.0e-8,
        help=(
            "Relative singular-value threshold for the data-only "
            "rank-aware optimizer. Directions with sigma/sigma_max "
            "below this value are treated as local gauge/ridge directions."
        ),
    )
'''
    source = replace_once(
        source,
        gtol_argument,
        gtol_replacement,
        "optimizer SVD CLI option",
    )

    run_anchor = '''    _resolve_lag_defaults(arguments)
    if (
        not np.isfinite(arguments.optimizer_svd_rcond)
        or not 0.0 < arguments.optimizer_svd_rcond < 1.0
    ):
        raise ValueError(
            "--optimizer-svd-rcond must lie strictly between 0 and 1"
        )


    original_stdout = sys.stdout
'''
    if run_anchor not in source:
        older_anchor = '''    _resolve_lag_defaults(arguments)


    original_stdout = sys.stdout
'''
        replacement = '''    _resolve_lag_defaults(arguments)
    if (
        not np.isfinite(arguments.optimizer_svd_rcond)
        or not 0.0 < arguments.optimizer_svd_rcond < 1.0
    ):
        raise ValueError(
            "--optimizer-svd-rcond must lie strictly between 0 and 1"
        )


    original_stdout = sys.stdout
'''
        source = replace_once(
            source,
            older_anchor,
            replacement,
            "optimizer SVD validation",
        )

    source = replace_once(
        source,
        '''        "optimizer_rank_handling": (
            "SciPy least_squares TRF with dense exact trust-region solver "
            "and Jacobian scaling; no prior regularization"
        ),
''',
        '''        "optimizer_rank_handling": (
            "custom truncated-SVD Gauss-Newton in Jacobian-scaled "
            "coordinates with local gauge projection; no prior regularization"
        ),
''',
        "optimizer metadata",
    )

    settings_anchor = '''        "optimizer_tolerances": {
            "ftol": arguments.ftol,
            "xtol": arguments.xtol,
            "gtol": arguments.gtol,
            "note": (
                "ftol and xtol are disabled by default in the SG estimator; "
                "gtol or max_nfev terminates the solve unless explicitly overridden"
            ),
        },
'''
    settings_replacement = '''        "optimizer_tolerances": {
            "ftol": arguments.ftol,
            "xtol": arguments.xtol,
            "gtol": arguments.gtol,
            "note": (
                "The active rank-aware optimizer uses projected-gradient "
                "gtol and max_nfev. ftol/xtol are retained only for CLI/report "
                "compatibility with earlier runs."
            ),
        },
        "optimizer_svd_rcond": float(
            arguments.optimizer_svd_rcond
        ),
        "optimizer_gauge_reference": (
            "vehicle-model zero coordinate for physical parameters"
        ),
'''
    source = replace_once(
        source,
        settings_anchor,
        settings_replacement,
        "optimizer settings metadata",
    )

    report_anchor = '''    optimizer = selected.optimizer
    diagnostics = optimizer.get("diagnostics") if isinstance(optimizer, Mapping) else None
'''
    report_replacement = '''    optimizer = selected.optimizer
    rank_aware = (
        optimizer.get("rank_aware_svd")
        if isinstance(optimizer, Mapping)
        else None
    )
    if isinstance(rank_aware, Mapping):
        lines.extend(
            [
                "",
                "Rank-aware data-only solver",
                "  method={}".format(rank_aware.get("method")),
                "  svd rcond={}".format(rank_aware.get("svd_rcond")),
                "  retained rank={} / {}".format(
                    rank_aware.get("numerical_rank"),
                    PHYSICAL_DIMENSION,
                ),
                "  truncated dimension={}".format(
                    rank_aware.get("truncated_dimension")
                ),
                "  relative singular values={}".format(
                    rank_aware.get("relative_singular_values")
                ),
                "  accepted steps={}".format(
                    rank_aware.get("accepted_steps")
                ),
                "  backtrack evaluations={}".format(
                    rank_aware.get("backtrack_evaluations")
                ),
                "  max gauge projection scaled norm={}".format(
                    rank_aware.get(
                        "maximum_gauge_projection_scaled_norm"
                    )
                ),
            ]
        )
    diagnostics = optimizer.get("diagnostics") if isinstance(optimizer, Mapping) else None
'''
    source = replace_once(
        source,
        report_anchor,
        report_replacement,
        "rank-aware parameter report",
    )

    required_after = (
        "minimal-deterministic-savgol-dynamics/v6",
        "class _RankAwareLeastSquaresResult",
        "def _rank_aware_least_squares(",
        "truncated-SVD Gauss-Newton with ",
        "--optimizer-svd-rcond",
        '"optimizer_svd_rcond"',
        "maximum_gauge_projection_scaled_norm",
    )
    missing_after = [
        token for token in required_after
        if token not in source
    ]
    if missing_after:
        raise RuntimeError(
            "patched estimator is missing markers: {}".format(
                ", ".join(missing_after)
            )
        )

    ast.parse(source)
    return source


def patch_dictionary(source: str) -> str:
    if (
        "data-only deterministic SG fit" not in source
        or "No Gaussian physical-prior residual" not in source
    ):
        raise RuntimeError(
            "data dictionary does not contain the data-only patch"
        )

    old = '''The split-lag physical solves use SciPy `least_squares` with `method="trf"`,
`tr_solver="exact"`, and `x_scale="jac"`. Thus the dense trust-region solve is
rank-aware without introducing a probabilistic prior. The reported Jacobian
spectrum remains the diagnostic for exact or near ridge directions.
'''
    new = '''The split-lag physical solves use a custom rank-aware
truncated-SVD Gauss--Newton method. At every iteration the analytic Jacobian is
column-scaled, decomposed as `J = U S V^T`, and singular directions satisfying

```text
sigma_i / sigma_max <= --optimizer-svd-rcond
```

are excluded from the deterministic step. The default threshold is `1e-8`.

After every trial step, the total displacement from the fixed vehicle-model
reference is projected back onto the current retained right-singular subspace.
This prevents small local steps from accumulating along a curved ridge. It is a
deterministic gauge choice and introduces no residual, penalty, covariance, or
probabilistic prior.

Finite command-lag bounds are handled as active constraints. The physical 14-D
coordinates remain unbounded. Singular values, retained numerical rank,
truncated weak directions, accepted steps, and gauge-projection magnitude are
reported in the optimizer payload.
'''
    source = replace_once(
        source,
        old,
        new,
        "rank-aware optimizer documentation",
    )

    source += '''

## 12. Rank-aware deterministic gauge selection

The deterministic optimizer uses the analytic Jacobian spectrum directly. In
Jacobian-scaled coordinates, directions below `--optimizer-svd-rcond` do not
participate in the Gauss--Newton step.

The representative point is selected without a probabilistic prior: after a
trial step, the total displacement from the vehicle-model zero-coordinate
reference is projected onto the retained right-singular subspace. Therefore the
optimizer moves wherever the data identify a direction and preserves the
reference gauge where the local data Jacobian does not identify one.

The default `--optimizer-svd-rcond` is `1e-8`. If a genuinely near-null ridge,
rather than an exact numerical nullspace, still causes drift, this threshold can
be raised explicitly and the resulting retained/truncated spectrum is recorded
in the optimizer output.
'''
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("."),
        help=(
            "ros/examples/grape-param-estim/minimal directory "
            "(default: current directory)"
        ),
    )
    root = parser.parse_args().root.expanduser().resolve()

    estimator_path = root / "deterministic_savgol_dynamics_estimator.py"
    dictionary_path = root / "deterministic_savgol_dynamics_data_dictionary.md"
    for path in (estimator_path, dictionary_path):
        if not path.is_file():
            raise SystemExit("missing target: {}".format(path))

    originals = {
        estimator_path: estimator_path.read_bytes(),
        dictionary_path: dictionary_path.read_bytes(),
    }
    replacements = {
        estimator_path: patch_estimator(
            originals[estimator_path].decode("utf-8")
        ).encode("utf-8"),
        dictionary_path: patch_dictionary(
            originals[dictionary_path].decode("utf-8")
        ).encode("utf-8"),
    }

    with tempfile.TemporaryDirectory(
        prefix="grape-rank-aware-svd-"
    ) as temporary_directory:
        temporary_estimator = (
            Path(temporary_directory)
            / "deterministic_savgol_dynamics_estimator.py"
        )
        temporary_estimator.write_bytes(
            replacements[estimator_path]
        )
        py_compile.compile(
            str(temporary_estimator),
            doraise=True,
        )

    written = []
    try:
        for path, data in replacements.items():
            temporary_path = path.with_name(
                path.name + ".patch-tmp"
            )
            temporary_path.write_bytes(data)
            os.replace(temporary_path, path)
            written.append(path)

        subprocess.run(
            [
                "git",
                "diff",
                "--check",
                "--",
                estimator_path.name,
                dictionary_path.name,
            ],
            cwd=root,
            check=True,
        )
    except Exception:
        for path in written:
            path.write_bytes(originals[path])
        raise

    print("installed rank-aware data-only SG optimizer")
    print("  prior residual: still absent")
    print("  step: truncated-SVD Gauss-Newton")
    print("  gauge: total displacement projected to retained SVD subspace")
    print("  default SVD rcond: 1e-8")
    print("  physical coordinate bounds: unchanged/unbounded")
    print("  finite lag bounds: active-bound aware")
    print("  schema: minimal-deterministic-savgol-dynamics/v6")
    print("  backups: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
