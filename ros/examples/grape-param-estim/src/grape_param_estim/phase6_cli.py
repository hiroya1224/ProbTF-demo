"""Command-line entry point for Phase-6 posterior-predictive decisions."""

import argparse
import json
import math

from grape_param_estim.posterior_predictive import (
    ControllerParameterCandidate,
    PosteriorPredictiveWeights,
    TrackingLossDefinition,
    default_controller_parameter_candidates,
    evaluate_posterior_predictive,
    input_from_phase5_artifact,
    save_posterior_predictive_decision,
)


def _candidate(value: str) -> ControllerParameterCandidate:
    fields = value.split(",")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "candidate must be ID,MASS_SCALE,ROLL_SCALE,PITCH_SCALE,YAW_SCALE"
        )
    try:
        return ControllerParameterCandidate(
            fields[0],
            controller_mass_scale=float(fields[1]),
            roll_pid_scale=float(fields[2]),
            pitch_pid_scale=float(fields[3]),
            yaw_pid_scale=float(fields[4]),
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate explicit controller parameter candidates against every "
            "raw member of one selected-mode Phase-5 posterior."
        )
    )
    parser.add_argument("--phase5-artifact", required=True)
    parser.add_argument(
        "--output", default="grape_phase6_posterior_predictive.npz"
    )
    parser.add_argument(
        "--residual-policy",
        choices=("posterior_replay", "zero"),
        default="posterior_replay",
        help=(
            "posterior_replay repeats each Phase-5 residual path; zero makes "
            "the counterfactual zero-residual assumption explicit"
        ),
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=_candidate,
        help=(
            "repeatable ID,MASS_SCALE,ROLL_SCALE,PITCH_SCALE,YAW_SCALE; "
            "omitting it uses the nine audited defaults"
        ),
    )
    parser.add_argument("--translation-scale", type=float, default=0.10)
    parser.add_argument(
        "--rotation-scale-deg", type=float, default=10.0
    )
    parser.add_argument("--failure-threshold", type=float, default=1.0)
    parser.add_argument("--cvar-level", type=float, default=0.90)
    parser.add_argument("--mean-weight", type=float, default=1.0)
    parser.add_argument("--cvar-weight", type=float, default=0.5)
    parser.add_argument("--failure-weight", type=float, default=5.0)
    parser.add_argument("--change-weight", type=float, default=0.05)
    arguments = parser.parse_args()

    predictive_input = input_from_phase5_artifact(
        arguments.phase5_artifact,
        residual_policy=arguments.residual_policy,
    )
    candidates = (
        tuple(arguments.candidate)
        if arguments.candidate
        else default_controller_parameter_candidates()
    )
    loss = TrackingLossDefinition(
        translation_scale=arguments.translation_scale,
        rotation_scale=(
            arguments.rotation_scale_deg * 3.141592653589793 / 180.0
        ),
    )
    weights = PosteriorPredictiveWeights(
        mean_tracking_loss=arguments.mean_weight,
        cvar_tracking_loss=arguments.cvar_weight,
        failure_probability=arguments.failure_weight,
        parameter_change=arguments.change_weight,
    )
    decision = evaluate_posterior_predictive(
        predictive_input,
        candidates=candidates,
        failure_threshold=arguments.failure_threshold,
        cvar_level=arguments.cvar_level,
        loss_definition=loss,
        weights=weights,
    )
    destination = save_posterior_predictive_decision(
        arguments.output, decision
    )
    print(
        json.dumps(
            {
                "schema": "grape-weak-constraint/phase6-summary",
                "source": str(arguments.phase5_artifact),
                "output": str(destination),
                "selected_mode": predictive_input.selected_mode_id,
                "scenario_assumption": (
                    predictive_input.scenario_assumption
                ),
                "members": int(predictive_input.member_ids.size),
                "recommendation_available": (
                    decision.recommendation_available
                ),
                "selected_candidate": (
                    decision.selected_candidate.candidate_id
                    if decision.recommendation_available
                    else None
                ),
                "candidates": [
                    {
                        "id": value.candidate.candidate_id,
                        "scales": value.candidate.scales.tolist(),
                        "controller_mass": (
                            predictive_input.controller_parameters.mass
                            * value.candidate.controller_mass_scale
                        ),
                        "mean_tracking_loss": value.mean_tracking_loss,
                        "cvar_tracking_loss": value.cvar_tracking_loss,
                        "failure_probability": value.failure_probability,
                        "forecast_failures": int(
                            value.forecast_success.size
                            - value.forecast_success.sum()
                        ),
                        "parameter_change_magnitude": (
                            value.parameter_change_magnitude
                        ),
                        "decision_score": (
                            value.decision_score
                            if math.isfinite(value.decision_score)
                            else None
                        ),
                    }
                    for value in decision.evaluations
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
