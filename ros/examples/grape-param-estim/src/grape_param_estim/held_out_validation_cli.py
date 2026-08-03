"""Worker entry point for leakage-resistant held-out validation."""

from pathlib import Path
from typing import Optional

from grape_param_estim.artifact_io import load_assimilation_run
from grape_param_estim.held_out_validation import (
    forecast_held_out_posterior,
    load_held_out_validation_request,
    prepare_held_out_episode,
    raw_physical_posterior_from_run,
    save_held_out_validation,
    score_held_out_forecasts,
)
from grape_param_estim.inspection import (
    InspectionBagRequest,
    inspect_flight_arrays,
)
from grape_param_estim.progress import CancellationToken
from grape_param_estim.real_rosbag import (
    build_real_flight_episode,
    read_grape_rosbag_arrays,
)
from grape_param_estim.system import ActuatorParameters


def run_request(
    request_path: str,
    output_path: str,
    cancellation_token: Optional[CancellationToken] = None,
) -> Path:
    """Validate provenance, run every member, and publish one artifact."""

    cancellation = (
        CancellationToken()
        if cancellation_token is None
        else cancellation_token
    )
    if not isinstance(cancellation, CancellationToken):
        raise TypeError("cancellation_token must be a CancellationToken")
    request, payload = load_held_out_validation_request(request_path)
    cancellation.raise_if_cancelled()
    source = load_assimilation_run(request.assimilation_run)
    posterior = raw_physical_posterior_from_run(source)
    held_out = request.held_out_bag
    if (
        posterior.source_configuration_fingerprint
        != held_out.configuration_fingerprint
    ):
        raise ValueError(
            "held-out configuration fingerprint differs from source run"
        )

    arrays = read_grape_rosbag_arrays(
        held_out.path,
        compute_sha256=True,
        checkpoint=cancellation.raise_if_cancelled,
    )
    cancellation.raise_if_cancelled()
    if arrays.bag_sha256 != held_out.sha256:
        raise ValueError("held-out rosbag SHA-256 does not match request")
    inspection = inspect_flight_arrays(
        InspectionBagRequest(
            bag_id=held_out.bag_id,
            path=held_out.path,
            episode_index=held_out.episode_index,
            configuration_provenance=held_out.configuration_provenance,
        ),
        arrays,
        preview_max_samples=2,
        source_path=Path(held_out.path),
    )
    actual_fingerprint = inspection.configuration_fingerprint.value
    if actual_fingerprint != held_out.configuration_fingerprint:
        raise ValueError(
            "held-out rosbag configuration fingerprint does not match request"
        )

    episode = build_real_flight_episode(
        arrays,
        sample_period=request.sample_period,
        episode_index=held_out.episode_index,
        start_local=held_out.selected_interval[0],
        end_local=held_out.selected_interval[1],
        window_state=held_out.window_state,
        actuator_parameters=ActuatorParameters(),
        controller_source_revision="held-out-recorded-snapshot",
    )
    scenario, target = prepare_held_out_episode(held_out.bag_id, episode)
    forecasts = forecast_held_out_posterior(
        posterior, scenario, cancellation_token=cancellation
    )
    result = score_held_out_forecasts(forecasts, scenario, target)
    cancellation.raise_if_cancelled()
    return save_held_out_validation(
        output_path,
        request,
        payload,
        posterior,
        scenario,
        target,
        result,
    )


__all__ = ["run_request"]
