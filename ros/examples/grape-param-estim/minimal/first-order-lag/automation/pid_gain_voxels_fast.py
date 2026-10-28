#!/usr/bin/env python3
"""Exact 65-level PID volumes with gain-scaled hover-trim seeds and interior anchors."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Mapping
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pid_gain_voxels_interior import (
    HERE, SCHEMA, RUNTIME_VERSIONS, PID_GROUPS, COVARIANCE_NAMES,
    GAIN_CAPS, Volume, axes_for_group, coordinates_for_nodes,
    atomic_json, utc_now, reference_payload, write_index, plot_volume,
    source_commit, load_estimate_json, _recorded_gain_matrix,
    load_configuration, load_vehicle_model, _sobol_quotient_coordinates,
    quotient_to_scale_free_plants, ScaleFreePlant, actuator_parameters_from_estimate,
)
from pid_voxel_fast_evaluator import FastGainEvaluator

def input_fingerprint(arguments: argparse.Namespace, estimate_path: Path,
                      success_path: Path, model_path: Path) -> tuple[str, Mapping[str, str]]:
    paths = {estimate_path, model_path}
    paths.update(HERE.glob("*.py"))
    paths.update(HERE.parent.glob("*.py"))
    paths.update((HERE.parent.parent / "src").rglob("*.py"))
    paths.add(HERE.parent / "single_bag_savgol_reports.py")
    paths.add(Path(__file__).resolve())
    paths.add(Path(__file__).with_name('pid_gain_voxels_interior.py').resolve())
    paths.add(Path(__file__).with_name('pid_voxel_fast_evaluator.py').resolve())
    paths.add(Path(__file__).with_name('pid_voxel_domains.py').resolve())
    paths.add(Path(__file__).with_name('pid_voxel_survival.py').resolve())
    paths.add(Path(arguments.domain_config).expanduser().resolve())
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    configuration = {"input_hashes": hashes, "covariance": arguments.covariance,
                     "samples": arguments.samples, "seed": arguments.seed,
                     "grid_sizes": arguments.grid_sizes, "groups": arguments.groups,
                     "context": "recorded_failure", "evaluator": "gain_scaled_full_exact/v1", "schema": SCHEMA, "runtime_versions": RUNTIME_VERSIONS}
    fingerprint = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    return fingerprint, hashes


def analyze(arguments: argparse.Namespace) -> None:
    sizes = [int(x) for x in arguments.grid_sizes]
    if not sizes or any(x < 3 or (x - 1) & (x - 2) for x in sizes):
        raise ValueError("grid sizes must be 2^k+1 (3, 5, 9, 17, 33, ...)")
    if sizes != sorted(set(sizes)):
        raise ValueError("grid sizes must strictly increase")
    if len(arguments.groups) != len(set(arguments.groups)):
        raise ValueError("groups must not repeat")
    if arguments.samples <= 0 or arguments.workers <= 0 or arguments.batch_size <= 0 or arguments.preview_seconds <= 0:
        raise ValueError("sample, worker, batch, and preview settings must be positive")
    estimate_path = Path(arguments.estimate_json).expanduser().resolve()
    success_path = (Path(arguments.success_json).expanduser().resolve()
                    if arguments.success_json else None)
    estimate = load_estimate_json(estimate_path)
    gains = _recorded_gain_matrix(estimate)
    base_gains = gains.copy()
    domain_bounds, domain_anchors, domain_report, _, domain_clearance = load_configuration(
        arguments.domain_config, HERE, load_estimate_json, _recorded_gain_matrix,
        [estimate_path])
    model_path = Path(estimate["input"]["vehicle_model"]).expanduser().resolve()
    model = load_vehicle_model(model_path)
    fingerprint, hashes = input_fingerprint(arguments, estimate_path, success_path, model_path)
    output = Path(arguments.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("output contains a different calculation; choose a new --output directory")
    quotient, sampling = _sobol_quotient_coordinates(estimate, arguments.covariance, arguments.samples, arguments.seed)
    plants = quotient_to_scale_free_plants(estimate, quotient, model.parameters, ScaleFreePlant)
    controller_dt = float(estimate["controller_timing"]["median_seconds"])
    case = str(estimate.get("case_name", estimate_path.parent.name))
    totals = {}
    for group in arguments.groups:
        j = PID_GROUPS.index(group)
        axes = axes_for_group(sizes[-1], domain_bounds[group], domain_anchors[group], domain_clearance)
        totals[group] = int(np.prod([len(a) for a in axes]))
    manifest = {"schema": SCHEMA, "fingerprint": fingerprint, "source_commit": source_commit(),
                "runtime_versions": RUNTIME_VERSIONS,
                "input_sha256": hashes, "case": case, "arguments": vars(arguments),
                "covariance": arguments.covariance, "plant_samples": arguments.samples,
                "sampling": sampling, "controller_dt_seconds": controller_dt,
                "implementation": "gain_scaled_full_exact/v1",
                "warm_start": "preserve K_I * integral_error; exact equilibrium and pole workers unchanged",
                "fixed_controller_gains": base_gains, "full_recorded_case_gains": gains,
                "groups": list(PID_GROUPS), "gain_names": ["P", "I", "D"],
                "exploration_bounds": domain_report,
                "gain_coordinate_scales": GAIN_CAPS, "final_node_counts": totals,
                "criterion": "stable fraction = count(finite rho < 1) / all N fixed plant samples; RdYlGn [0,1]; alpha=0.95",
                "absolute_margin_diagnostic": "delta = 1 - max_i spectral_radius(exact_26_state_sampled_closed_loop_i); raw Volume.classification and *_margin counts are diagnostics, not plot colors",
                "semantics": ["Only one P/I/D triple varies; the other nine gains equal this recorded failure controller.",
                              "Successful-flight gains are an optional post-map full-12-gain comparison only.",
                              "The failure marker is this actual recorded full-12-gain controller; no success marker is projected into a failure-context slice.",
                              "Every original plant sample remains in the denominator; non-finite radii do not count as stable and are counted separately.",
                              "Primary stability_fraction.npz classification: 0 pending, 2 below 95%, 3 at or above 95%; this is not an all-samples guarantee.",
                              "Grid-node data; no interpolation or certification of entire voxel cells.",
                              "Exploration bounds are independent of the GAIN_CAPS normalization scales.",
                              "The sole failure anchor and declared fixed scale factors define the box without successful-flight inputs."]}
    atomic_json(manifest_path, manifest)
    samples_path = output / "plant_samples.npz"
    if samples_path.exists():
        with np.load(samples_path) as saved_samples:
            if not np.array_equal(saved_samples["quotient_coordinates"], quotient):
                raise ValueError("saved plant coordinates differ from the fixed sample set")
    else:
        temporary = output / "plant_samples.tmp.npz"
        np.savez_compressed(temporary, quotient_coordinates=quotient)
        os.replace(temporary, samples_path)
    state = {"status": "running", "started_utc": utc_now(), "updated_utc": utc_now(),
             "pid": os.getpid(), "case": case, "completed_volumes": [], "final_node_counts": totals}
    if (output / "status.json").exists():
        old = json.loads((output / "status.json").read_text())
        state["first_started_utc"] = old.get("first_started_utc", old.get("started_utc"))
    atomic_json(output / "status.json", state)
    write_index(output, state)
    stop = [False]

    def request_stop(_signal: int, _frame: Any) -> None:
        stop[0] = True
        print("Stop requested; finishing and checkpointing the current batch.", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    evaluator = FastGainEvaluator(reference_gains=base_gains, plants=plants, vehicle_model=model,
                                             actuator_parameters=actuator_parameters_from_estimate(estimate),
                                             controller_dt=controller_dt, workers=arguments.workers)
    current = None
    started = time.monotonic()
    new_nodes = 0
    previous = {}
    try:
        print(f"{case}: {arguments.samples} fixed plants, {arguments.workers} workers, final nodes={sum(totals.values())}", flush=True)
        references_path = output / "references.json"
        if references_path.exists():
            references = json.loads(references_path.read_text())
        else:
            row = evaluator.evaluate_many([(gains / GAIN_CAPS).ravel()], progress=False)[0]
            references = {"recorded_case": reference_payload(row)}
            temporary = output / "reference_trims.tmp.npz"
            np.savez_compressed(temporary, recorded_case=row.trim_vectors)
            os.replace(temporary, output / "reference_trims.npz")
            atomic_json(references_path, references)
        warm = np.load(output / "reference_trims.npz")["recorded_case"]
        print(f"Recorded failure controller exact delta={references['recorded_case']['delta']}", flush=True)
        for size in sizes:
            for group in arguments.groups:
                j = PID_GROUPS.index(group)
                axes = axes_for_group(size, domain_bounds[group], domain_anchors[group], domain_clearance)
                current = Volume(output / group / f"grid_{size}", axes, arguments.samples, fingerprint, group, base_gains)
                if group in previous:
                    current.inherit(previous[group])
                pending = np.argwhere(~current.fields["evaluated"])
                # Order by distance from the recorded failure controller only.
                q_points = coordinates_for_nodes(pending, axes, base_gains, j)
                order = np.argsort(np.linalg.norm(q_points - (base_gains / GAIN_CAPS).ravel(), axis=1), kind="stable")
                last_preview = -float("inf")
                print(f"{group} grid {size}, actual shape={current.shape}, pending={len(pending)}", flush=True)
                for begin in range(0, len(order), arguments.batch_size):
                    if stop[0]:
                        break
                    chosen = order[begin:begin + arguments.batch_size]
                    tick = time.monotonic()
                    rows = evaluator.evaluate_many(q_points[chosen], initial_trims=warm, progress=False)
                    current.store(pending[chosen], rows)
                    new_nodes += len(rows)
                    # Bounded continuation cache: do not retain every grid
                    # point and scan a growing O(number_of_nodes) cache.
                    evaluator._cache = dict(list(evaluator._cache.items())[-128:])
                    counts = current.counts()
                    state.update(status="running", updated_utc=utc_now(), current_group=group,
                                 current_grid=size, current_shape=current.shape, current_counts=counts,
                                 elapsed_seconds=time.monotonic() - started, nodes_evaluated_this_run=new_nodes,
                                 seconds_per_new_node=(time.monotonic() - started) / max(new_nodes, 1),
                                 latest_volume=str(current.directory))
                    atomic_json(output / "status.json", state)
                    atomic_json(current.directory / "status.json", {"updated_utc": utc_now(), **counts})
                    print(f"{group} grid={size} {counts['evaluated']}/{counts['total']} | absolute_delta_positive={counts['positive_margin']} absolute_delta_non_positive={counts['non_positive_margin']} unresolved={counts['unresolved']} | batch {time.monotonic() - tick:.1f}s", flush=True)
                    if time.monotonic() - last_preview >= arguments.preview_seconds:
                        current.export(complete=False)
                        plot_volume(current, group, case, references)
                        write_index(output, state)
                        last_preview = time.monotonic()
                complete = not np.any(~current.fields["evaluated"])
                current.export(complete=complete)
                plot_volume(current, group, case, references)
                atomic_json(current.directory / "status.json", {"updated_utc": utc_now(), "complete": complete, **current.counts()})
                write_index(output, state)
                if stop[0]:
                    state["status"] = "stopped"
                    return
                previous[group] = current
                state["completed_volumes"].append(str(current.directory))
                atomic_json(output / "status.json", state)
        state["status"] = "complete"
        # This comparison cannot seed, select samples for, or alter any map.
        if success_path is not None:
            success_gains = _recorded_gain_matrix(load_estimate_json(success_path))
            row = evaluator.evaluate_many([(success_gains / GAIN_CAPS).ravel()],
                                          initial_trims=warm, progress=False)[0]
            references["success_full_controller"] = reference_payload(row)
            references["success_comparison_input"] = {
                "path": str(success_path),
                "sha256": hashlib.sha256(success_path.read_bytes()).hexdigest(),
                "role": "post-map full-12-gain comparison only"}
            atomic_json(references_path, references)
        print("All exact PID volumes complete.", flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state.update(updated_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
        atomic_json(output / "status.json", state)
        write_index(output, state)
        evaluator.close()


def parser() -> argparse.ArgumentParser:
    selected = argparse.ArgumentParser(description=__doc__)
    selected.add_argument("--estimate-json", default=str(HERE / "outputs/failure1/estimate.json"))
    selected.add_argument("--success-json", default=None,
                          help="Optional full-controller comparison, evaluated only after the maps are complete")
    selected.add_argument("--output", default=str(HERE / "outputs/failure1/pid_gain_voxels_fast_interior_failure_context"))
    selected.add_argument("--domain-config", default=str(Path(__file__).with_name('pid_voxel_domains.json')))
    selected.add_argument("--validate-domain-only", action='store_true')
    selected.add_argument("--covariance", choices=COVARIANCE_NAMES, default="conservative_fusion")
    selected.add_argument("--samples", type=int, default=128)
    selected.add_argument("--seed", type=int, default=0)
    selected.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    selected.add_argument("--batch-size", type=int, default=16)
    selected.add_argument("--grid-sizes", nargs="+", type=int, default=[3, 5, 9, 17, 33, 65])
    selected.add_argument("--groups", nargs="+", choices=PID_GROUPS, default=["roll_pitch", "xy", "z", "yaw"])
    selected.add_argument("--preview-seconds", type=float, default=900.0)
    return selected


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.validate_domain_only:
        bounds, anchors, report, _, clearance = load_configuration(
            arguments.domain_config, HERE, load_estimate_json, _recorded_gain_matrix,
            [arguments.estimate_json])
        for group in arguments.groups:
            axes = axes_for_group(max(arguments.grid_sizes), bounds[group], anchors[group], clearance)
            report[group]['grid_shape'] = [len(axis) for axis in axes]
            report[group]['grid_nodes'] = int(np.prod([len(axis) for axis in axes]))
        print(json.dumps(report, indent=2))
        raise SystemExit(0)
    output_path = Path(arguments.output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another PID voxel job is already writing this output directory.")
        analyze(arguments)
