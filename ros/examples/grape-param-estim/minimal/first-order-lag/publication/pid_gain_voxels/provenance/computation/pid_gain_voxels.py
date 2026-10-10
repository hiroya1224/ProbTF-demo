#!/usr/bin/env python3
"""Exact P/I/D volumes with a fixed, explicit nine-gain controller context.

Every grid node is evaluated on the same ordered posterior plant samples.
No gain-affine surrogate, max projection, or stable-baseline conditioning is
used.  Values at nodes do not certify stability throughout a voxel cell.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent, HERE.parent.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core import (COVARIANCE_NAMES, PID_GROUPS, actuator_parameters_from_estimate,
                  load_estimate_json, quotient_to_scale_free_plants)
from grape_param_estim.gimbalrotor_pid_postprocess import ScaleFreePlant, load_vehicle_model
from pid_safe_margin_slices import (GAIN_CAPS, MarginEvaluation, _recorded_gain_matrix,
                                   _sobol_quotient_coordinates)
from pid_spectral_gradient_path import BatchedForwardMarginEvaluator
from single_bag_savgol_reports import json_sanitize, source_commit

SCHEMA = "grape-param-estim/exact-pid-gain-volume/v1"
RUNTIME_VERSIONS = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__}
FIELD_SPECS = {
    "delta": (np.float64, np.nan),
    "worst_spectral_radius": (np.float64, np.nan),
    "worst_sample_index": (np.int32, -1),
    "pole_valid_count": (np.int32, 0),
    "near_kink_count": (np.int32, 0),
    "spectral_radius": (np.float64, np.nan),
    "evaluated": (np.bool_, False),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_sanitize(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def axes_for_group(size: int, caps: np.ndarray, anchors: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Include recorded triples exactly, rather than putting them on a nearby node."""
    if size < 2:
        raise ValueError("grid size must be at least 2")
    selected = np.asarray(caps, dtype=float)
    if selected.shape != (3,) or np.any(~np.isfinite(selected)) or np.any(selected <= 0):
        raise ValueError("three positive finite exploration caps required")
    triples = np.asarray(anchors, dtype=float)
    if triples.ndim != 2 or triples.shape[1] != 3 or np.any(~np.isfinite(triples)):
        raise ValueError("anchor gains must be finite triples")
    if np.any(triples < 0) or np.any(triples > selected):
        raise ValueError("anchor gains outside exploration box")
    return tuple(np.unique(np.r_[np.linspace(0, selected[j], size), triples[:, j]]) for j in range(3))


def coordinates_for_nodes(indices: np.ndarray, axes: Sequence[np.ndarray],
                          base_gains: np.ndarray, group: int) -> np.ndarray:
    """Change exactly the selected P/I/D triple; keep the other nine gains fixed."""
    index = np.asarray(indices, dtype=int)
    gains = np.repeat(np.asarray(base_gains)[None, :, :], len(index), axis=0)
    for j in range(3):
        gains[:, group, j] = np.asarray(axes[j])[index[:, j]]
    return (gains / GAIN_CAPS[None, :, :]).reshape(-1, 12)


class Volume:
    """Per-field memmaps; evaluated is committed last so partial batches can resume."""

    def __init__(self, directory: Path, axes: Sequence[np.ndarray], samples: int,
                 fingerprint: str, group: str, base_gains: np.ndarray) -> None:
        self.directory = directory
        self.axes = tuple(np.asarray(a, dtype=float) for a in axes)
        self.shape = tuple(len(a) for a in self.axes)
        self.samples = int(samples)
        self.fingerprint = fingerprint
        meta = {"schema": SCHEMA, "fingerprint": fingerprint, "group": group,
                "shape": self.shape, "axes": [a.tolist() for a in self.axes],
                "axis_order": ["P", "I", "D"], "plant_samples": self.samples,
                "full_fixed_controller_gains": np.asarray(base_gains).tolist(),
                "association": "grid nodes, not certified cell interiors"}
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / "metadata.json"
        resuming = metadata_path.exists()
        if resuming:
            if json.loads(metadata_path.read_text()) != json.loads(json.dumps(meta)):
                raise ValueError(f"incompatible checkpoint: {metadata_path}")
        else:
            # Existing arrays without metadata are never silently reinterpreted.
            if any(directory.glob("*.npy")):
                raise ValueError(f"arrays exist without metadata: {directory}")
            atomic_json(metadata_path, meta)
        self.fields = {}
        existing_done = directory / "evaluated.npy"
        if resuming and existing_done.exists() and np.any(np.load(existing_done, mmap_mode="r")):
            missing = [name for name in FIELD_SPECS if not (directory / f"{name}.npy").exists()]
            if missing:
                raise ValueError(f"committed checkpoint is missing fields: {missing}")
        for name, (dtype, fill) in FIELD_SPECS.items():
            shape = self.shape + ((self.samples,) if name == "spectral_radius" else ())
            path = directory / f"{name}.npy"
            if path.exists():
                field = np.load(path, mmap_mode="r+")
                if field.shape != shape or field.dtype != np.dtype(dtype):
                    raise ValueError(f"invalid checkpoint array: {path}")
            else:
                field = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
                field[:] = fill
                field.flush()
            self.fields[name] = field

    def store(self, indices: np.ndarray, rows: Sequence[MarginEvaluation]) -> None:
        if len(indices) != len(rows):
            raise ValueError("node/result count mismatch")
        for index, row in zip(indices, rows):
            ix = tuple(int(i) for i in index)
            if np.asarray(row.spectral_radius).shape != (self.samples,):
                raise ValueError("plant sample count changed")
            self.fields["delta"][ix] = row.delta
            self.fields["worst_spectral_radius"][ix] = row.worst_spectral_radius
            self.fields["worst_sample_index"][ix] = -1 if row.worst_sample_index is None else row.worst_sample_index
            self.fields["pole_valid_count"][ix] = np.count_nonzero(row.pole_valid_mask)
            self.fields["near_kink_count"][ix] = np.count_nonzero(row.near_kink_mask)
            self.fields["spectral_radius"][ix] = row.spectral_radius
        for name, field in self.fields.items():
            if name != "evaluated":
                field.flush()
        for index in indices:
            self.fields["evaluated"][tuple(index)] = True
        self.fields["evaluated"].flush()

    def inherit(self, previous: Volume) -> int:
        if self.samples != previous.samples or self.fingerprint != previous.fingerprint:
            raise ValueError("cannot reuse a different plant sample set")
        positions = []
        for old, new in zip(previous.axes, self.axes):
            p = np.searchsorted(new, old)
            if np.any(p >= len(new)) or not np.array_equal(new[p], old):
                raise ValueError("refinement grids must be nested")
            positions.append(p)
        source_indices = np.argwhere(previous.fields["evaluated"])
        target_indices = np.column_stack([positions[j][source_indices[:, j]] for j in range(3)])
        src = tuple(source_indices.T)
        dst = tuple(target_indices.T)
        for name, field in self.fields.items():
            if name != "evaluated":
                field[dst] = previous.fields[name][src]
                field.flush()
        self.fields["evaluated"][dst] = True
        self.fields["evaluated"].flush()
        return len(source_indices)

    def classes(self) -> np.ndarray:
        # 0 pending, 1 unresolved, 2 non-positive margin, 3 positive margin.
        done = self.fields["evaluated"]
        delta = self.fields["delta"]
        resolved = done & np.isfinite(delta) & (self.fields["pole_valid_count"] == self.samples)
        classes = np.zeros(self.shape, dtype=np.uint8)
        classes[done & ~resolved] = 1
        classes[resolved & (delta <= 0)] = 2
        classes[resolved & (delta > 0)] = 3
        return classes

    def counts(self) -> Mapping[str, int]:
        classes = self.classes()
        return {"total": int(classes.size), "evaluated": int(np.count_nonzero(classes)),
                "pending": int(np.count_nonzero(classes == 0)),
                "unresolved": int(np.count_nonzero(classes == 1)),
                "non_positive_margin": int(np.count_nonzero(classes == 2)),
                "positive_margin": int(np.count_nonzero(classes == 3))}

    def export(self, complete: bool) -> None:
        # Rectilinear VTK is required because the exact recorded anchors make
        # the otherwise uniform grid non-uniform.
        extent = " ".join(f"0 {n - 1}" for n in self.shape)
        root = ET.Element("VTKFile", type="RectilinearGrid", version="0.1", byte_order="LittleEndian")
        grid = ET.SubElement(root, "RectilinearGrid", WholeExtent=extent)
        piece = ET.SubElement(grid, "Piece", Extent=extent)
        data = ET.SubElement(piece, "PointData", Scalars="classification")
        fields = {"classification": self.classes(), **{k: v for k, v in self.fields.items() if k != "spectral_radius"}}
        for name, field in fields.items():
            vtk_type = "Float64" if np.issubdtype(field.dtype, np.floating) else "Int32"
            item = ET.SubElement(data, "DataArray", type=vtk_type, Name=name, format="ascii")
            # VTK X/P coordinate varies fastest; numpy's default C order would
            # silently swap the P and D directions.
            values = np.asarray(field).ravel(order="F")
            item.text = " ".join(format(float(v), ".17g") if vtk_type == "Float64" else str(int(v)) for v in values)
        coords = ET.SubElement(piece, "Coordinates")
        for name, axis in zip(("P", "I", "D"), self.axes):
            item = ET.SubElement(coords, "DataArray", type="Float64", Name=name, format="ascii")
            item.text = " ".join(format(float(v), ".17g") for v in axis)
        temporary = self.directory / "volume.vtr.tmp"
        ET.ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)
        os.replace(temporary, self.directory / "volume.vtr")
        if complete:
            temporary = self.directory / "volume.tmp.npz"
            np.savez_compressed(temporary, p=self.axes[0], i=self.axes[1], d=self.axes[2],
                                classification=self.classes(), **self.fields)
            os.replace(temporary, self.directory / "volume.npz")


def plot_volume(volume: Volume, group: str, case: str, reference: Mapping[str, Any]) -> None:
    classes = volume.classes()
    figure = go.Figure()
    colors = {1: ("Unresolved", "#909090"), 2: ("Margin <= 0", "#d62728"), 3: ("Margin > 0", "#169b46")}
    static = plt.figure(figsize=(8, 6))
    axis = static.add_subplot(111, projection="3d")
    # Plot sampled nodes only.  Filling/interpolating cells would assert a
    # continuous stability region that this calculation does not establish.
    for code in (2, 1, 3):
        indices = np.argwhere(classes == code)
        if not len(indices):
            continue
        xyz = np.column_stack([volume.axes[j][indices[:, j]] for j in range(3)])
        delta = np.asarray(volume.fields["delta"][tuple(indices.T)])
        valid = volume.fields["pole_valid_count"][tuple(indices.T)]
        label, color = colors[code]
        figure.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], name=label, mode="markers",
                                     marker=dict(size=2.5, color=color, opacity=0.35 if code != 3 else 0.85),
                                     customdata=np.column_stack((delta, valid)),
                                     hovertemplate="P=%{x}<br>I=%{y}<br>D=%{z}<br>delta=%{customdata[0]:.7e}<br>valid plants=%{customdata[1]}<extra>%{fullData.name}</extra>"))
        axis.scatter(*xyz.T, c=color, s=8, alpha=0.22 if code != 3 else 0.85, label=label)
    group_index = PID_GROUPS.index(group)
    xyz = np.asarray(reference["success"]["gains"])[group_index]
    delta = reference["success"]["delta"]
    label = f"Recorded successful flight; model delta={delta:.3e}" if delta is not None else "Recorded successful flight; model unresolved"
    figure.add_trace(go.Scatter3d(x=[xyz[0]], y=[xyz[1]], z=[xyz[2]], mode="markers", name=label,
                                 marker=dict(size=8, color="#2070c0", symbol="diamond")))
    axis.scatter(*xyz, c="#2070c0", marker="D", s=80, label="Successful flight (exact controller)")
    title = f"{case} posterior (N={volume.samples}) / {group} PID / other 9 gains fixed to success"
    counts = volume.counts()
    subtitle = f"Exact nodes {counts['evaluated']}/{counts['total']}; pending nodes omitted; no cell-interior guarantee"
    figure.update_layout(title=f"{title}<br><sup>{subtitle}</sup>",
                         scene=dict(xaxis_title="P gain", yaxis_title="I gain", zaxis_title="D gain",
                                    xaxis=dict(range=[volume.axes[0][0], volume.axes[0][-1]]),
                                    yaxis=dict(range=[volume.axes[1][0], volume.axes[1][-1]]),
                                    zaxis=dict(range=[volume.axes[2][0], volume.axes[2][-1]])),
                         legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, b=0))
    # Standalone HTML: no network or CDN required when opened later.
    temporary = volume.directory / "pid_3d.tmp.html"
    figure.write_html(str(temporary), include_plotlyjs=True)
    os.replace(temporary, volume.directory / "pid_3d.html")
    axis.set(xlabel="P gain", ylabel="I gain", zlabel="D gain",
             xlim=(volume.axes[0][0], volume.axes[0][-1]),
             ylim=(volume.axes[1][0], volume.axes[1][-1]),
             zlim=(volume.axes[2][0], volume.axes[2][-1]))
    axis.set_title(title + "\n" + subtitle, fontsize=9)
    axis.legend(fontsize=7)
    static.tight_layout()
    temporary = volume.directory / "pid_3d.tmp.png"
    static.savefig(temporary, dpi=150)
    plt.close(static)
    os.replace(temporary, volume.directory / "pid_3d.png")


def reference_payload(row: MarginEvaluation) -> Mapping[str, Any]:
    return json_sanitize({"gains": row.gains, "delta": row.delta,
                          "worst_spectral_radius": row.worst_spectral_radius,
                          "worst_sample_index": row.worst_sample_index,
                          "pole_valid_count": np.count_nonzero(row.pole_valid_mask),
                          "safe_on_fixed_sample_set": row.safe_on_sample_set,
                          "spectral_radius": row.spectral_radius})


def write_index(output: Path, state: Mapping[str, Any]) -> None:
    rows = []
    for group in PID_GROUPS:
        for directory in sorted((output / group).glob("grid_*"), key=lambda p: int(p.name.split("_")[-1])):
            if not (directory / "pid_3d.html").exists():
                continue
            status = json.loads((directory / "status.json").read_text())
            relative = directory.relative_to(output).as_posix()
            links = f'<a href="{relative}/pid_3d.html">Rotate 3D</a> · <a href="{relative}/pid_3d.png">PNG</a> · <a href="{relative}/volume.vtr">VTK</a>'
            if (directory / "volume.npz").exists():
                links += f' · <a href="{relative}/volume.npz">NumPy</a>'
            rows.append(f"<tr><td>{group}</td><td>{directory.name}</td><td>{status['evaluated']}/{status['total']}</td><td>{status['positive_margin']}</td><td>{status['non_positive_margin']}</td><td>{status['unresolved']}</td><td>{links}</td></tr>")
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Exact PID 3D volumes</title><style>body{font:16px sans-serif;margin:2em}table{border-collapse:collapse}th,td{padding:.6em;border:1px solid #ddd;text-align:left}a{color:#2070c0}</style>'
    page += f"<h1>Exact PID 3D volumes</h1><p>{html.escape(str(state['case']))} posterior. Status: {html.escape(str(state['status']))}. Updated {html.escape(str(state['updated_utc']))}.</p>"
    page += '<p>P/I/D axes; other nine gains fixed to the recorded success controller. Green: positive empirical pole margin. Red: non-positive margin. Gray: unresolved. Pending nodes are omitted. Flight success and model classification are separate. These are evaluated nodes, without a cell-interior stability guarantee.</p><p><a href="status.json">Progress</a> · <a href="manifest.json">Calculation metadata</a> · <a href="references.json">Exact recorded-controller evaluations</a></p>'
    page += '<table><tr><th>Group</th><th>Grid</th><th>Evaluated nodes</th><th>Green</th><th>Red</th><th>Unresolved</th><th>Files</th></tr>' + "".join(rows) + "</table></html>"
    temporary = output / "index.html.tmp"
    temporary.write_text(page)
    os.replace(temporary, output / "index.html")


def input_fingerprint(arguments: argparse.Namespace, estimate_path: Path,
                      success_path: Path, model_path: Path) -> tuple[str, Mapping[str, str]]:
    paths = {estimate_path, success_path, model_path}
    paths.update(HERE.glob("*.py"))
    paths.update(HERE.parent.glob("*.py"))
    paths.update((HERE.parent.parent / "src").rglob("*.py"))
    paths.add(HERE.parent / "single_bag_savgol_reports.py")
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    configuration = {"input_hashes": hashes, "covariance": arguments.covariance,
                     "samples": arguments.samples, "seed": arguments.seed,
                     "grid_sizes": arguments.grid_sizes, "groups": arguments.groups,
                     "context": "success", "schema": SCHEMA, "runtime_versions": RUNTIME_VERSIONS}
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
    success_path = Path(arguments.success_json).expanduser().resolve()
    estimate = load_estimate_json(estimate_path)
    success = load_estimate_json(success_path)
    gains = _recorded_gain_matrix(estimate)
    base_gains = _recorded_gain_matrix(success)
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
        axes = axes_for_group(sizes[-1], GAIN_CAPS[j], [base_gains[j], gains[j]])
        totals[group] = int(np.prod([len(a) for a in axes]))
    manifest = {"schema": SCHEMA, "fingerprint": fingerprint, "source_commit": source_commit(),
                "runtime_versions": RUNTIME_VERSIONS,
                "input_sha256": hashes, "case": case, "arguments": vars(arguments),
                "covariance": arguments.covariance, "plant_samples": arguments.samples,
                "sampling": sampling, "controller_dt_seconds": controller_dt,
                "full_success_gains": base_gains, "full_recorded_case_gains": gains,
                "groups": list(PID_GROUPS), "gain_names": ["P", "I", "D"],
                "exploration_caps": GAIN_CAPS, "final_node_counts": totals,
                "criterion": "delta = 1 - max_i spectral_radius(exact_26_state_sampled_closed_loop_i)",
                "semantics": ["Only one P/I/D triple varies; the other nine gains equal the success controller.",
                              "The successful flight is an exact evaluated node in each volume, not a projection.",
                              "Recorded failure triples with success's other nine gains are hybrid controllers; no failure-flight markers are projected.",
                              "Positive margin is empirical on the fixed sample set, not a flight guarantee.",
                              "Classes: 0 pending, 1 unresolved, 2 delta <= 0, 3 delta > 0.",
                              "Grid-node data; no interpolation or certification of entire voxel cells.",
                              "Exploration caps reuse term clamp numbers; they are not enforced PID gain limits."]}
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
    evaluator = BatchedForwardMarginEvaluator(plants=plants, vehicle_model=model,
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
            rows = evaluator.evaluate_many([(gains / GAIN_CAPS).ravel(), (base_gains / GAIN_CAPS).ravel()], progress=False)
            references = {"recorded_case": reference_payload(rows[0]), "success": reference_payload(rows[1])}
            temporary = output / "reference_trims.tmp.npz"
            np.savez_compressed(temporary, success=rows[1].trim_vectors)
            os.replace(temporary, output / "reference_trims.npz")
            atomic_json(references_path, references)
        warm = np.load(output / "reference_trims.npz")["success"]
        print(f"Full success controller exact delta={references['success']['delta']}", flush=True)
        for size in sizes:
            for group in arguments.groups:
                j = PID_GROUPS.index(group)
                axes = axes_for_group(size, GAIN_CAPS[j], [base_gains[j], gains[j]])
                current = Volume(output / group / f"grid_{size}", axes, arguments.samples, fingerprint, group, base_gains)
                if group in previous:
                    current.inherit(previous[group])
                pending = np.argwhere(~current.fields["evaluated"])
                # Process nodes near the success controller first; this is only
                # ordering and does not change samples, nodes, or acceptance.
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
                    print(f"{group} grid={size} {counts['evaluated']}/{counts['total']} | green={counts['positive_margin']} red={counts['non_positive_margin']} unresolved={counts['unresolved']} | batch {time.monotonic() - tick:.1f}s", flush=True)
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
    selected.add_argument("--success-json", default=str(HERE / "outputs/success/estimate.json"))
    selected.add_argument("--output", default=str(HERE / "outputs/failure1/pid_gain_voxels_success_context"))
    selected.add_argument("--covariance", choices=COVARIANCE_NAMES, default="conservative_fusion")
    selected.add_argument("--samples", type=int, default=128)
    selected.add_argument("--seed", type=int, default=0)
    selected.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    selected.add_argument("--batch-size", type=int, default=16)
    selected.add_argument("--grid-sizes", nargs="+", type=int, default=[3, 5, 9, 17, 33])
    selected.add_argument("--groups", nargs="+", choices=PID_GROUPS, default=["roll_pitch", "xy", "z", "yaw"])
    selected.add_argument("--preview-seconds", type=float, default=120.0)
    return selected


if __name__ == "__main__":
    arguments = parser().parse_args()
    output_path = Path(arguments.output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another PID voxel job is already writing this output directory.")
        analyze(arguments)
