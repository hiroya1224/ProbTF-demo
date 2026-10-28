#!/usr/bin/env python3
"""Exact P/I/D volumes with a fixed, explicit nine-gain controller context.

Every grid node is evaluated on the same ordered posterior plant samples.
No gain-affine surrogate or max projection is used. Colors follow the historical
full-sample stability definition. Values at nodes do not certify entire cells.
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

HERE = Path(os.environ.get('PID_VOXEL_CODE_DIR', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent))
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
from pid_voxel_domains import axes_for_box, load_configuration
from pid_voxel_survival import sample_stability

SCHEMA = "grape-param-estim/exact-pid-gain-volume/v2-unconditional-failure-context"
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


def axes_for_group(size: int, bounds: np.ndarray, anchors: Sequence[np.ndarray],
                   minimum_relative_clearance: float = .125) -> tuple[np.ndarray, ...]:
    """Include recorded triples exactly, rather than putting them on a nearby node."""
    return axes_for_box(size, bounds, anchors, minimum_relative_clearance)


class ExpandedGainEvaluator(BatchedForwardMarginEvaluator):
    @staticmethod
    def _key(q: Sequence[float]) -> tuple[float, ...]:
        # GAIN_CAPS supplies coordinate units, not a physical gain constraint.
        selected = np.asarray(q, dtype=float)
        if selected.shape != (12,) or np.any(~np.isfinite(selected)) or np.any(selected < 0):
            raise ValueError('normalized PID coordinates must be finite nonnegative length 12')
        return tuple(float(value) for value in selected)


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
    stability = sample_stability(volume.fields['spectral_radius'],
                                 volume.fields['evaluated'], .95)
    classes = stability.alpha_classification
    np.savez_compressed(volume.directory/'stability_fraction.npz',
                        p=volume.axes[0], i=volume.axes[1], d=volume.axes[2],
                        stable_fraction=stability.stable_fraction,
                        stable_count=stability.stable_count,
                        valid_count=stability.valid_count,
                        unresolved_count=stability.unresolved_count,
                        sample_count=np.array(stability.sample_count),
                        classification=classes, alpha=np.array(.95))
    atomic_json(volume.directory/'stability_fraction.json', {
        'sample_count':stability.sample_count,
        'criterion':'finite rho < 1 on all fixed samples; unresolved stays in denominator',
        'at_or_above_alpha':int(np.count_nonzero(classes==3)),
        'below_alpha':int(np.count_nonzero(classes==2)),
        'pending':int(np.count_nonzero(classes==0)),
        'with_unresolved':int(np.count_nonzero(stability.unresolved_count>0)),
        'alpha':.95})
    figure = go.Figure()
    colors = {2: ("Local stable fraction < 95%", "#d62728"), 3: ("Local stable fraction >= 95%", "#169b46")}
    static = plt.figure(figsize=(8, 6))
    axis = static.add_subplot(111, projection="3d")
    # Plot sampled nodes only.  Filling/interpolating cells would assert a
    # continuous stability region that this calculation does not establish.
    for code in (2, 3):
        indices = np.argwhere(classes == code)
        if not len(indices):
            continue
        xyz = np.column_stack([volume.axes[j][indices[:, j]] for j in range(3)])
        delta = np.asarray(volume.fields["delta"][tuple(indices.T)])
        valid = volume.fields["pole_valid_count"][tuple(indices.T)]
        survival = stability.stable_fraction[tuple(indices.T)]
        label, color = colors[code]
        figure.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], name=label, mode="markers",
                                     marker=dict(size=2.5, color=survival, coloraxis='coloraxis', opacity=0.35 if code != 3 else 0.85),
                                     customdata=np.column_stack((delta, valid, survival)),
                                     hovertemplate="P=%{x}<br>I=%{y}<br>D=%{z}<br>Local stable fraction=%{customdata[2]:.2%}<br>absolute delta (diagnostic)=%{customdata[0]:.7e}<br>valid plants=%{customdata[1]}<extra>%{fullData.name}</extra>"))
        axis.scatter(*xyz.T, c=survival, cmap='RdYlGn', vmin=0, vmax=1, s=8, alpha=0.22 if code != 3 else 0.85, label=label)
    group_index = PID_GROUPS.index(group)
    failure_xyz = np.asarray(reference['recorded_case']['gains'])[group_index]
    figure.add_trace(go.Scatter3d(x=[failure_xyz[0]],y=[failure_xyz[1]],z=[failure_xyz[2]],
                                 mode='markers+text',name=f'{case} recorded group gains',text=[case],
                                 marker=dict(size=9,color='#9c27b0',symbol='x')))
    axis.scatter(*failure_xyz,c='#9c27b0',marker='x',s=80,label=f'{case} recorded group gains')
    cmap = matplotlib.colormaps['RdYlGn']
    scale = [[float(v),matplotlib.colors.to_hex(cmap(float(v)))] for v in np.linspace(0,1,256)]
    title = f"{case} posterior (N={volume.samples}) / {group} PID / other 9 gains fixed to recorded failure"
    counts = volume.counts()
    subtitle = f"Exact nodes {counts['evaluated']}/{counts['total']}; pending nodes omitted; no cell-interior guarantee"
    figure.update_layout(title=f"{title}<br><sup>{subtitle}</sup>",
                         coloraxis=dict(colorscale=scale,cmin=0,cmax=1,colorbar=dict(title='Local stable fraction',tickvals=[0,.5,.95,1])),
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
                          "sample_count": len(row.spectral_radius),
                          "stable_count": np.count_nonzero(np.isfinite(row.spectral_radius) & (np.asarray(row.spectral_radius) < 1)),
                          "stable_fraction": np.count_nonzero(np.isfinite(row.spectral_radius) & (np.asarray(row.spectral_radius) < 1)) / len(row.spectral_radius),
                          "unresolved_count": np.count_nonzero(~np.isfinite(row.spectral_radius)),
                          "spectral_radius": row.spectral_radius})


def write_index(output: Path, state: Mapping[str, Any]) -> None:
    rows = []
    for group in PID_GROUPS:
        for directory in sorted((output / group).glob("grid_*"), key=lambda p: int(p.name.split("_")[-1])):
            if not (directory / "pid_3d.html").exists():
                continue
            status = json.loads((directory / "status.json").read_text())
            survival = json.loads((directory/'stability_fraction.json').read_text())
            relative = directory.relative_to(output).as_posix()
            links = f'<a href="{relative}/pid_3d.html">Rotate 3D</a> · <a href="{relative}/pid_3d.png">PNG</a> · <a href="{relative}/volume.vtr">VTK</a>'
            if (directory / "volume.npz").exists():
                links += f' · <a href="{relative}/volume.npz">NumPy</a>'
            links += f' · <a href="{relative}/stability_fraction.npz">Conditional-survival data</a>'
            rows.append(f"<tr><td>{group}</td><td>{directory.name}</td><td>{status['evaluated']}/{status['total']}</td><td>{survival['at_or_above_alpha']}</td><td>{survival['below_alpha']}</td><td>{survival['with_unresolved']}</td><td>{links}</td></tr>")
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Exact PID 3D volumes</title><style>body{font:16px sans-serif;margin:2em}table{border-collapse:collapse}th,td{padding:.6em;border:1px solid #ddd;text-align:left}a{color:#2070c0}</style>'
    page += f"<h1>Exact PID 3D volumes</h1><p>{html.escape(str(state['case']))} posterior. Status: {html.escape(str(state['status']))}. Updated {html.escape(str(state['updated_utc']))}.</p>"
    page += '<p>P/I/D axes; other nine gains fixed to this recorded failure controller. Colors show finite rho &lt; 1 count divided by all N fixed plant samples, with RdYlGn on [0,1] and an explicit 95% classification. Non-finite radii remain in the denominator and are counted separately. Pending nodes are omitted. No arbitrary combination of points from different panels is certified stable.</p><p><a href="status.json">Progress</a> · <a href="manifest.json">Calculation metadata and domain bounds</a> · <a href="references.json">Recorded controllers</a></p>'
    page += '<table><tr><th>Group</th><th>Grid</th><th>Evaluated nodes</th><th>Survival >= 95%</th><th>Survival < 95%</th><th>With unresolved samples</th><th>Files</th></tr>' + "".join(rows) + "</table></html>"
    temporary = output / "index.html.tmp"
    temporary.write_text(page)
    os.replace(temporary, output / "index.html")


def input_fingerprint(arguments: argparse.Namespace, estimate_path: Path,
                      success_path: Path, model_path: Path) -> tuple[str, Mapping[str, str]]:
    paths = {estimate_path, model_path}
    paths.update(HERE.glob("*.py"))
    paths.update(HERE.parent.glob("*.py"))
    paths.update((HERE.parent.parent / "src").rglob("*.py"))
    paths.add(HERE.parent / "single_bag_savgol_reports.py")
    paths.add(Path(__file__).resolve())
    paths.add(Path(__file__).with_name('pid_voxel_domains.py').resolve())
    paths.add(Path(__file__).with_name('pid_voxel_survival.py').resolve())
    paths.add(Path(arguments.domain_config).expanduser().resolve())
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    configuration = {"input_hashes": hashes, "covariance": arguments.covariance,
                     "samples": arguments.samples, "seed": arguments.seed,
                     "grid_sizes": arguments.grid_sizes, "groups": arguments.groups,
                     "context": "recorded_failure", "schema": SCHEMA, "runtime_versions": RUNTIME_VERSIONS}
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
                "fixed_controller_gains": base_gains, "full_recorded_case_gains": gains,
                "groups": list(PID_GROUPS), "gain_names": ["P", "I", "D"],
                "exploration_bounds": domain_report,
                "gain_coordinate_scales": GAIN_CAPS, "final_node_counts": totals,
                "criterion": "stable fraction = count(finite rho < 1) / all N fixed plant samples; RdYlGn [0,1]; alpha=0.95",
                "absolute_margin_diagnostic": "delta = 1 - max_i spectral_radius(exact_26_state_sampled_closed_loop_i); raw Volume.classification and *_margin counts are diagnostics, not plot colors",
                "semantics": ["Only one P/I/D triple varies; the other nine gains equal this recorded failure controller.",
                              "Successful-flight gains, if explicitly requested, are evaluated only after every map is complete as a separate full-12-gain controller.",
                              "The recorded failure marker is the actual full-12-gain failure controller in every panel. Success markers are not projected onto these failure-context slices.",
                              "Every original plant sample remains in the denominator; non-finite radii do not count as stable and are counted separately.",
                              "Primary stability_fraction.npz classification: 0 pending, 2 below 95%, 3 at or above 95%; 95% does not mean every sample is stable.",
                              "Grid-node data; no interpolation or certification of entire voxel cells.",
                              "Exploration bounds are independent of the GAIN_CAPS normalization scales.",
                              "The sole recorded failure gain triple is an interior anchor; the box is the declared fixed scale range about that failure triple only."]}
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
    evaluator = ExpandedGainEvaluator(plants=plants, vehicle_model=model,
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
    selected.add_argument("--output", default=str(HERE / "outputs/failure1/pid_gain_voxels_interior_failure_context"))
    selected.add_argument("--domain-config", default=str(Path(__file__).with_name('pid_voxel_domains.json')))
    selected.add_argument("--validate-domain-only", action='store_true')
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
