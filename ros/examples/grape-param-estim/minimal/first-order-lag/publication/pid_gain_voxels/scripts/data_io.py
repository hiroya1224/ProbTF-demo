"""Portable, exact node data and cell geometry; no numerical solver required."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ("failure1", "failure2")
GROUPS = ("roll_pitch", "yaw", "xy", "z")
SOURCE_GROUPS = ("xy", "z", "roll_pitch", "yaw")
LABELS = {"roll_pitch": "Roll / Pitch", "yaw": "Yaw", "xy": "XY", "z": "Z"}
GAIN_NAMES = ("P", "I", "D")


def midpoint_edges(axis):
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or len(axis) < 2 or not np.all(np.diff(axis) > 0):
        raise ValueError("expected a strictly increasing gain axis")
    return np.r_[axis[0], (axis[:-1] + axis[1:]) / 2, axis[-1]]


def unpack_samples(packed, count):
    return np.unpackbits(packed, axis=-1, count=int(count), bitorder="little").astype(bool)


def survival_from_bits(stable_bits, valid_bits, baseline_mask):
    baseline_mask = np.asarray(baseline_mask, dtype=bool)
    if not baseline_mask.any():
        raise ValueError("conditioning set is empty")
    stable = unpack_samples(stable_bits, len(baseline_mask))
    valid = unpack_samples(valid_bits, len(baseline_mask))
    if np.any(stable & ~valid):
        raise ValueError("stable sample must have a finite radius")
    breaks = np.count_nonzero(~stable[..., baseline_mask], axis=-1).astype(np.int32)
    unresolved = np.count_nonzero(~valid[..., baseline_mask], axis=-1).astype(np.int32)
    survival = 1.0 - breaks.astype(np.float64) / int(baseline_mask.sum())
    return survival, breaks, unresolved


@dataclass
class Grid:
    case: str
    group: str
    size: int
    arrays: dict
    references: dict
    metadata: dict

    @property
    def axes(self):
        return [self.arrays[key] for key in ("p", "i", "d")]

    @property
    def edges(self):
        return [midpoint_edges(axis) for axis in self.axes]

    @property
    def survival(self):
        return self.arrays["survival_fraction"]

    @property
    def baseline_count(self):
        return int(self.arrays["baseline_stable_mask"].sum())

    def classes(self, alpha=.95):
        """Historical archive classification; visualization uses survival directly."""
        if not 0 < alpha <= 1:
            raise ValueError("alpha must lie in (0, 1]")
        return np.where(self.survival >= alpha, 3, 2).astype(np.uint8)

    def gain(self, name):
        ref = "success" if name == "success" else "recorded_case"
        return np.asarray(self.references[ref]["gains"][SOURCE_GROUPS.index(self.group)], dtype=float)

    def index(self, gain):
        index = []
        for value, axis in zip(gain, self.axes):
            found = np.flatnonzero(axis == value)
            if len(found) != 1:
                raise ValueError("reference must be an exact computed grid node")
            index.append(int(found[0]))
        return tuple(index)

    def mark(self, name):
        gain = self.gain(name)
        index = self.index(gain)
        return {"gain": gain.tolist(), "survival": float(self.survival[index]),
                "breaks": int(self.arrays["caused_break_count"][index]),
                "unresolved": int(self.arrays["conditional_unresolved_count"][index])}


def load_grid(case, group, size=65, root=ROOT):
    root = Path(root)
    folder = root / "data" / case
    with np.load(folder / group / ("grid_%d.npz" % size), allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    references = json.loads((folder / "references.json").read_text())
    metadata = json.loads((folder / "export.json").read_text())
    return Grid(case, group, size, arrays, references, metadata)


def slice_at(grid, fixed_axis, fixed_index):
    """Rows are the second free axis, columns the first, as pcolormesh expects."""
    free = [k for k in range(3) if k != fixed_axis]
    values = np.take(grid.survival, fixed_index, axis=fixed_axis).T
    return free, values, [grid.edges[k] for k in free]


def marks_in_slice(grid, fixed_axis, fixed_index):
    value = grid.axes[fixed_axis][fixed_index]
    return [name for name in ("success", "recorded") if grid.gain(name)[fixed_axis] == value]


def exposed_faces(grid):
    """Exact quads and rates; remove only faces between equal-rate neighbors.

    Faces describe a piecewise-constant visualization of evaluated nodes, not a
    stability certificate for every gain in their volume.
    """
    values = grid.survival
    edges = grid.edges
    all_faces, all_rates = [], []
    for axis in range(3):
        free = [k for k in range(3) if k != axis]
        for side in (0, 1):
            visible = np.ones(values.shape, dtype=bool)
            dst, src = [slice(None)] * 3, [slice(None)] * 3
            dst[axis] = slice(1, None) if side == 0 else slice(None, -1)
            src[axis] = slice(None, -1) if side == 0 else slice(1, None)
            visible[tuple(dst)] = values[tuple(dst)] != values[tuple(src)]
            locations = np.argwhere(visible)
            faces = np.empty((len(locations), 4, 3), dtype=float)
            faces[:, :, axis] = edges[axis][locations[:, axis] + side, None]
            for corner, (a, b) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                faces[:, corner, free[0]] = edges[free[0]][locations[:, free[0]] + a]
                faces[:, corner, free[1]] = edges[free[1]][locations[:, free[1]] + b]
            all_faces.append(faces)
            all_rates.append(values[tuple(locations.T)])
    return np.concatenate(all_faces), np.concatenate(all_rates)
