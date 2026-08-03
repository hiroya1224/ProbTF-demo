"""Recorded PID snapshot selection and proposal-only YAML rendering."""

from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from grape_param_estim.controller import ControllerConfig, PIDConfig


PID_GROUPS = ("xy", "z", "roll_pitch", "yaw")
PID_GAIN_NAMES = ("p_gain", "i_gain", "d_gain")
CONTROLLER_GAIN_TOPIC_TEMPLATE = (
    "/gimbalrotor/controller/{}/parameter_updates"
)


class BaselineSelectionRequired(ValueError):
    """Raised when recorded bags disagree and no baseline bag was selected."""

    def __init__(self, bag_ids: Sequence[str]):
        self.bag_ids = tuple(str(value) for value in bag_ids)
        super().__init__(
            "controller snapshots differ; select one baseline bag: {}".format(
                ", ".join(self.bag_ids)
            )
        )


@dataclass(frozen=True)
class ControllerSnapshotProvenance:
    bag_id: str
    topics: Tuple[str, ...]
    record_times: np.ndarray
    source_kinds: Tuple[str, ...]

    def __post_init__(self) -> None:
        bag_id = str(self.bag_id)
        topics = tuple(str(value) for value in self.topics)
        times = np.asarray(self.record_times, dtype=float)
        source_kinds = tuple(str(value) for value in self.source_kinds)
        if (
            not bag_id
            or len(topics) != len(PID_GROUPS)
            or times.shape != (len(PID_GROUPS),)
            or np.any(~np.isfinite(times))
            or len(source_kinds) != len(PID_GROUPS)
            or any(not value for value in topics)
            or any(not value for value in source_kinds)
        ):
            raise ValueError("controller snapshot provenance is invalid")
        object.__setattr__(self, "bag_id", bag_id)
        object.__setattr__(self, "topics", topics)
        object.__setattr__(self, "record_times", times.copy())
        object.__setattr__(self, "source_kinds", source_kinds)


@dataclass(frozen=True)
class PidGainConfiguration:
    """Exact P/I/D values in canonical group and gain order."""

    values: np.ndarray
    provenance: Optional[ControllerSnapshotProvenance] = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if (
            values.shape != (len(PID_GROUPS), len(PID_GAIN_NAMES))
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise ValueError("PID gain configuration must be a finite 4x3 array")
        if (
            self.provenance is not None
            and not isinstance(self.provenance, ControllerSnapshotProvenance)
        ):
            raise TypeError("PID gain provenance has the wrong type")
        object.__setattr__(self, "values", values.copy())

    def group(self, group: str) -> np.ndarray:
        try:
            index = PID_GROUPS.index(str(group))
        except ValueError as error:
            raise KeyError("unknown PID group: {}".format(group)) from error
        return self.values[index].copy()

    def as_nested_mapping(self) -> Mapping[str, Mapping[str, float]]:
        return {
            group: {
                gain: float(self.values[group_index, gain_index])
                for gain_index, gain in enumerate(PID_GAIN_NAMES)
            }
            for group_index, group in enumerate(PID_GROUPS)
        }


@dataclass(frozen=True)
class PidGainComparison:
    current: PidGainConfiguration
    proposed: PidGainConfiguration
    difference: np.ndarray
    ratio: np.ndarray
    ratio_configured: np.ndarray

    @classmethod
    def from_configurations(
        cls, current: PidGainConfiguration, proposed: PidGainConfiguration
    ) -> "PidGainComparison":
        difference = proposed.values - current.values
        configured = current.values != 0.0
        ratio = np.full_like(current.values, np.nan)
        np.divide(
            proposed.values,
            current.values,
            out=ratio,
            where=configured,
        )
        return cls(current, proposed, difference, ratio, configured)

    def __post_init__(self) -> None:
        shape = (len(PID_GROUPS), len(PID_GAIN_NAMES))
        difference = np.asarray(self.difference, dtype=float)
        ratio = np.asarray(self.ratio, dtype=float)
        configured = np.asarray(self.ratio_configured, dtype=bool)
        if (
            not isinstance(self.current, PidGainConfiguration)
            or not isinstance(self.proposed, PidGainConfiguration)
            or difference.shape != shape
            or ratio.shape != shape
            or configured.shape != shape
            or np.any(~np.isfinite(difference))
            or np.any(~np.isfinite(ratio[configured]))
            or np.any(~np.isnan(ratio[~configured]))
        ):
            raise ValueError("PID gain comparison arrays are invalid")
        object.__setattr__(self, "difference", difference.copy())
        object.__setattr__(self, "ratio", ratio.copy())
        object.__setattr__(self, "ratio_configured", configured.copy())


def configuration_from_controller_snapshot(
    snapshot, bag_id: str
) -> PidGainConfiguration:
    """Adapt a complete bag-recorded snapshot without consulting YAML values."""

    groups = tuple(str(value) for value in snapshot.groups)
    if groups != PID_GROUPS:
        raise ValueError("controller snapshot is missing a canonical PID group")
    provenance = ControllerSnapshotProvenance(
        bag_id=str(bag_id),
        topics=tuple(
            CONTROLLER_GAIN_TOPIC_TEMPLATE.format(group)
            for group in PID_GROUPS
        ),
        record_times=np.asarray(snapshot.record_times, dtype=float),
        source_kinds=tuple(snapshot.source_kinds),
    )
    return PidGainConfiguration(
        np.asarray(snapshot.gains, dtype=float), provenance
    )


def select_baseline_pid_configuration(
    configurations: Mapping[str, PidGainConfiguration],
    baseline_bag_id: Optional[str] = None,
) -> PidGainConfiguration:
    """Select one exact recorded snapshot; differing snapshots are never averaged."""

    values = {
        str(bag_id): configuration
        for bag_id, configuration in configurations.items()
    }
    if not values:
        raise ValueError("at least one recorded controller snapshot is required")
    if any(
        not isinstance(configuration, PidGainConfiguration)
        for configuration in values.values()
    ):
        raise TypeError("baseline inputs must be PID gain configurations")
    selected_id = None if baseline_bag_id is None else str(baseline_bag_id)
    if selected_id is not None:
        if selected_id not in values:
            raise ValueError("baseline bag is not one of the selected bags")
        return values[selected_id]
    ordered = sorted(values)
    first = values[ordered[0]]
    if all(
        np.array_equal(first.values, values[bag_id].values)
        for bag_id in ordered[1:]
    ):
        return first
    raise BaselineSelectionRequired(ordered)


def apply_pid_gain_configuration(
    controller: ControllerConfig, gains: PidGainConfiguration
) -> ControllerConfig:
    """Replace only exact gains while preserving every controller limit/flag."""

    if not isinstance(controller, ControllerConfig):
        raise TypeError("controller must be a ControllerConfig")
    if not isinstance(gains, PidGainConfiguration):
        raise TypeError("gains must be a PidGainConfiguration")
    axis_groups = (0, 0, 1, 2, 2, 3)
    pid = []
    for original, group_index in zip(controller.pid, axis_groups):
        if not isinstance(original, PIDConfig):
            raise TypeError("controller PID entry has the wrong type")
        values = gains.values[group_index]
        pid.append(
            replace(
                original,
                p_gain=float(values[0]),
                i_gain=float(values[1]),
                d_gain=float(values[2]),
            )
        )
    return replace(controller, pid=tuple(pid))


def _format_number(value: float) -> str:
    return "{:.17g}".format(float(value))


def render_proposed_pid_yaml(configuration: PidGainConfiguration) -> str:
    """Render only the four writable PID groups, without a parent key."""

    if not isinstance(configuration, PidGainConfiguration):
        raise TypeError("configuration must be a PidGainConfiguration")
    lines = []
    for group_index, group in enumerate(PID_GROUPS):
        lines.append("{}:".format(group))
        for gain_index, gain in enumerate(PID_GAIN_NAMES):
            lines.append(
                "  {}: {}".format(
                    gain,
                    _format_number(configuration.values[group_index, gain_index]),
                )
            )
    return "\n".join(lines) + "\n"


def render_pid_diff_yaml(comparison: PidGainComparison) -> str:
    """Render current/proposed/difference/ratio evidence for changed gains."""

    if not isinstance(comparison, PidGainComparison):
        raise TypeError("comparison must be a PidGainComparison")
    lines = []
    for group_index, group in enumerate(PID_GROUPS):
        changed = np.flatnonzero(comparison.difference[group_index] != 0.0)
        if changed.size == 0:
            continue
        lines.append("{}:".format(group))
        for gain_index in changed:
            gain = PID_GAIN_NAMES[int(gain_index)]
            ratio = (
                _format_number(comparison.ratio[group_index, gain_index])
                if comparison.ratio_configured[group_index, gain_index]
                else "not_configured"
            )
            lines.extend(
                (
                    "  {}:".format(gain),
                    "    current: {}".format(
                        _format_number(
                            comparison.current.values[group_index, gain_index]
                        )
                    ),
                    "    proposed: {}".format(
                        _format_number(
                            comparison.proposed.values[group_index, gain_index]
                        )
                    ),
                    "    difference: {}".format(
                        _format_number(
                            comparison.difference[group_index, gain_index]
                        )
                    ),
                    "    ratio: {}".format(ratio),
                )
            )
    return ("{}\n" if not lines else "\n".join(lines) + "\n")


def validate_controller_yaml_key_contract(path: str) -> Tuple[str, ...]:
    """Validate key availability only; numeric YAML values are never returned."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    in_controller = False
    current_group = None
    found = set()
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if re.match(r"^controller\s*:\s*$", line):
            in_controller = True
            current_group = None
            continue
        if not in_controller:
            continue
        group_match = re.match(r"^  ([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$", line)
        if group_match:
            current_group = group_match.group(1)
            continue
        if re.match(r"^[^ ]", line):
            in_controller = False
            current_group = None
            continue
        gain_match = re.match(
            r"^    (p_gain|i_gain|d_gain)\s*:\s*.+$", line
        )
        if current_group in PID_GROUPS and gain_match:
            found.add("{}.{}".format(current_group, gain_match.group(1)))
    expected = tuple(
        "{}.{}".format(group, gain)
        for group in PID_GROUPS
        for gain in PID_GAIN_NAMES
    )
    missing = tuple(value for value in expected if value not in found)
    if missing:
        raise ValueError(
            "controller YAML is missing PID key paths: {}".format(
                ", ".join(missing)
            )
        )
    return expected


__all__ = [
    "BaselineSelectionRequired",
    "CONTROLLER_GAIN_TOPIC_TEMPLATE",
    "ControllerSnapshotProvenance",
    "PID_GAIN_NAMES",
    "PID_GROUPS",
    "PidGainComparison",
    "PidGainConfiguration",
    "apply_pid_gain_configuration",
    "configuration_from_controller_snapshot",
    "render_pid_diff_yaml",
    "render_proposed_pid_yaml",
    "select_baseline_pid_configuration",
    "validate_controller_yaml_key_contract",
]
