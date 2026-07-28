"""Strict JSON inputs for production exact closed-loop assimilation.

The current rosbag set does not contain enough information to construct these
objects.  This module therefore only accepts explicit, content-addressed
bundles and never manufactures a controller snapshot or a zero controller
state from incomplete evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from grape_param_estim.alternative_backends import (
    ExactOracleConformanceFixture,
    ExactOracleConformanceReport,
    ExactOracleIdentity,
    ExactOracleProtocolError,
)
from grape_param_estim.data import (
    CONTROLLER_REPLAY_FIXTURE_SCHEMA,
    ControllerReplayFixture,
    EpisodeTimeGrids,
    EventGrid,
)
from grape_param_estim.episode import stable_hash

from .contracts import (
    ControllerCoreInput,
    ControllerCoreState,
    FrozenMapping,
    deep_thaw,
)
from .external_oracle import build_exact_replay_payload
from .snapshot import ControllerSnapshot


FIXTURE_BUNDLE_SCHEMA = "grape.controller-replay-fixture-bundle/v1"
SNAPSHOT_BUNDLE_SCHEMA = "grape.controller-snapshot-bundle/v1"
STATE_BUNDLE_SCHEMA = "grape.controller-state-bundle/v1"
EXACT_EPISODE_CONFORMANCE_EVIDENCE_SCHEMA = (
    "grape_exact_episode_conformance_evidence/v1"
)
EXACT_EPISODE_CONFORMANCE_BUNDLE_SCHEMA = (
    "grape_exact_episode_conformance_bundle/v1"
)
EXACT_EPISODE_REQUEST_BINDING_SCHEMA = (
    "grape_exact_episode_request_binding/v1"
)
EXACT_EPISODE_CONTROLLER_INPUT_SCHEMA = (
    "grape_exact_episode_controller_inputs/v1"
)
CONTROLLER_SNAPSHOT_SCHEMA = "grape.controller-snapshot/v2"
_GRID_FIELDS = (
    "controller_tick_grid",
    "plant_integration_grid",
    "observation_grid",
    "likelihood_grid",
    "report_grid",
)


def _json_object(path: Any, label: str) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "{} must be a readable UTF-8 JSON object".format(label)
        ) from exc
    if not isinstance(value, dict):
        raise TypeError("{} must contain one JSON object".format(label))
    return value


def _exact_keys(
    values: Mapping[str, Any],
    *,
    required: Sequence[str],
    label: str,
) -> None:
    expected = set(required)
    actual = set(values)
    if actual != expected:
        missing = tuple(sorted(expected - actual))
        unexpected = tuple(sorted(actual - expected))
        details = []
        if missing:
            details.append("missing {}".format(", ".join(missing)))
        if unexpected:
            details.append("unexpected {}".format(", ".join(unexpected)))
        raise ValueError(
            "{} has invalid fields ({})".format(label, "; ".join(details))
        )


def _sha256(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(
            "{} must be a lowercase SHA-256".format(label)
        )
    return digest


def _controller_input_payload(
    fixture: ControllerReplayFixture,
) -> Mapping[str, Any]:
    if not isinstance(fixture, ControllerReplayFixture):
        raise TypeError("fixture must be ControllerReplayFixture")
    return {
        "schema": EXACT_EPISODE_CONTROLLER_INPUT_SCHEMA,
        "episode_id": fixture.episode_id,
        "controller_inputs": [
            item.to_mapping() for item in fixture.controller_inputs
        ],
    }


def _episode_request_binding(
    fixture: ControllerReplayFixture,
    snapshot: ControllerSnapshot,
) -> Mapping[str, Any]:
    return {
        "schema": EXACT_EPISODE_REQUEST_BINDING_SCHEMA,
        "episode_id": fixture.episode_id,
        "source_bag_sha256": fixture.source_bag_sha256,
        "controller_replay_fixture_sha256": fixture.fixture_sha256,
        "controller_input_sha256": stable_hash(
            _controller_input_payload(fixture)
        ),
        "controller_snapshot_sha256": snapshot.snapshot_id,
    }


@dataclass(frozen=True)
class ExactEpisodeConformanceEvidence:
    """One episode's factual replay, bound to its actual inference inputs."""

    episode_id: str
    source_bag_sha256: str
    controller_replay_fixture_sha256: str
    controller_input_sha256: str
    controller_snapshot_sha256: str
    request_payload_sha256: str
    fixture_data_sha256: str
    fixture_provenance_sha256: str
    conformance_evidence_sha256: str
    initial_controller_state: ControllerCoreState
    request_payload: Mapping[str, Any]
    conformance_fixture: ExactOracleConformanceFixture
    conformance_report: ExactOracleConformanceReport
    content_sha256: str = ""
    schema: str = EXACT_EPISODE_CONFORMANCE_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        episode_id = str(self.episode_id).strip()
        if not episode_id:
            raise ValueError(
                "episode conformance evidence requires episode_id"
            )
        if self.schema != EXACT_EPISODE_CONFORMANCE_EVIDENCE_SCHEMA:
            raise ValueError(
                "unsupported exact episode conformance evidence schema"
            )
        if not isinstance(
            self.initial_controller_state, ControllerCoreState
        ):
            raise TypeError(
                "episode conformance requires typed initial controller state"
            )
        if not isinstance(self.request_payload, Mapping):
            raise TypeError(
                "episode conformance request_payload must be a mapping"
            )
        if not isinstance(
            self.conformance_fixture, ExactOracleConformanceFixture
        ):
            raise TypeError(
                "episode conformance requires its typed replay fixture"
            )
        report = self.conformance_report
        if (
            not isinstance(report, ExactOracleConformanceReport)
            or not report.content_is_valid()
            or report.passed is not True
            or report.status != "PASS"
            or not isinstance(report.identity, ExactOracleIdentity)
        ):
            raise ValueError(
                "episode conformance requires a passing typed report"
            )
        source_hash = _sha256(
            self.source_bag_sha256, "source_bag_sha256"
        )
        fixture_hash = _sha256(
            self.controller_replay_fixture_sha256,
            "controller_replay_fixture_sha256",
        )
        input_hash = _sha256(
            self.controller_input_sha256,
            "controller_input_sha256",
        )
        snapshot_hash = _sha256(
            self.controller_snapshot_sha256,
            "controller_snapshot_sha256",
        )
        request_hash = _sha256(
            self.request_payload_sha256,
            "request_payload_sha256",
        )
        data_hash = _sha256(
            self.fixture_data_sha256, "fixture_data_sha256"
        )
        provenance_hash = _sha256(
            self.fixture_provenance_sha256,
            "fixture_provenance_sha256",
        )
        evidence_hash = _sha256(
            self.conformance_evidence_sha256,
            "conformance_evidence_sha256",
        )
        request = FrozenMapping(self.request_payload)
        provenance = self.conformance_fixture.provenance
        binding = request.get("evidence_binding")
        expected_binding = {
            "schema": EXACT_EPISODE_REQUEST_BINDING_SCHEMA,
            "episode_id": episode_id,
            "source_bag_sha256": source_hash,
            "controller_replay_fixture_sha256": fixture_hash,
            "controller_input_sha256": input_hash,
            "controller_snapshot_sha256": snapshot_hash,
        }
        if (
            not isinstance(binding, Mapping)
            or deep_thaw(binding) != expected_binding
            or stable_hash(request) != request_hash
            or request_hash != report.request_payload_sha256
            or request_hash
            != provenance.fixture_input_payload_sha256
            or source_hash != provenance.source_bag_sha256
            or data_hash != provenance.fixture_data_sha256
            or provenance_hash != provenance.content_sha256
            or provenance_hash != report.fixture_content_sha256
            or evidence_hash != report.evidence_sha256
            or provenance.content_sha256
            != self.conformance_fixture.provenance.content_sha256
        ):
            raise ValueError(
                "episode conformance request/data/provenance/report "
                "hash linkage is invalid"
            )
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "source_bag_sha256", source_hash)
        object.__setattr__(
            self, "controller_replay_fixture_sha256", fixture_hash
        )
        object.__setattr__(self, "controller_input_sha256", input_hash)
        object.__setattr__(
            self, "controller_snapshot_sha256", snapshot_hash
        )
        object.__setattr__(
            self, "request_payload_sha256", request_hash
        )
        object.__setattr__(self, "fixture_data_sha256", data_hash)
        object.__setattr__(
            self, "fixture_provenance_sha256", provenance_hash
        )
        object.__setattr__(
            self, "conformance_evidence_sha256", evidence_hash
        )
        object.__setattr__(self, "request_payload", request)
        computed = stable_hash(self._content_payload())
        supplied = str(self.content_sha256).lower()
        if supplied and _sha256(
            supplied, "episode conformance content_sha256"
        ) != computed:
            raise ValueError(
                "episode conformance evidence content hash mismatch"
            )
        object.__setattr__(self, "content_sha256", computed)

    @classmethod
    def create(
        cls,
        *,
        fixture: ControllerReplayFixture,
        snapshot: ControllerSnapshot,
        initial_controller_state: ControllerCoreState,
        conformance_fixture: ExactOracleConformanceFixture,
        conformance_report: ExactOracleConformanceReport,
    ) -> "ExactEpisodeConformanceEvidence":
        binding = _episode_request_binding(fixture, snapshot)
        request = build_exact_replay_payload(
            snapshot,
            initial_controller_state,
            fixture.controller_inputs,
            evidence_binding=binding,
        )
        provenance = conformance_fixture.provenance
        return cls(
            episode_id=fixture.episode_id,
            source_bag_sha256=fixture.source_bag_sha256,
            controller_replay_fixture_sha256=fixture.fixture_sha256,
            controller_input_sha256=binding[
                "controller_input_sha256"
            ],
            controller_snapshot_sha256=snapshot.snapshot_id,
            request_payload_sha256=stable_hash(request),
            fixture_data_sha256=provenance.fixture_data_sha256,
            fixture_provenance_sha256=provenance.content_sha256,
            conformance_evidence_sha256=(
                conformance_report.evidence_sha256
            ),
            initial_controller_state=initial_controller_state,
            request_payload=request,
            conformance_fixture=conformance_fixture,
            conformance_report=conformance_report,
        )

    def _content_payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "episode_id": self.episode_id,
            "source_bag_sha256": self.source_bag_sha256,
            "controller_replay_fixture_sha256": (
                self.controller_replay_fixture_sha256
            ),
            "controller_input_sha256": self.controller_input_sha256,
            "controller_snapshot_sha256": (
                self.controller_snapshot_sha256
            ),
            "request_payload_sha256": self.request_payload_sha256,
            "fixture_data_sha256": self.fixture_data_sha256,
            "fixture_provenance_sha256": (
                self.fixture_provenance_sha256
            ),
            "conformance_evidence_sha256": (
                self.conformance_evidence_sha256
            ),
            "initial_controller_state": (
                self.initial_controller_state.to_mapping()
            ),
            "request_payload": deep_thaw(self.request_payload),
            "conformance_fixture": self.conformance_fixture.to_mapping(),
            "conformance_report": self.conformance_report.to_mapping(),
        }

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            **self._content_payload(),
            "content_sha256": self.content_sha256,
        }

    def binds(
        self,
        fixture: ControllerReplayFixture,
        snapshot: ControllerSnapshot,
    ) -> bool:
        try:
            if (
                not isinstance(fixture, ControllerReplayFixture)
                or not isinstance(snapshot, ControllerSnapshot)
                or fixture.episode_id != self.episode_id
                or fixture.source_bag_sha256
                != self.source_bag_sha256
                or fixture.fixture_sha256
                != self.controller_replay_fixture_sha256
                or snapshot.snapshot_id
                != self.controller_snapshot_sha256
                or stable_hash(_controller_input_payload(fixture))
                != self.controller_input_sha256
            ):
                return False
            canonical = build_exact_replay_payload(
                snapshot,
                self.initial_controller_state,
                fixture.controller_inputs,
                evidence_binding=_episode_request_binding(
                    fixture, snapshot
                ),
            )
            return bool(
                stable_hash(canonical) == self.request_payload_sha256
                and stable_hash(canonical)
                == stable_hash(self.request_payload)
                and stable_hash(self._content_payload())
                == self.content_sha256
            )
        except (AttributeError, TypeError, ValueError):
            return False

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "ExactEpisodeConformanceEvidence":
        if not isinstance(values, Mapping):
            raise TypeError(
                "exact episode conformance evidence must be a mapping"
            )
        _exact_keys(
            values,
            required=(
                "schema",
                "episode_id",
                "source_bag_sha256",
                "controller_replay_fixture_sha256",
                "controller_input_sha256",
                "controller_snapshot_sha256",
                "request_payload_sha256",
                "fixture_data_sha256",
                "fixture_provenance_sha256",
                "conformance_evidence_sha256",
                "initial_controller_state",
                "request_payload",
                "conformance_fixture",
                "conformance_report",
                "content_sha256",
            ),
            label="exact episode conformance evidence",
        )
        try:
            return cls(
                episode_id=values["episode_id"],
                source_bag_sha256=values["source_bag_sha256"],
                controller_replay_fixture_sha256=values[
                    "controller_replay_fixture_sha256"
                ],
                controller_input_sha256=values[
                    "controller_input_sha256"
                ],
                controller_snapshot_sha256=values[
                    "controller_snapshot_sha256"
                ],
                request_payload_sha256=values[
                    "request_payload_sha256"
                ],
                fixture_data_sha256=values["fixture_data_sha256"],
                fixture_provenance_sha256=values[
                    "fixture_provenance_sha256"
                ],
                conformance_evidence_sha256=values[
                    "conformance_evidence_sha256"
                ],
                initial_controller_state=ControllerCoreState.from_mapping(
                    values["initial_controller_state"]
                ),
                request_payload=values["request_payload"],
                conformance_fixture=(
                    ExactOracleConformanceFixture.from_mapping(
                        values["conformance_fixture"]
                    )
                ),
                conformance_report=(
                    ExactOracleConformanceReport.from_mapping(
                        values["conformance_report"]
                    )
                ),
                content_sha256=values["content_sha256"],
                schema=values["schema"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "exact episode conformance evidence is invalid"
            ) from exc


@dataclass(frozen=True)
class ExactEpisodeConformanceBundle:
    """All episode reports behind one identity-consistent aggregate gate."""

    episodes: Mapping[str, ExactEpisodeConformanceEvidence]
    content_sha256: str = ""
    schema: str = EXACT_EPISODE_CONFORMANCE_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXACT_EPISODE_CONFORMANCE_BUNDLE_SCHEMA:
            raise ValueError(
                "unsupported exact episode conformance bundle schema"
            )
        episodes = {
            str(key): value for key, value in self.episodes.items()
        }
        if (
            not episodes
            or any(
                not key
                or not isinstance(
                    value, ExactEpisodeConformanceEvidence
                )
                or value.episode_id != key
                for key, value in episodes.items()
            )
        ):
            raise TypeError(
                "conformance bundle requires typed evidence keyed by episode"
            )
        identities = {
            stable_hash(
                value.conformance_report.to_mapping()["identity"]
            )
            for value in episodes.values()
        }
        if len(identities) != 1:
            raise ValueError(
                "all episode conformance reports must use one exact identity"
            )
        object.__setattr__(
            self,
            "episodes",
            MappingProxyType(dict(sorted(episodes.items()))),
        )
        computed = stable_hash(self._content_payload())
        supplied = str(self.content_sha256).lower()
        if supplied and _sha256(
            supplied, "conformance bundle content_sha256"
        ) != computed:
            raise ValueError(
                "exact episode conformance bundle hash mismatch"
            )
        object.__setattr__(self, "content_sha256", computed)

    @property
    def identity(self) -> ExactOracleIdentity:
        return self.representative_report.identity

    @property
    def representative_report(self) -> ExactOracleConformanceReport:
        first = next(iter(self.episodes.values()))
        return first.conformance_report

    def require_bound(
        self,
        fixtures: Mapping[str, ControllerReplayFixture],
        snapshots: Mapping[str, ControllerSnapshot],
    ) -> None:
        if (
            set(self.episodes) != set(fixtures)
            or set(self.episodes) != set(snapshots)
        ):
            raise ValueError(
                "episode conformance bundle must cover exactly the "
                "controller fixtures and snapshots"
            )
        unbound = tuple(
            episode_id
            for episode_id, evidence in self.episodes.items()
            if not evidence.binds(
                fixtures[episode_id], snapshots[episode_id]
            )
        )
        if unbound:
            raise ValueError(
                "episode conformance evidence is not linked to: {}".format(
                    ", ".join(unbound)
                )
            )

    def _content_payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "episodes": [
                value.to_mapping()
                for value in self.episodes.values()
            ],
        }

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            **self._content_payload(),
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "ExactEpisodeConformanceBundle":
        if not isinstance(values, Mapping):
            raise TypeError(
                "exact episode conformance bundle must be a mapping"
            )
        _exact_keys(
            values,
            required=("schema", "episodes", "content_sha256"),
            label="exact episode conformance bundle",
        )
        raw_episodes = values["episodes"]
        if not isinstance(raw_episodes, list) or not raw_episodes:
            raise ValueError(
                "exact episode conformance bundle requires episodes"
            )
        items = tuple(
            ExactEpisodeConformanceEvidence.from_mapping(item)
            for item in raw_episodes
        )
        episodes = {item.episode_id: item for item in items}
        if len(episodes) != len(items):
            raise ValueError(
                "exact episode conformance bundle repeats an episode"
            )
        return cls(
            episodes=episodes,
            content_sha256=values["content_sha256"],
            schema=values["schema"],
        )


def _hashed_bundle(
    path: Any,
    *,
    schema: str,
    payload_field: str,
    label: str,
) -> Tuple[Mapping[str, Any], str]:
    values = _json_object(path, label)
    _exact_keys(
        values,
        required=("schema", payload_field, "content_sha256"),
        label=label,
    )
    if values["schema"] != schema:
        raise ValueError("unsupported {} schema".format(label))
    digest = _sha256(
        values["content_sha256"], "{}.content_sha256".format(label)
    )
    payload = {
        "schema": values["schema"],
        payload_field: values[payload_field],
    }
    if stable_hash(payload) != digest:
        raise ValueError("{} content hash mismatch".format(label))
    return values, digest


def _event_grid(
    values: Any, expected_name: str, label: str
) -> EventGrid:
    if not isinstance(values, Mapping):
        raise TypeError("{} must be a mapping".format(label))
    _exact_keys(
        values,
        required=("name", "timestamps"),
        label=label,
    )
    if values["name"] != expected_name:
        raise ValueError(
            "{} must be named {}".format(label, expected_name)
        )
    if not isinstance(values["timestamps"], list):
        raise TypeError("{}.timestamps must be a JSON array".format(label))
    return EventGrid(
        name=expected_name,
        timestamps=tuple(values["timestamps"]),
    )


def _episode_grids(values: Any, label: str) -> EpisodeTimeGrids:
    if not isinstance(values, Mapping):
        raise TypeError("{} must be a mapping".format(label))
    _exact_keys(values, required=_GRID_FIELDS, label=label)
    return EpisodeTimeGrids(
        controller_tick_grid=_event_grid(
            values["controller_tick_grid"],
            "controller_tick",
            "{}.controller_tick_grid".format(label),
        ),
        plant_integration_grid=_event_grid(
            values["plant_integration_grid"],
            "plant_integration",
            "{}.plant_integration_grid".format(label),
        ),
        observation_grid=_event_grid(
            values["observation_grid"],
            "observation",
            "{}.observation_grid".format(label),
        ),
        likelihood_grid=_event_grid(
            values["likelihood_grid"],
            "likelihood",
            "{}.likelihood_grid".format(label),
        ),
        report_grid=_event_grid(
            values["report_grid"],
            "report",
            "{}.report_grid".format(label),
        ),
    )


def _controller_input(values: Any, label: str) -> ControllerCoreInput:
    if not isinstance(values, Mapping):
        raise TypeError("{} must be a mapping".format(label))
    try:
        return ControllerCoreInput.from_mapping(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} is not a valid controller input".format(label)) from exc


def _fixture(values: Any, index: int) -> ControllerReplayFixture:
    label = "fixture_bundle.fixtures[{}]".format(index)
    if not isinstance(values, Mapping):
        raise TypeError("{} must be a mapping".format(label))
    _exact_keys(
        values,
        required=(
            "schema",
            "episode_id",
            "source_bag_sha256",
            "topic_inventory_sha256",
            "replay_start_offset_s",
            "score_start_offset_s",
            "score_end_offset_s",
            "factual_controller_tick_grid",
            "grids",
            "controller_inputs",
            "metadata",
            "fixture_sha256",
        ),
        label=label,
    )
    if values["schema"] != CONTROLLER_REPLAY_FIXTURE_SCHEMA:
        raise ValueError("unsupported controller replay fixture schema")
    if not isinstance(values["controller_inputs"], list):
        raise TypeError(
            "{}.controller_inputs must be a JSON array".format(label)
        )
    if not isinstance(values["metadata"], Mapping):
        raise TypeError("{}.metadata must be a mapping".format(label))
    factual_grid = _event_grid(
        values["factual_controller_tick_grid"],
        "factual_controller_tick",
        "{}.factual_controller_tick_grid".format(label),
    )
    fixture = ControllerReplayFixture(
        schema=values["schema"],
        episode_id=values["episode_id"],
        source_bag_sha256=values["source_bag_sha256"],
        topic_inventory_sha256=values["topic_inventory_sha256"],
        replay_start_offset_s=values["replay_start_offset_s"],
        score_start_offset_s=values["score_start_offset_s"],
        score_end_offset_s=values["score_end_offset_s"],
        grids=_episode_grids(
            values["grids"], "{}.grids".format(label)
        ),
        controller_inputs=tuple(
            _controller_input(
                item, "{}.controller_inputs[{}]".format(label, item_index)
            )
            for item_index, item in enumerate(values["controller_inputs"])
        ),
        metadata=values["metadata"],
    )
    if (
        fixture.factual_controller_tick_grid.timestamps
        != factual_grid.timestamps
    ):
        raise ValueError(
            "{} factual controller tick grid does not match its inputs".format(
                label
            )
        )
    expected = _sha256(
        values["fixture_sha256"], "{}.fixture_sha256".format(label)
    )
    if fixture.fixture_sha256 != expected:
        raise ValueError("{} fixture hash mismatch".format(label))
    return fixture


def load_fixture_bundle(
    path: Any,
) -> Tuple[Mapping[str, ControllerReplayFixture], str]:
    """Load a hash-bound collection of replay fixtures."""

    values, bundle_hash = _hashed_bundle(
        path,
        schema=FIXTURE_BUNDLE_SCHEMA,
        payload_field="fixtures",
        label="fixture bundle",
    )
    items = values["fixtures"]
    if not isinstance(items, list) or not items:
        raise ValueError("fixture bundle requires a non-empty fixtures array")
    fixtures: Dict[str, ControllerReplayFixture] = {}
    for index, item in enumerate(items):
        fixture = _fixture(item, index)
        if fixture.episode_id in fixtures:
            raise ValueError(
                "fixture bundle repeats episode {}".format(
                    fixture.episode_id
                )
            )
        fixtures[fixture.episode_id] = fixture
    return MappingProxyType(dict(sorted(fixtures.items()))), bundle_hash


def _snapshot(values: Any, label: str) -> ControllerSnapshot:
    if not isinstance(values, Mapping):
        raise TypeError("{} must be a mapping".format(label))
    expected_fields = (
        "backend_id",
        "source_commit",
        "artifact_sha256",
        "nominal_model_sha256",
        "parameter_dump_sha256",
        "controller_rate_hz",
        "gains",
        "limits",
        "static_options",
        "nominal_mass",
        "nominal_cog",
        "nominal_inertia",
        "nominal_geometry",
        "schema",
    )
    _exact_keys(values, required=expected_fields, label=label)
    if values["schema"] != CONTROLLER_SNAPSHOT_SCHEMA:
        raise ValueError("unsupported controller snapshot schema")
    try:
        return ControllerSnapshot.from_mapping(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "{} is not a valid controller snapshot".format(label)
        ) from exc


def load_snapshot_bundle(
    path: Any,
) -> Tuple[Mapping[str, ControllerSnapshot], str]:
    """Load per-episode immutable controller snapshots."""

    values, bundle_hash = _hashed_bundle(
        path,
        schema=SNAPSHOT_BUNDLE_SCHEMA,
        payload_field="snapshots",
        label="snapshot bundle",
    )
    items = values["snapshots"]
    if not isinstance(items, list) or not items:
        raise ValueError("snapshot bundle requires a non-empty snapshots array")
    snapshots: Dict[str, ControllerSnapshot] = {}
    for index, item in enumerate(items):
        label = "snapshot_bundle.snapshots[{}]".format(index)
        if not isinstance(item, Mapping):
            raise TypeError("{} must be a mapping".format(label))
        _exact_keys(
            item,
            required=("episode_id", "snapshot", "snapshot_sha256"),
            label=label,
        )
        episode_id = str(item["episode_id"]).strip()
        if not episode_id or episode_id in snapshots:
            raise ValueError(
                "{} has an empty or repeated episode_id".format(label)
            )
        snapshot = _snapshot(
            item["snapshot"], "{}.snapshot".format(label)
        )
        expected = _sha256(
            item["snapshot_sha256"],
            "{}.snapshot_sha256".format(label),
        )
        if snapshot.snapshot_id != expected:
            raise ValueError("{} snapshot hash mismatch".format(label))
        snapshots[episode_id] = snapshot
    return MappingProxyType(dict(sorted(snapshots.items()))), bundle_hash


@dataclass(frozen=True)
class ControllerStateSelection:
    """One explicit episode state or states keyed by nuisance sample ID."""

    shared_state: Optional[ControllerCoreState] = None
    sample_states: Mapping[str, ControllerCoreState] = MappingProxyType({})

    def __post_init__(self) -> None:
        shared = self.shared_state
        samples = {
            str(key): value for key, value in self.sample_states.items()
        }
        if (shared is None) == (not samples):
            raise ValueError(
                "state selection requires exactly one shared state or "
                "sample_states mapping"
            )
        if shared is not None and not isinstance(
            shared, ControllerCoreState
        ):
            raise TypeError("shared controller state has the wrong type")
        if any(
            not key or not isinstance(value, ControllerCoreState)
            for key, value in samples.items()
        ):
            raise TypeError(
                "sample controller states require non-empty IDs and typed states"
            )
        object.__setattr__(
            self,
            "sample_states",
            MappingProxyType(dict(sorted(samples.items()))),
        )

    def state_for(self, state_sample_id: str) -> ControllerCoreState:
        if self.shared_state is not None:
            return self.shared_state
        try:
            return self.sample_states[str(state_sample_id)]
        except KeyError as exc:
            raise ValueError(
                "controller-state bundle lacks nuisance sample {}".format(
                    state_sample_id
                )
            ) from exc


def _core_state(values: Any, label: str) -> ControllerCoreState:
    if not isinstance(values, Mapping):
        raise TypeError("{} must be a mapping".format(label))
    try:
        return ControllerCoreState.from_mapping(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "{} is not a valid controller state".format(label)
        ) from exc


def load_controller_state_bundle(
    path: Any,
) -> Tuple[Mapping[str, ControllerStateSelection], str]:
    """Load explicit controller state for every episode or nuisance sample."""

    values, bundle_hash = _hashed_bundle(
        path,
        schema=STATE_BUNDLE_SCHEMA,
        payload_field="episodes",
        label="controller-state bundle",
    )
    items = values["episodes"]
    if not isinstance(items, list) or not items:
        raise ValueError(
            "controller-state bundle requires a non-empty episodes array"
        )
    selections: Dict[str, ControllerStateSelection] = {}
    for index, item in enumerate(items):
        label = "controller_state_bundle.episodes[{}]".format(index)
        if not isinstance(item, Mapping):
            raise TypeError("{} must be a mapping".format(label))
        keys = set(item)
        shared = keys == {"episode_id", "controller_state"}
        per_sample = keys == {"episode_id", "sample_states"}
        if not shared and not per_sample:
            raise ValueError(
                "{} requires exactly episode_id plus controller_state or "
                "sample_states".format(label)
            )
        episode_id = str(item["episode_id"]).strip()
        if not episode_id or episode_id in selections:
            raise ValueError(
                "{} has an empty or repeated episode_id".format(label)
            )
        if shared:
            selection = ControllerStateSelection(
                shared_state=_core_state(
                    item["controller_state"],
                    "{}.controller_state".format(label),
                )
            )
        else:
            records = item["sample_states"]
            if not isinstance(records, list) or not records:
                raise ValueError(
                    "{}.sample_states must be a non-empty array".format(
                        label
                    )
                )
            sample_states: Dict[str, ControllerCoreState] = {}
            for sample_index, record in enumerate(records):
                record_label = "{}.sample_states[{}]".format(
                    label, sample_index
                )
                if not isinstance(record, Mapping):
                    raise TypeError(
                        "{} must be a mapping".format(record_label)
                    )
                _exact_keys(
                    record,
                    required=("state_sample_id", "controller_state"),
                    label=record_label,
                )
                sample_id = str(record["state_sample_id"]).strip()
                if not sample_id or sample_id in sample_states:
                    raise ValueError(
                        "{} has an empty or repeated sample ID".format(
                            record_label
                        )
                    )
                sample_states[sample_id] = _core_state(
                    record["controller_state"],
                    "{}.controller_state".format(record_label),
                )
            selection = ControllerStateSelection(
                sample_states=sample_states
            )
        selections[episode_id] = selection
    return MappingProxyType(dict(sorted(selections.items()))), bundle_hash


def load_conformance_report(path: Any) -> ExactOracleConformanceReport:
    """Load typed, internally hash-bound factual conformance evidence."""

    values = _json_object(path, "factual conformance report")
    loader = getattr(ExactOracleConformanceReport, "from_mapping", None)
    if not callable(loader):
        raise RuntimeError(
            "this build lacks the typed conformance-report loader"
        )
    try:
        report = loader(values)
    except (
        ExactOracleProtocolError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "factual conformance report failed schema/hash validation"
        ) from exc
    if not isinstance(report, ExactOracleConformanceReport):
        raise TypeError(
            "factual conformance loader did not return its typed contract"
        )
    serialized = report.to_mapping()
    if (
        set(values) != set(serialized)
        or stable_hash(values) != stable_hash(serialized)
    ):
        raise ValueError(
            "factual conformance report is not its strict canonical schema"
        )
    return report


def load_episode_conformance_bundle(
    path: Any,
    fixtures: Mapping[str, ControllerReplayFixture],
    snapshots: Mapping[str, ControllerSnapshot],
) -> ExactEpisodeConformanceBundle:
    """Load and mechanically bind every factual report to its episode."""

    values = _json_object(path, "exact episode conformance bundle")
    try:
        bundle = ExactEpisodeConformanceBundle.from_mapping(values)
        bundle.require_bound(fixtures, snapshots)
    except (
        ExactOracleProtocolError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "exact episode conformance bundle failed schema/hash/input "
            "linkage validation"
        ) from exc
    serialized = bundle.to_mapping()
    if (
        set(values) != set(serialized)
        or stable_hash(values) != stable_hash(serialized)
    ):
        raise ValueError(
            "exact episode conformance bundle is not canonical"
        )
    return bundle


def inject_controller_states(
    prepared: Mapping[str, Any],
    selections: Mapping[str, ControllerStateSelection],
    state_bundle_sha256: str,
) -> Mapping[str, Any]:
    """Inject state and bind each nuisance cache ID to that exact evidence."""

    episode_ids = set(str(key) for key in prepared)
    bundle_hash = _sha256(
        state_bundle_sha256, "controller-state bundle hash"
    )
    if not episode_ids or set(selections) != episode_ids:
        raise ValueError(
            "controller-state bundle must cover exactly the prepared episodes"
        )
    result: Dict[str, Any] = {}
    for episode_id, episode in prepared.items():
        nuisance_samples = tuple(episode.nuisance_samples)
        if not nuisance_samples:
            raise ValueError(
                "episode {} has no nuisance samples".format(episode_id)
            )
        selection = selections[episode_id]
        sample_ids = tuple(
            str(item.state_sample_id) for item in nuisance_samples
        )
        if selection.shared_state is None and set(
            selection.sample_states
        ) != set(sample_ids):
            raise ValueError(
                "episode {} sample-specific states do not exactly match "
                "the nuisance posterior".format(episode_id)
            )
        replaced_items = []
        for item in nuisance_samples:
            controller_state = selection.state_for(
                item.state_sample_id
            )
            bound_id = stable_hash(
                {
                    "schema": "grape.controller-state-bound-nuisance/v1",
                    "original_state_sample_id": item.state_sample_id,
                    "controller_state_sha256": (
                        controller_state.content_sha256
                    ),
                    "controller_state_bundle_sha256": bundle_hash,
                }
            )
            replaced_items.append(
                replace(
                    item,
                    controller_state=controller_state,
                    state_sample_id=bound_id,
                )
            )
        replaced = tuple(replaced_items)
        if any(
            not isinstance(item.controller_state, ControllerCoreState)
            for item in replaced
        ):
            raise RuntimeError(
                "controller-state injection failed for {}".format(
                    episode_id
                )
            )
        result[episode_id] = replace(
            episode, nuisance_samples=replaced
        )
    return MappingProxyType(dict(sorted(result.items())))


def require_episode_alignment(
    prepared: Mapping[str, Any],
    fixtures: Mapping[str, ControllerReplayFixture],
    snapshots: Mapping[str, ControllerSnapshot],
    states: Mapping[str, ControllerStateSelection],
) -> None:
    """Reject partial or surplus exact-controller episode inputs."""

    expected = set(str(key) for key in prepared)
    if (
        not expected
        or set(fixtures) != expected
        or set(snapshots) != expected
        or set(states) != expected
    ):
        raise ValueError(
            "fixture, snapshot, and controller-state bundles must each "
            "cover exactly the prepared episodes"
        )


__all__ = [
    "CONTROLLER_SNAPSHOT_SCHEMA",
    "ControllerStateSelection",
    "EXACT_EPISODE_CONFORMANCE_BUNDLE_SCHEMA",
    "EXACT_EPISODE_CONFORMANCE_EVIDENCE_SCHEMA",
    "EXACT_EPISODE_CONTROLLER_INPUT_SCHEMA",
    "EXACT_EPISODE_REQUEST_BINDING_SCHEMA",
    "ExactEpisodeConformanceBundle",
    "ExactEpisodeConformanceEvidence",
    "FIXTURE_BUNDLE_SCHEMA",
    "SNAPSHOT_BUNDLE_SCHEMA",
    "STATE_BUNDLE_SCHEMA",
    "inject_controller_states",
    "load_conformance_report",
    "load_episode_conformance_bundle",
    "load_controller_state_bundle",
    "load_fixture_bundle",
    "load_snapshot_bundle",
    "require_episode_alignment",
]
