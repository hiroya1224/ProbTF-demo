"""Atomic, pickle-free checkpoints for completed PID forecast batches."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import numpy as np

from grape_param_estim.artifact_io import (
    ArtifactValidationError,
    load_npz_strict,
    read_json,
    write_json_atomic,
    write_npz_atomic,
)
from grape_param_estim.pid.metrics import ForecastMetricRecord, ForecastMetrics


PID_FORECAST_CHECKPOINT_SCHEMA = "grape-param-estim/pid-forecast-checkpoint/v1"
_CHECKPOINT_STATUS = "checkpoint"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_MANIFEST_KEYS = (
    "schema",
    "status",
    "evaluation_id",
    "estimation_run_id",
    "request_fingerprint",
    "estimation_request_fingerprint",
    "record_count",
    "record_batch",
)
_RECORD_KEYS = (
    "candidate_id",
    "sample_id",
    "bag_id",
    "replicate_index",
    "discrepancy_seed",
    "position_rmse",
    "orientation_rmse",
    "maximum_position_error",
    "maximum_orientation_error",
    "forecast_completion",
    "numerical_failure_count",
    "actuator_saturation_duration",
    "actuator_saturation_rate",
)


def _canonical(value: object, name: str) -> str:
    selected = str(value)
    if not selected or selected.strip() != selected or "\x00" in selected:
        raise ArtifactValidationError(
            "{} must be a canonical non-empty string".format(name)
        )
    return selected


def _fingerprint(value: object, name: str) -> str:
    selected = _canonical(value, name)
    if _SHA256.fullmatch(selected) is None:
        raise ArtifactValidationError(
            "{} must have form sha256:<64 lowercase hex>".format(name)
        )
    return selected


def _strict_keys(
    value: Mapping[str, Any], expected: Sequence[str], location: str
) -> None:
    missing = sorted(set(expected) - set(value))
    unknown = sorted(set(value) - set(expected))
    if missing or unknown:
        raise ArtifactValidationError(
            "{} keys disagree; missing={}, unknown={}".format(
                location, missing, unknown
            )
        )


@dataclass(frozen=True)
class PidForecastCheckpointIdentity:
    evaluation_id: str
    estimation_run_id: str
    request_fingerprint: str
    estimation_request_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evaluation_id", _canonical(self.evaluation_id, "evaluation_id")
        )
        object.__setattr__(
            self,
            "estimation_run_id",
            _canonical(self.estimation_run_id, "estimation_run_id"),
        )
        for name in ("request_fingerprint", "estimation_request_fingerprint"):
            object.__setattr__(
                self, name, _fingerprint(getattr(self, name), name)
            )


def checkpoint_root_for_output(output_directory: Union[str, Path]) -> Path:
    output = Path(output_directory).expanduser().resolve()
    return output.parent / ".{}.pid-forecast-checkpoint".format(output.name)


def _record_arrays(records: Sequence[ForecastMetricRecord]) -> Dict[str, np.ndarray]:
    selected = tuple(records)
    if not selected or any(
        not isinstance(value, ForecastMetricRecord) for value in selected
    ):
        raise ArtifactValidationError(
            "checkpoint record batch must contain forecast records"
        )
    metrics = tuple(value.metrics for value in selected)
    return {
        "candidate_id": np.asarray(
            tuple(value.candidate_id for value in selected), dtype=np.str_
        ),
        "sample_id": np.asarray(
            tuple(value.sample_id for value in selected), dtype=np.str_
        ),
        "bag_id": np.asarray(
            tuple(value.bag_id for value in selected), dtype=np.str_
        ),
        "replicate_index": np.asarray(
            tuple(value.replicate_index for value in selected), dtype=np.int64
        ),
        "discrepancy_seed": np.asarray(
            tuple(value.discrepancy_seed for value in selected), dtype=np.uint64
        ),
        "position_rmse": np.asarray(
            tuple(value.position_rmse for value in metrics), dtype=np.float64
        ),
        "orientation_rmse": np.asarray(
            tuple(value.orientation_rmse for value in metrics), dtype=np.float64
        ),
        "maximum_position_error": np.asarray(
            tuple(value.maximum_position_error for value in metrics),
            dtype=np.float64,
        ),
        "maximum_orientation_error": np.asarray(
            tuple(value.maximum_orientation_error for value in metrics),
            dtype=np.float64,
        ),
        "forecast_completion": np.asarray(
            tuple(value.forecast_completion for value in metrics), dtype=np.float64
        ),
        "numerical_failure_count": np.asarray(
            tuple(value.numerical_failure_count for value in metrics),
            dtype=np.int64,
        ),
        "actuator_saturation_duration": np.asarray(
            tuple(value.actuator_saturation_duration for value in metrics),
            dtype=np.float64,
        ),
        "actuator_saturation_rate": np.asarray(
            tuple(value.actuator_saturation_rate for value in metrics),
            dtype=np.float64,
        ),
    }


def _payload_digest(arrays: Mapping[str, np.ndarray]) -> str:
    _strict_keys(arrays, _RECORD_KEYS, "checkpoint record batch")
    digest = hashlib.sha256()
    for key in _RECORD_KEYS:
        value = np.asarray(arrays[key])
        if value.dtype.hasobject:
            raise ArtifactValidationError("checkpoint arrays cannot use object dtype")
        digest.update(key.encode("utf-8") + b"\x00")
        digest.update(value.dtype.str.encode("ascii") + b"\x00")
        digest.update(",".join(str(item) for item in value.shape).encode("ascii"))
        digest.update(b"\x00")
        digest.update(np.ascontiguousarray(value).tobytes())
    return "sha256:" + digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _records_from_arrays(
    arrays: Mapping[str, np.ndarray], expected_count: int
) -> Tuple[ForecastMetricRecord, ...]:
    _strict_keys(arrays, _RECORD_KEYS, "checkpoint record batch")
    count = int(expected_count)
    if count < 1:
        raise ArtifactValidationError("checkpoint record count must be positive")
    strings = {}
    for key in ("candidate_id", "sample_id", "bag_id"):
        value = np.asarray(arrays[key])
        if value.shape != (count,) or value.dtype.kind not in "US":
            raise ArtifactValidationError("checkpoint {} is invalid".format(key))
        strings[key] = value.astype(str)
    numeric = {}
    integer_keys = (
        "replicate_index",
        "discrepancy_seed",
        "numerical_failure_count",
    )
    for key in _RECORD_KEYS[3:]:
        value = np.asarray(arrays[key])
        if (
            value.shape != (count,)
            or value.dtype.hasobject
            or not np.issubdtype(value.dtype, np.number)
            or np.any(~np.isfinite(value))
            or (key in integer_keys and not np.issubdtype(value.dtype, np.integer))
        ):
            raise ArtifactValidationError("checkpoint {} is invalid".format(key))
        numeric[key] = value
    result = []
    for index in range(count):
        result.append(
            ForecastMetricRecord(
                candidate_id=str(strings["candidate_id"][index]),
                sample_id=str(strings["sample_id"][index]),
                bag_id=str(strings["bag_id"][index]),
                replicate_index=int(numeric["replicate_index"][index]),
                discrepancy_seed=int(numeric["discrepancy_seed"][index]),
                metrics=ForecastMetrics(
                    position_rmse=float(numeric["position_rmse"][index]),
                    orientation_rmse=float(numeric["orientation_rmse"][index]),
                    maximum_position_error=float(
                        numeric["maximum_position_error"][index]
                    ),
                    maximum_orientation_error=float(
                        numeric["maximum_orientation_error"][index]
                    ),
                    forecast_completion=float(
                        numeric["forecast_completion"][index]
                    ),
                    numerical_failure_count=int(
                        numeric["numerical_failure_count"][index]
                    ),
                    actuator_saturation_duration=float(
                        numeric["actuator_saturation_duration"][index]
                    ),
                    actuator_saturation_rate=float(
                        numeric["actuator_saturation_rate"][index]
                    ),
                ),
            )
        )
    identities = {
        (
            value.candidate_id,
            value.sample_id,
            value.bag_id,
            value.replicate_index,
        )
        for value in result
    }
    if len(identities) != len(result):
        raise ArtifactValidationError("checkpoint contains duplicate forecast records")
    return tuple(result)


def _manifest(
    identity: PidForecastCheckpointIdentity,
    record_count: int,
    descriptor: object,
) -> Dict[str, Any]:
    return {
        "schema": PID_FORECAST_CHECKPOINT_SCHEMA,
        "status": _CHECKPOINT_STATUS,
        "evaluation_id": identity.evaluation_id,
        "estimation_run_id": identity.estimation_run_id,
        "request_fingerprint": identity.request_fingerprint,
        "estimation_request_fingerprint": identity.estimation_request_fingerprint,
        "record_count": int(record_count),
        "record_batch": descriptor,
    }


class PidForecastCheckpointStore:
    """Mutable checkpoint index whose record payloads are immutable objects."""

    def __init__(
        self,
        root: Path,
        identity: PidForecastCheckpointIdentity,
        records: Sequence[ForecastMetricRecord],
        flush_size: int,
    ) -> None:
        self.root = Path(root).resolve()
        self.identity = identity
        self._records = list(records)
        self._pending = []
        self._identities = {
            (
                value.candidate_id,
                value.sample_id,
                value.bag_id,
                value.replicate_index,
            )
            for value in self._records
        }
        if (
            isinstance(flush_size, (bool, np.bool_))
            or not isinstance(flush_size, (int, np.integer))
            or flush_size < 1
        ):
            raise ValueError("checkpoint flush_size must be positive")
        self.flush_size = int(flush_size)
        self.resumed_record_count = len(self._records)

    @property
    def records(self) -> Tuple[ForecastMetricRecord, ...]:
        return tuple(self._records) + tuple(self._pending)

    @classmethod
    def open(
        cls,
        root: Union[str, Path],
        identity: PidForecastCheckpointIdentity,
        *,
        resume: bool,
        flush_size: int = 8,
    ) -> "PidForecastCheckpointStore":
        if not isinstance(identity, PidForecastCheckpointIdentity):
            raise TypeError("checkpoint identity has the wrong type")
        selected = Path(root).expanduser().resolve()
        if resume:
            if not selected.is_dir() or selected.is_symlink():
                raise ArtifactValidationError(
                    "PID forecast checkpoint does not exist for resume"
                )
            manifest = read_json(selected / "manifest.json")
            _strict_keys(manifest, _MANIFEST_KEYS, "checkpoint manifest")
            if (
                manifest["schema"] != PID_FORECAST_CHECKPOINT_SCHEMA
                or manifest["status"] != _CHECKPOINT_STATUS
            ):
                raise ArtifactValidationError("PID forecast checkpoint schema is invalid")
            count = manifest["record_count"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ArtifactValidationError("checkpoint record_count is invalid")
            expected = _manifest(identity, count, manifest["record_batch"])
            for key in (
                "evaluation_id",
                "estimation_run_id",
                "request_fingerprint",
                "estimation_request_fingerprint",
            ):
                if manifest[key] != expected[key]:
                    raise ArtifactValidationError(
                        "PID forecast checkpoint {} mismatch".format(key)
                    )
            descriptor = manifest["record_batch"]
            if count == 0:
                if descriptor is not None:
                    raise ArtifactValidationError(
                        "empty checkpoint must not name a record batch"
                    )
                records = tuple()
            else:
                if not isinstance(descriptor, Mapping):
                    raise ArtifactValidationError(
                        "checkpoint record_batch must be an object"
                    )
                _strict_keys(
                    descriptor, ("path", "content_sha256"), "record_batch"
                )
                digest = _fingerprint(
                    descriptor["content_sha256"], "record_batch.content_sha256"
                )
                expected_path = "objects/{}.npz".format(digest[7:])
                if descriptor["path"] != expected_path:
                    raise ArtifactValidationError(
                        "checkpoint record batch is not content-addressed"
                    )
                path = selected / expected_path
                if not path.is_file():
                    raise ArtifactValidationError(
                        "checkpoint record batch is missing"
                    )
                if _file_digest(path) != digest:
                    raise ArtifactValidationError(
                        "checkpoint record batch content digest disagrees"
                    )
                arrays = load_npz_strict(path)
                _payload_digest(arrays)
                records = _records_from_arrays(arrays, count)
            return cls(selected, identity, records, flush_size)
        if selected.exists() or selected.is_symlink():
            raise ArtifactValidationError(
                "PID forecast checkpoint already exists; use resume"
            )
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.mkdir()
        (selected / "objects").mkdir()
        write_json_atomic(selected / "manifest.json", _manifest(identity, 0, None))
        return cls(selected, identity, tuple(), flush_size)

    def record_completed(self, record: ForecastMetricRecord) -> None:
        if not isinstance(record, ForecastMetricRecord):
            raise TypeError("checkpoint accepts ForecastMetricRecord values")
        identity = (
            record.candidate_id,
            record.sample_id,
            record.bag_id,
            record.replicate_index,
        )
        if identity in self._identities:
            raise ArtifactValidationError(
                "checkpoint forecast identity was completed twice"
            )
        self._identities.add(identity)
        self._pending.append(record)
        if len(self._pending) >= self.flush_size:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        records = tuple(self._records) + tuple(self._pending)
        arrays = _record_arrays(records)
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=".records-", suffix=".npz", dir=str(self.root / "objects")
        )
        os.close(temporary_fd)
        os.unlink(temporary_name)
        temporary = Path(temporary_name)
        try:
            write_npz_atomic(temporary, arrays)
            digest = _file_digest(temporary)
            relative = Path("objects") / "{}.npz".format(digest[7:])
            destination = self.root / relative
            if destination.exists():
                if _file_digest(destination) != digest:
                    raise ArtifactValidationError(
                        "content-addressed checkpoint object is corrupt"
                    )
                temporary.unlink()
            else:
                os.replace(str(temporary), str(destination))
            descriptor = {
                "path": relative.as_posix(),
                "content_sha256": digest,
            }
            write_json_atomic(
                self.root / "manifest.json",
                _manifest(self.identity, len(records), descriptor),
            )
            self._records = list(records)
            self._pending = []
        finally:
            if temporary.exists():
                temporary.unlink()

    def discard(self) -> None:
        if self.root.is_dir() and not self.root.is_symlink():
            shutil.rmtree(str(self.root))


__all__ = [
    "PID_FORECAST_CHECKPOINT_SCHEMA",
    "PidForecastCheckpointIdentity",
    "PidForecastCheckpointStore",
    "checkpoint_root_for_output",
]
