"""Content-addressed rollout cache keys and a bounded in-memory cache."""

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any, Optional

from grape_param_estim.episode import stable_hash


@dataclass(frozen=True)
class RolloutCacheKey:
    source_bag_sha256: str
    normalized_episode_sha256: str
    controller_snapshot_sha256: str
    controller_artifact_sha256: str
    plant_backend_model_id: str
    parameter_vector_sha256: str
    initial_state_sample_id: str
    process_noise_seed: int
    source_commit: str

    def __post_init__(self) -> None:
        required = (
            "source_bag_sha256",
            "normalized_episode_sha256",
            "controller_snapshot_sha256",
            "controller_artifact_sha256",
            "plant_backend_model_id",
            "parameter_vector_sha256",
            "initial_state_sample_id",
            "source_commit",
        )
        for name in required:
            if not str(getattr(self, name)):
                raise ValueError("{} is required in a rollout cache key".format(name))

    @property
    def digest(self) -> str:
        return stable_hash(
            {
                "source_bag_sha256": self.source_bag_sha256,
                "normalized_episode_sha256": self.normalized_episode_sha256,
                "controller_snapshot_sha256": self.controller_snapshot_sha256,
                "controller_artifact_sha256": self.controller_artifact_sha256,
                "plant_backend_model_id": self.plant_backend_model_id,
                "parameter_vector_sha256": self.parameter_vector_sha256,
                "initial_state_sample_id": self.initial_state_sample_id,
                "process_noise_seed": int(self.process_noise_seed),
                "source_commit": self.source_commit,
            }
        )


class RolloutCache:
    """Thread-safe LRU cache; cached values never alter inference semantics."""

    def __init__(self, maximum_entries: int = 4096) -> None:
        maximum = int(maximum_entries)
        if maximum < 1:
            raise ValueError("maximum_entries must be positive")
        self.maximum_entries = maximum
        self._values = OrderedDict()
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: RolloutCacheKey) -> Optional[Any]:
        if not isinstance(key, RolloutCacheKey):
            raise TypeError("cache key must be RolloutCacheKey")
        digest = key.digest
        with self._lock:
            if digest not in self._values:
                self.misses += 1
                return None
            self.hits += 1
            value = self._values.pop(digest)
            self._values[digest] = value
            return value

    def put(self, key: RolloutCacheKey, value: Any) -> None:
        if not isinstance(key, RolloutCacheKey):
            raise TypeError("cache key must be RolloutCacheKey")
        digest = key.digest
        with self._lock:
            self._values.pop(digest, None)
            self._values[digest] = value
            while len(self._values) > self.maximum_entries:
                self._values.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)


__all__ = ["RolloutCache", "RolloutCacheKey"]
